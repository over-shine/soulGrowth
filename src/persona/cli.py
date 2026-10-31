"""命令行入口。

用法：
    persona birth                         出生（写第一条事件与初始特质）
    persona day --day 0                   过一天
    persona run --days 30                 连跑 30 天
    persona live --pace 60                按真实时间一格格往下走
    persona body --port 8767              身体服务（另一个进程），core 用 --body ws://… 连它
    persona verify                        体检（契约 + 事件日志）
    persona why --episode ep-...          追溯一条 episode 的来龙去脉
    persona report                        今天的画像
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from persona import __version__
from persona.config import Config, load as load_config
from persona.errors import PersonaError
from persona.log import log_path_for, reconfigure_streams, setup as setup_logging
from persona.runtime.runner import Simulation, open_simulation
from persona.runtime.scheduler import LiveScheduler, describe_ts


def _cfg(paths: Sequence[str]) -> Config:
    return load_config(list(paths) or ["config/default.json"])


def _open(args: argparse.Namespace) -> Simulation:
    """载入配置、拉起模拟，并把这次运行的日志钉在 data/<个体>/logs/persona.log。"""
    cfg = _cfg(args.config)
    if getattr(args, "force", False):
        # 同一天重跑是幂等的；要覆盖已有的一天必须显式 --force
        cfg.data.setdefault("runtime", {})["force_rerun"] = True
    if getattr(args, "no_log", False):
        setup_logging(None, level=_level(args))
    else:
        setup_logging(log_path_for(cfg.str_("store.path")), level=_level(args))
    sim = open_simulation(cfg)
    url = getattr(args, "body", None)
    if url:
        # 身体在别的进程里（docs/02 的 L1）。core 不知道自己面对的是本地对象还是网线。
        from persona.transport.body_io import RemoteBody

        try:
            sim.attach_body(RemoteBody(str(url)))
        except PersonaError:
            sim.close()
            raise
    return sim


def _level(args: argparse.Namespace) -> int:
    import logging

    name = str(getattr(args, "log_level", "INFO") or "INFO").upper()
    return int(getattr(logging, name, logging.INFO))


def _print(obj: object) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_birth(args: argparse.Namespace) -> int:
    sim = _open(args)
    try:
        first = sim.born()
        print(f"{'出生' if first else '已经活着'}：{sim.individual_id}，阶段 {sim.stage}")
        return 0
    finally:
        sim.close()


def cmd_day(args: argparse.Namespace) -> int:
    sim = _open(args)
    try:
        res = sim.run_day(int(args.day))
        print(
            f"第 {res.day} 天（{res.date}）：{res.episodes} 条 episode，"
            f"{res.degraded} 条 degraded，阶段 {sim.stage}"
        )
        if res.advanced:
            print(f"阶段推进 → {res.advanced}")
        for m in res.milestones:
            print(f"  {m['id']} {m['status']:12s} value={m['value']} threshold={m['threshold']}")
        print(f"身体：{res.body}")
        print(f"特质：{res.traits.get('T')}")
        return 0
    finally:
        sim.close()


def cmd_run(args: argparse.Namespace) -> int:
    sim = _open(args)
    try:
        start = int(args.start)
        for day in range(start, start + int(args.days)):
            res = sim.run_day(day)
            note = f" → {res.advanced}" if res.advanced else ""
            print(f"day {day:3d} {res.date} eps={res.episodes:2d} degraded={res.degraded:2d}{note}")
        return 0
    finally:
        sim.close()


def cmd_live(args: argparse.Namespace) -> int:
    """按真实时间一格格往下走，而不是把一天压进一秒。"""
    sim = _open(args)
    stats = None
    sched = LiveScheduler(
        sim,
        pace=float(args.pace),
        tick_s=float(args.tick),
        on_episode=None if args.quiet else (lambda ep: _live_line(ep)),
        on_day=None if args.quiet else (lambda res: _live_day(res)),
    )
    try:
        print(
            f"{sim.individual_id} 开始 live：pace={args.pace}（1 真实秒 = {args.pace} 模拟秒），"
            f"一个 30 分钟的格子 = {sched.slot_real_s:.1f} 秒，Ctrl+C 停"
        )
        stats = sched.run(
            until_day=None if args.until_day < 0 else int(args.until_day),
            max_days=None if int(args.days) <= 0 else int(args.days),
        )
        return 0
    except KeyboardInterrupt:
        sched.stop()
        print("\n停下了。这一格已经落盘，下次 live 会接着走。")
        return 0
    finally:
        sim.close()
        if stats is not None:
            print(
                f"live 结束：day {stats.to_dict()['day']}，{stats.episodes} 条 episode，"
                f"{stats.catch_up} 条补做，用时 {stats.to_dict()['elapsed_s']:.1f}s"
            )


def _live_line(ep: object) -> None:
    action = getattr(ep, "action", None)
    prim = getattr(action, "primitive", "?")
    target = getattr(action, "target", None)
    lang = getattr(ep, "language", None)
    utter = getattr(lang, "utterance", None)
    ts = describe_ts(getattr(ep, "ts", ""))
    tail = f" {target}" if target else ""
    if utter:
        tail += f" 「{utter}」"
    print(f"{ts}  {prim:<12s}{tail}")


def _live_day(res: object) -> None:
    print(
        f"—— day {getattr(res, 'day', '?')} 结束，"
        f"{getattr(res, 'episodes', 0)} 条 episode"
    )


def cmd_verify(args: argparse.Namespace) -> int:
    sim = _open(args)
    try:
        problems = sim.repo.verify(max_per_day=int(args.max_per_day))
        if problems:
            print(f"发现 {len(problems)} 个问题：")
            for p in problems:
                print(f"  - {p}")
            return 1
        print(
            f"体检通过：{sim.db.count('episodes')} 条 episode、"
            f"{sim.db.count('memories')} 条记忆、{sim.db.count('events')} 条事件"
        )
        return 0
    finally:
        sim.close()


def cmd_why(args: argparse.Namespace) -> int:
    sim = _open(args)
    try:
        ep = sim.repo.get_episode(str(args.episode))
        if ep is None:
            print(f"没有这条 episode：{args.episode}", file=sys.stderr)
            return 2
        _print(_trace(sim, ep))
        return 0
    finally:
        sim.close()


def _trace(sim: Simulation, ep: dict) -> dict:
    """把一条 episode 还原成可读的因果链。证据全部指向库内真实记录。"""
    from persona.observe.trace import trace_episode

    return trace_episode(ep, _unlocked(sim, str(ep.get("stage"))))


def _unlocked(sim: Simulation, stage: str) -> set[str]:
    return {p.id for p in sim.registry.for_stage(stage)} | {"A0.NONE"}


def cmd_report(args: argparse.Namespace) -> int:
    sim = _open(args)
    try:
        date = str(args.date) if args.date else sim.repo.last_episode_ts()
        if not date:
            print("还没有任何记录", file=sys.stderr)
            return 2
        eps = sim.repo.episodes_of_day(date[:10])
        trait = sim.repo.latest_trait() or {}
        _print(
            {
                "date": date[:10],
                "stage": sim.stage,
                "episodes": len(eps),
                "kind": {"awake": sum(1 for e in eps if e.get("kind") == "awake"), "sleep": sum(1 for e in eps if e.get("kind") == "sleep")},
                "actions": _counts([str(e.get("action", {}).get("primitive")) for e in eps]),
                "utterances": [e["language"]["utterance"] for e in eps if e.get("language", {}).get("utterance")],
                "salient": [e.get("observation", {}).get("summary") for e in eps[:3]],
                "traits": trait.get("T"),
                "milestones": [
                    {"id": m["id"], "status": m["status"], "value": m["value"]}
                    for m in sim.repo.milestones_of_stage(sim.stage)
                ],
            }
        )
        return 0
    finally:
        sim.close()


def cmd_diary(args: argparse.Namespace) -> int:
    """把一天（或整段）写成 Markdown 日记。观察者视角，不是他自己写的。"""
    from persona.observe.diary import render_day, write_day

    sim = _open(args)
    try:
        if args.day is not None:
            days = [int(args.day)]
        else:
            days = list(range(0, max(1, int(args.days))))
        # 里程碑与特质是「到目前为止」的概念，库里只存最新一份。所以只有最后
        # 一天能挂这两块；更早的日子硬挂上去，就等于拿 30 天后的判定冒充当天的。
        pending: list[tuple[int, str, list[dict]]] = []
        for day in days:
            result_date = sim._date_of(day).isoformat()
            episodes = sim.repo.episodes_of_day(result_date)
            if episodes:
                pending.append((day, result_date, episodes))
        written = []
        texts = []
        all_memories = sim.repo.list_memories(limit=1000)
        for idx, (day, result_date, episodes) in enumerate(pending):
            last = idx == len(pending) - 1
            day_stage = str(episodes[-1].get("stage", sim.stage))
            text = render_day(
                day=day,
                date=result_date,
                stage=day_stage,
                episodes=episodes,
                milestones=sim.repo.milestones_of_stage(day_stage) if last else (),
                traits=(sim.repo.latest_trait() or None) if last else None,
                memories=[m for m in all_memories if str(m.get("created", ""))[:10] == result_date],
                name=sim.cfg.str_("individual.name", "小满"),
            )
            path = Path(sim.cfg.str_("store.path")).parent / "diary" / f"{result_date}.md"
            if path.exists() and not bool(getattr(args, "force", False)):
                # 跑批时写下的日记带着「那一天的」里程碑与特质，比事后重建更准。
                # 除非 --force，否则不覆盖它。
                continue
            write_day(path, text)
            written.append(str(path))
            texts.append(text)
        for text in texts[-1:]:
            print(text)
        print(f"写了 {len(written)} 天的日记", file=sys.stderr)
        return 0
    finally:
        sim.close()


def _counts(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[it] = out.get(it, 0) + 1
    return dict(sorted(out.items()))


def cmd_watch(args: argparse.Namespace) -> int:
    """只读观察台：独立进程，不打开 Simulation，也不会写任何东西。"""
    from persona.observe.server import Observer, WatchServer

    cfg = _cfg(args.config)
    observer = Observer(cfg)
    server = WatchServer(observer, host=str(args.host), port=int(args.port))
    url = server.url
    print(f"观察台已开：{url}")
    print(f"数据：{observer.db_path}", file=sys.stderr)
    print("只读、只回答 GET；Ctrl+C 退出", file=sys.stderr)
    if args.open:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - 打不开浏览器不影响服务
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n关掉了。", file=sys.stderr)
    finally:
        try:
            server.stop()
        except Exception:  # noqa: BLE001
            pass
    return 0


def cmd_body(args: argparse.Namespace) -> int:
    """身体服务：独立进程，只做感知/列候选/编译，不评价、不选择、不写库。"""
    from persona.transport.server import BodyServer, describe
    from persona.transport.service import BodyService

    service = BodyService(noise=float(args.noise))
    server = BodyServer(service, host=str(args.host), port=int(args.port))
    print(describe(server))
    print("core 用 `persona --body ws://… day --day 0` 连上来；Ctrl+C 退出", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n关掉了。", file=sys.stderr)
    finally:
        try:
            server.stop()
        except Exception:  # noqa: BLE001
            pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="persona", description="一个人的成长内核")
    p.add_argument("--version", action="version", version=f"persona {__version__}")
    p.add_argument(
        "--config",
        action="append",
        default=[],
        help="配置文件路径，可重复；后者覆盖前者（默认 config/default.json）",
    )
    sub = p.add_subparsers(dest="command", required=True)
    p.add_argument(
        "--force",
        action="store_true",
        help="允许重跑已存在的日子（默认跳过，保证幂等）",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别（默认 INFO）",
    )
    p.add_argument(
        "--no-log",
        action="store_true",
        help="不写日志文件，只在控制台输出（默认写 data/<个体>/logs/persona.log）",
    )
    p.add_argument(
        "--body",
        default=None,
        metavar="ws://host:port",
        help="身体在另一个进程里（默认在本进程内）；先用 `persona body` 起一个",
    )

    sub.add_parser("birth", help="出生").set_defaults(func=cmd_birth)

    d = sub.add_parser("day", help="过一天")
    d.add_argument("--day", type=int, default=0)
    d.set_defaults(func=cmd_day)

    r = sub.add_parser("run", help="连跑多天")
    r.add_argument("--days", type=int, default=1)
    r.add_argument("--start", type=int, default=0)
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("verify", help="体检")
    v.add_argument("--max-per-day", type=int, default=60)
    v.set_defaults(func=cmd_verify)

    w = sub.add_parser("why", help="追溯一条 episode")
    w.add_argument("--episode", required=True)
    w.set_defaults(func=cmd_why)

    rp = sub.add_parser("report", help="画像")
    rp.add_argument("--date", default=None)
    rp.set_defaults(func=cmd_report)

    dy = sub.add_parser("diary", help="写日记（Markdown，观察者视角）")
    dy.add_argument("--day", type=int, default=None, help="只写这一天（默认写最近一段）")
    dy.add_argument("--days", type=int, default=30, help="不给 --day 时，从第 0 天写到这里")
    dy.set_defaults(func=cmd_diary)

    lv = sub.add_parser("live", help="按真实时间一格格往前走（Ctrl+C 停）")
    lv.add_argument("--pace", type=float, default=60.0, help="1 真实秒 = 多少模拟秒（默认 60）")
    lv.add_argument("--tick", type=float, default=0.25, help="检查时钟的间隔秒数（默认 0.25）")
    lv.add_argument("--days", type=int, default=0, help="走几天（0 = 一直走）")
    lv.add_argument("--until-day", type=int, default=-1, help="走到第几天为止（-1 = 不看）")
    lv.add_argument("--quiet", action="store_true", help="不打印每条 episode")
    lv.set_defaults(func=cmd_live)

    wt = sub.add_parser("watch", help="只读观察台（独立进程，浏览器里看进度）")
    wt.add_argument("--host", default="127.0.0.1", help="绑定地址（默认只绑本机）")
    wt.add_argument("--port", type=int, default=8765, help="端口（默认 8765，0 表示随便挑一个）")
    wt.add_argument("--open", action="store_true", help="顺手打开浏览器")

    bd = sub.add_parser("body", help="身体服务（独立进程，WS；core 用 --body 连它）")
    bd.add_argument("--host", default="127.0.0.1", help="绑定地址（默认只绑本机）")
    bd.add_argument("--port", type=int, default=8767, help="端口（默认 8767，0 表示随便挑一个）")
    bd.add_argument("--noise", type=float, default=0.05, help="感知噪声（默认 0.05，与 core 内一致）")
    bd.set_defaults(func=cmd_body)
    wt.set_defaults(func=cmd_watch)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    reconfigure_streams()  # 先钉住编码，再允许任何输出
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except PersonaError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
