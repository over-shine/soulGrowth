"""一天是怎么过去的：episode 主循环。

时序严格照 docs/02：

    plan → snapshot → observe → appraise → retrieve → decide → compile → execute → outcome → write

这里没有「时间快进」的黑魔法：每个 slot 都真的推进世界与身体，
落一条 episode，写一条事件。所以重跑同一天、同一个 seed，结果必然一致。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

from persona.body.compiler import Compiler, DENIAL_CODES
from persona.body.perception import Bundle, Perception
from persona.body.primitives import default_registry
from persona.body.state import BodyState
from persona.clock import TZ, day_index, parse_iso
from persona.cognition.appraisal import History, appraise, stranger_signal
from persona.cognition.drives import DriveContext, compute as compute_drives
from persona.cognition.policy import PriorProvider, decide
from persona.cognition.traits import (
    Bias,
    Temperament,
    Traits,
    bias as bias_of,
    sample_temperament,
    t0_from_temperament,
    update_traits,
)
from persona.config import Config
from persona.errors import PersonaError
from persona.language.lexicon import Lexicon
from persona.log import get as get_logger
from persona.llm.budget import Budget
from persona.llm.provider import LLMProvider, make_provider
from persona.rng import bernoulli, derive
from persona.safety.gate import SafetyGate
from persona.stages import judge, milestones
from persona.store.db import Database
from persona.store.events import EventLog
from persona.store.repo import Repository
from persona.transport.body_io import LocalBody
from persona.types import (
    Action,
    Appraisal,
    Drives,
    Episode,
    Language,
    LLMDebug,
    Outcome,
    clip01,
    r6,
)
from persona.world.generator import create_world, reset_day
from persona.world.npc import Caregiver
from persona.world.objects import (
    CRIB_HALF_X,
    CRIB_HALF_Y,
    ROOM_HALF_X,
    ROOM_HALF_Y,
    WorldState,
    edge_distance,
    room_distance,
)

#: 一天的清醒时段（小时）：06:00 起床，22:00 入睡
DAY_START_HOUR = 6
DAY_END_HOUR = 22
SLOT_MINUTES = 30
NIGHT_SLEEP_MINUTES = 480
#: 每 6 段（3 小时）喂一次奶；夜里那顿算在夜间 episode 里
FEED_INTERVAL_SLOTS = 6
#: 一天最多记几条。多了等于没记——记忆的价值在于它是选择的结果。
MEMORY_PER_DAY_CAP = 8
#: 照料者主动说话的概率（每格）。S0 只有声音没有句子，所以这一项在 S0 恒不触发。
CAREGIVER_SPEECH_PROB = 0.35
#: 摇铃从哪天起会被盖住（客体永久性要等他会坐会爬才有意义）
OCCLUDE_FIRST_DAY = 30
#: 摇铃被盖住几格之后会被收走（不能让他永远找不到，那是世界坏了，不是他笨）
OCCLUDE_MAX_SLOTS = 6
#: 陌生人从哪一周起会来（docs/03 的陌生人焦虑在 6–9 个月明显）
STRANGER_FIRST_DAY = 35
#: 一格之内来一个陌生人的概率；来一次待两格
STRANGER_ARRIVAL_PROB = 0.04
STRANGER_STAY_SLOTS = 2
#: 社会性微笑的门槛：看见熟悉的脸 + 效价够正 → 自动笑（这是反射，不是决定）
SMILE_MIN_VALENCE = 0.12
#: 一天里照料者主动指着东西给他看的频率（共同注意的「指」是大人先发起的）
CAREGIVER_POINT_PROB = 0.08
#: 大人把玩具收在手里不给的概率（延迟满足的情境由世界提供，不是他决定的）
WITHHOLD_PROB = 0.05
#: 照护者把三张卡片摆出来让他挑的频率（摆出来才有得挑，不是随时都能挑）
SHOW_CARDS_PROB = 0.10
logger = get_logger("runner")


@dataclass
class DayResult:
    day: int
    date: str
    episodes: int = 0
    degraded: int = 0
    denials: dict[str, int] = field(default_factory=dict)
    milestones: list[dict[str, Any]] = field(default_factory=list)
    advanced: str | None = None
    body: dict[str, Any] = field(default_factory=dict)
    traits: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "date": self.date,
            "episodes": self.episodes,
            "degraded": self.degraded,
            "denials": dict(self.denials),
            "milestones": self.milestones,
            "advanced": self.advanced,
            "body": self.body,
            "traits": self.traits,
        }


@dataclass
class _DaySession:
    """一天走到一半时的全部「进行中状态」。

    ``run_day`` 一次跑完一天；``live`` 模式要把一天拆成 32 次「走一格」：每格他
    看一次、评价一次、动一次，中间隔着真实的 30 分钟（按 pace 缩放）。所以这些
    变量不能是局部变量，必须能活过 tick，也必须能落盘后重建。
    """

    day: int
    date_str: str
    stage: str
    traits: Any
    bias: Any
    seq: int
    k: int
    slot_count: int
    start: datetime
    result: DayResult
    done: bool = False


class Simulation:
    """一个人。创建时从磁盘恢复世界与身体；没有就出生。"""

    def __init__(self, cfg: Config, *, provider: LLMProvider | None = None, total: bool = False) -> None:
        self.cfg = cfg
        self.individual_id = cfg.str_("individual.id")
        self.birth_iso = cfg.str_("individual.birth")
        self.root = Path(cfg.str_("store.path")).parent
        self.root.mkdir(parents=True, exist_ok=True)

        self.db = Database(cfg.str_("store.path"))
        self.db.init_schema()
        self.log = EventLog(cfg.str_("store.events"))
        self.repo = Repository(self.db, self.log)

        self.registry = default_registry()
        self.perception = Perception(noise=cfg.float_("perception.noise", 0.05))
        self.compiler = Compiler(self.registry)
        self.gate = SafetyGate()
        self.caregiver = Caregiver()
        #: 身体的那三个入口（看 / 列候选 / 编译）。默认进程内；可以换成网线那头。
        self.body_io: Any = LocalBody(self.perception, self.compiler)

        self._world_path = self.root / "world_state.json"
        self._body_path = self.root / "body.json"
        self._lexicon_path = self.root / "lexicon.json"
        self._temperament_path = self.root / "temperament.json"

        self.world = self._load_world() or create_world(
            individual_id=self.individual_id, birth_iso=self.birth_iso
        )
        self.body = self._load_body()
        self.lexicon = self._load_lexicon()
        self.temperament = self._load_temperament()

        self.provider = provider if provider is not None else make_provider(cfg)
        self.llm_enabled = int(cfg.int_("llm.max_calls_per_episode", 2)) > 0
        self.llm_weight = cfg.float_("policy.llm_prior_weight", 0.45)
        self.temperature = cfg.float_("policy.temperature", 0.7)
        self.shortlist_size = cfg.int_("policy.shortlist_size", 5)
        self.affordance_limit = cfg.int_("policy.affordance_limit", 12)
        self.max_episodes_per_day = cfg.int_("runtime.max_episodes_per_day", 60)
        self.force_rerun = bool(cfg.get("runtime.force_rerun", False))
        #: 当天已写了几条记忆（每天重置，见 begin_day）
        self._memories_today = 0
        #: 正在过的那一天；一次跑完一天时是临时的，live 模式跨 tick 存在
        self._session: _DaySession | None = None

    # -- 生命周期 -----------------------------------------------------------
    def attach_body(self, body_io: Any) -> Any:
        """把这具身体的入口换成别处（比如另一个进程）。返回旧的那个，方便还原。"""
        previous = self.body_io
        self.body_io = body_io
        return previous

    @property
    def stage(self) -> str:
        return self.repo.latest_stage() or self.cfg.str_("stage.start", "S0")

    def born(self) -> bool:
        """第一次运行：写下出生事件与初始特质。返回是否是首次。"""
        if self.repo.latest_stage() is not None:
            return False
        now = self._birth_dt().isoformat()
        self.repo.event("person.born", ts=now, ref=self.individual_id, actor="core")
        self.repo.record_stage_advance(
            "", self.stage, now, 0, [], list(self.registry.unlocked_at(self.stage))
        )
        t0 = t0_from_temperament(self.temperament, individual_id=self.individual_id, birth_iso=self.birth_iso)
        rec = Traits(week=_iso_week(self._birth_dt().date()), stage=self.stage, T=t0).to_dict()
        self.repo.put_trait(rec)
        self.save()
        logger.info(
            "出生：%s（%s），气质 S=%.3f NA=%.3f EC=%.3f AF=%.3f，阶段 %s",
            self.individual_id,
            now,
            self.temperament.s,
            self.temperament.na,
            self.temperament.ec,
            self.temperament.af,
            self.stage,
        )
        return True

    def save(self) -> None:
        self.world.save(self._world_path)
        self._write_json(self._body_path, self.body.to_dict())
        self.lexicon.save(self._lexicon_path)
        self._write_json(self._temperament_path, self.temperament.to_dict())

    def close(self) -> None:
        self.save()
        try:
            self.body_io.close()
        except Exception:  # 身体那边断了不该拦住存档
            pass
        self.provider.close()
        self.db.close()

    # -- 一天：可以一次跑完，也可以一格一格地走 ------------------------------
    def begin_day(self, day: int, *, resume: bool = False) -> _DaySession:
        """把一天准备好，但一条 episode 都不跑。

        live 模式要按真实时间一格格往前走，所以「今天的阶段、特质、偏置、序号、
        起点」得先固定下来、能跨 tick 存活；批量模式只是紧接着把这些格子一口气
        走完，两种模式走的是同一段代码。

        ``resume=True`` 用于「进程被杀在这天中间」：不重头跑，从最后一条 episode
        的下一个格子接着走（世界状态本来就停在那里）。
        """
        self.born()
        date_ = self._date_of(day)
        result = DayResult(day=day, date=date_.isoformat())
        # 同一天重跑是幂等的：已经在库里的日子直接返回，不再追加一遍
        # （否则 30 天会变成 1800 条，单日上限 I4 立刻炸）。
        already = self.repo.count_episodes_of_day(result.date)
        slot_count = (DAY_END_HOUR - DAY_START_HOUR) * 60 // SLOT_MINUTES
        start = datetime.combine(date_, dtime(hour=DAY_START_HOUR), tzinfo=TZ)
        if already and not self.force_rerun and not resume:
            logger.info("day %d %s 已有 %d 条 episode，跳过（--force 可重跑）", day, result.date, already)
            result.episodes = already
            result.body = self.body.to_dict()
            traits = self._current_traits(self.stage)
            result.traits = traits.to_dict()
            result.milestones = self.repo.milestones_of_stage(self.stage)
            return _DaySession(
                day=day,
                date_str=result.date,
                stage=self.stage,
                traits=traits,
                bias=bias_of(traits.T, temperament=self.temperament),
                seq=already,
                k=0,
                slot_count=0,
                start=start,
                result=result,
                done=True,
            )
        if resume and already:
            # 接着走：世界/身体状态停在最后一条 episode 之后，不能再 reset_day。
            episodes = self.repo.episodes_of_day(result.date)
            last_ts = parse_iso(episodes[-1]["ts"]) if episodes else start
            minutes = (last_ts - start).total_seconds() / 60.0
            k = max(0, min(slot_count, int(round(minutes / SLOT_MINUTES)) + 1))
            logger.warning("day %d 从第 %d/%d 格续跑", day, k, slot_count)
        else:
            reset_day(self.world, day=day)
            self._memories_today = 0
            k = 0
        stage = self.stage
        traits = self._current_traits(stage)
        session = _DaySession(
            day=day,
            date_str=result.date,
            stage=stage,
            traits=traits,
            bias=bias_of(traits.T, temperament=self.temperament),
            seq=already,
            k=k,
            slot_count=slot_count,
            start=start,
            result=result,
        )
        session.result.episodes = already
        self._session = session
        return session

    def run_step(self, *, catch_up: bool = False) -> tuple[str, Any]:
        """走一格。

        返回 ``("episode", Episode)`` 或 ``("day_end", DayResult)``。``catch_up``
        表示这一格不是「刚刚发生」的，而是补做的——它照样进库，但带上标记，
        人的投影里权重会被压低（I10）。
        """
        session = self._session
        if session is None:
            raise RuntimeError("没有正在过的一天：先调用 begin_day()")
        if session.done:
            self._session = None
            return ("day_end", session.result)

        if session.k < session.slot_count and session.result.episodes < self.max_episodes_per_day:
            k = session.k
            session.k += 1
            # 照料者按时喂奶：不是「哭了才喂」。没有这一条，饥饿会顶到 1.0 就下不来，
            # 于是所有哭声都只能是饿，三种哭声永远分不出来（M0.6）。
            if k > 0 and k % FEED_INTERVAL_SLOTS == 0:
                # 0.15 ≈ 两个半小时的积累量：抵消不掉的差额要靠「饿哭了才喂」补，
                # 所以饥饿一天里会真的爬到饥饿线以上，而不是永远为零。
                self.body.eat(0.15)
                self.world.counters["suck_streak"] = 0
                self.world.bump("caregiver_fed")
            ts = session.start + timedelta(minutes=SLOT_MINUTES * k)
            # 照料者每天傍晚离开一小时（从第 3 天起）：分开时的难受是「认照料者」的
            # 唯一行为证据——她没有熟人在场时的那种不安，只能靠对比「在」与「不在」看出来。
            self.world.caregiver_present = not (session.day >= 3 and 22 <= k < 28)
            session.seq += 1
            ep = self._episode(
                day=session.day,
                stage=session.stage,
                ts=ts,
                seq=session.seq,
                b=session.bias,
                traits=session.traits,
            )
            if catch_up:
                ep = replace(ep, catch_up=True)
            # 关注目标是「这一格」的，不是「这天」的：不留给下一格
            self.world.focus_target = None
            self.repo.save_episode(ep)
            session.result.episodes += 1
            if ep.llm.degraded:
                session.result.degraded += 1
            for code in ep.action.params.get("_denials", []) or []:
                session.result.denials[code] = session.result.denials.get(code, 0) + 1
            # 每格都落盘：进程随时可能被杀，下一格不该从头再来。
            self.save()
            return ("episode", ep)

        return ("day_end", self._finish_day(session))

    def _finish_day(self, session: _DaySession) -> DayResult:
        """夜间那一条 + 结算（特质、里程碑、日记）。这是「这一天结束了」的唯一出口。"""
        day = session.day
        result = session.result
        date_ = self._date_of(day)
        # 夜间：一整段睡眠，只记一条
        if session.seq < self.max_episodes_per_day:
            night_ts = datetime.combine(date_, dtime(hour=DAY_END_HOUR), tzinfo=TZ)
            session.seq += 1
            ep = self._night_episode(day=day, stage=session.stage, ts=night_ts, seq=session.seq)
            self.repo.save_episode(ep)
            result.episodes += 1

        self.save()
        result.body = self.body.to_dict()
        result.traits = self._settle_traits(day, session.stage, session.traits)
        result.milestones, result.advanced = self._judge(day, session.stage, result.date)
        passed = sum(1 for m in result.milestones if m.get("status") == "passed")
        logger.info(
            "day %d %s 结束：episode %d 条，degraded %d 条，里程碑 %d/%d 通过，"
            "能量 %.1f 饥饿 %.2f 疲劳 %.2f",
            day,
            result.date,
            result.episodes,
            result.degraded,
            passed,
            len(result.milestones),
            self.body.energy,
            self.body.hunger,
            self.body.fatigue,
        )
        if result.denials:
            logger.warning("day %d 被拒绝的动作：%s", day, dict(sorted(result.denials.items())))
        if result.advanced:
            logger.warning("day %d 阶段推进：%s → %s", day, session.stage, result.advanced)
        if bool(self.cfg.get("runtime.diary", True)):
            self._write_diary(day, result)
        session.done = True
        self._session = None
        return result

    def run_day(self, day: int) -> DayResult:
        """一口气走完一天（批量模式）。和 live 模式共用一个 state machine。"""
        session = self.begin_day(day)
        if session.done:
            return session.result
        while True:
            kind, payload = self.run_step()
            if kind == "day_end":
                return payload

    def _write_diary(self, day: int, result: DayResult) -> None:
        """写一份人可读的日记（观察者视角；他自己要能写日记得等到 L5）。"""
        from persona.observe.diary import render_day, write_day

        episodes = self.repo.episodes_of_day(result.date)
        day_memories = [
            m for m in self.repo.list_memories(limit=400) if str(m.get("created", ""))[:10] == result.date
        ]
        text = render_day(
            day=day,
            date=result.date,
            stage=str(episodes[-1].get("stage") if episodes else self.stage),
            episodes=episodes,
            milestones=result.milestones,
            traits=result.traits,
            memories=day_memories,
            name=self.cfg.str_("individual.name", "小满"),
        )
        path = Path(self.cfg.str_("store.path")).parent / "diary" / f"{result.date}.md"
        write_day(path, text)
        logger.debug("day %d 日记写到 %s", day, path)

    # -- 单条 episode --------------------------------------------------------
    def _episode(
        self,
        *,
        day: int,
        stage: str,
        ts: datetime,
        seq: int,
        b: Bias,
        traits: Traits,
    ) -> Episode:
        iso = ts.isoformat()
        ep_id = f"ep-{ts:%Y%m%d}-{seq:04d}"
        hours = SLOT_MINUTES / 60.0

        self.world.advance(hours)
        self.body.advance(hours, awake=True)
        clock_h = ts.hour + ts.minute / 60.0
        # 共同注意：**上一格**她指过的东西，这一格才是注意的焦点。
        # 必须先取旧值再跑 _world_tick——否则她这一格刚指的东西会被这一格自己吃掉，
        # 「跟随」就永远只发生在指的那一瞬，看起来像一次都没跟。
        joint = self.world.joint_target
        self.world.joint_target = None
        tick_events = self._world_tick(day=day, ep_id=ep_id, stage=stage, hour=clock_h)

        history = History.from_episodes(self.repo.recent_episodes(20))
        # 编译器要看见它：被指的东西必须有一个「看它」的候选，否则目光没处落
        self.world.focus_target = joint
        bundle = self.body_io.bundle(
            self.world,
            self.body,
            stage=stage,
            day=day,
            episode_id=ep_id,
            clock_hours=clock_h,
            goal_focus=joint or self._goal_focus(b, history),
        )
        appraisal = appraise(
            bundle,
            self.body,
            bias=b,
            history=history,
            caregiver_present=self.world.caregiver_present,
            expectation=0.0,
        )
        smiled = self._social_smile(stage, bundle, appraisal)
        # 唤醒度是被刺激驱动的状态量：每个 episode 由本次评价重置，而不是只增不减
        self.body.arousal = appraisal.arousal
        drives = compute_drives(
            self.body,
            bundle,
            DriveContext(
                caregiver_present=self.world.caregiver_present,
                minutes_since_care=float(self.world.caregiver_last_response_min or 0.0),
                recent_novelty=appraisal.novelty,
                recent_success=history.success_rate,
                threat=appraisal.threat,
                care_n=float(
                    sum(v for key, v in self.world.counters.items() if key.startswith("caregiver_"))
                ),
            ),
        )

        candidates = self.body_io.candidates(
            bundle,
            self.body,
            stage=stage,
            world=self.world,
            lexicon=self.lexicon,
            limit=self._affordance_limit(stage),
            social=drives.social,
        )
        budget = Budget(
            max_calls=int(self.cfg.int_("llm.max_calls_per_episode", 2)),
            max_input_tokens=int(self.cfg.int_("llm.max_input_tokens", 3000)),
            max_output_tokens=int(self.cfg.int_("llm.max_output_tokens", 400)),
        )
        provider: PriorProvider | None = None
        if self.llm_enabled and budget.can_call():
            provider = _BudgetedProvider(self.provider, budget)
        needs: dict[str, float] = {
                "fatigue": self.body.fatigue,
                "discomfort": self.body.nociception(appraisal.threat),
                # 视觉追踪不是出生就成熟的：第 1 周只有「有东西动会惊一下」，
                # 两周左右才稳定地把注视转过去（docs/03 前 30 天表：8–14 天追踪稳定）。
                "orient": (
                    min(1.0, 0.25 + 0.75 * day / 14.0) if bundle.onsets else 0.0
                ),
                "suck_streak": float(self.world.counters.get("suck_streak", 0)),
                # 发声驱力：还有没练熟的词就想练；一个都没有了也还是会咿呀
                # （咿呀不是为词服务的，它本身就是要发的声）
                "vocal": 1.0 if self.lexicon.practice_form() is not None else 0.6,
                "vocal_word": 1.0 if self.lexicon.productive_forms() else 0.0,
                # 积压：听见了但还说不出来的词有多少。积压越大，越想说——
                # 这是词汇爆发期的发动机（S2 两周五十个词，靠的不是奖励，是这个缺口）。
                "practice_backlog": self._practice_backlog(),
                "age_days": float(day),
                # 今天说过多少句：说腻了就不想说了（不然两岁那年一半的清醒时间在说词）
                "speech_sat": clip01(float(self.world.counters.get("said_today", 0)) / 12.0),
                # 假装也是会「玩腻」的
                "pretend_sat": clip01(
                    float(self.world.counters.get("pretend_today", 0)) / 8.0
                ),
                "babble_sat": clip01(float(self.world.counters.get("babble_today", 0)) / 10.0),
                "walk_sat": clip01(float(self.world.counters.get("walk_today", 0)) / 12.0),
                # 看不见了还想要：被盖住的东西会拉着他去找（客体永久性）
                "occluded": 1.0 if any(self.world.occluded.values()) else 0.0,
                # 不认识的人在：往照料者那边靠（陌生人焦虑）
                "stranger": stranger_signal(bundle),
                # 刚刚指过的东西：目光跟过去（共同注意）
                "joint": 1.0 if joint else 0.0,
                "joint_target": joint or "",
                # 跟随指向不是出生就有的：9–12 个月才稳定（docs/03 的 M1.7）
                "joint_gain": 6.0 * clip01((day - 45) / 25.0),
                # 想指给人看：她在场、眼前有东西
                # 指给人看：照料者在，眼前也有东西
                "share": 1.0 if (self.world.caregiver_present and bundle.stimuli) else 0.0,
                # 够不着的东西：想过去（学步不是漫无目的地走，是朝东西走）
                "far_target": self._far_target(bundle),
                # 假装：手边有玩具、有人在看，才有「演给你看」这件事
                "pretend_drive": 1.0
                if (self.world.caregiver_present and bundle.stimuli)
                else 0.0,
                # 等：有东西被收着，就有「等还是抢」这个选择
                "wait_drive": 1.0 if self.world.withheld else 0.0,
                # 挑：卡片举在眼前时才有得挑
                "choose_drive": 1.0 if self.world.shown_cards else 0.0,
        }
        # 挑过多少次：偏好是选出来的，不是写死的（M2.9 看的是 top1 占比）
        choose_total = 0
        for oid, n in self.world.counters.items():
            if oid.startswith("chose:"):
                needs[f"chose:{oid[6:]}"] = float(n)
                choose_total += int(n)
        needs["choose_total"] = float(choose_total)
        # 今天各做过几次：策略层用它给「已经做得太多的事」降温
        for key, val in self.world.counters.items():
            if key.startswith("done_today:"):
                needs[f"today:{key[11:]}"] = float(val)
        decision = decide(
            candidates,
            bias=b,
            drives=drives,
            appraisal=appraisal,
            provider=provider,
            llm_weight=self.llm_weight,
            temperature=self.temperature,
            shortlist_size=self.shortlist_size,
            rng=derive("policy", self.individual_id, day, ep_id),
            observation=bundle.observation(),
            needs=needs,
        )

        compiled = self.body_io.compile(
            decision.primitive,
            decision.params,
            stage=stage,
            body=self.body,
            world=self.world,
            lexicon=self.lexicon,
            intent=decision.intent,
            bundle=bundle,
            social=drives.social,
        )
        verdict = self.gate.check(
            compiled.utterance, stage=stage, level=compiled.level, lexicon=self.lexicon
        )
        if joint:
            logger.debug("共同注意：%s → %s", joint, decision.primitive)
        if verdict.blocked:
            self.repo.event(
                "safety.blocked",
                ts=iso,
                ref=ep_id,
                actor="safety",
                reason=verdict.reason,
                attempted=compiled.utterance,
            )

        success, events, reward = self._execute(compiled, appraisal, ts=iso, ep_id=ep_id)
        events = tick_events + events
        if smiled:
            events.append("smiled")
            self.world.bump("smiles")
        # 照料者会对他说点什么。S0 她只有声音（没有句子），S1 起这些话开始长成词。
        if stage != "S0":
            self._caregiver_speaks(events, ep_id=ep_id, day=day)
        self.body.note_episode()

        salience = clip01(
            0.5 * _mean_salience(bundle) + 0.3 * appraisal.arousal + 0.2 * abs(appraisal.valence)
        )
        memory_ids = self._remember(ep_id, iso, bundle, compiled, appraisal, salience, stage, events)
        #: 记下「这辈子第一次做成这个动作」，下一条同样的动作就不算第一次了
        self.world.counters[f"once:{compiled.primitive}"] = 1

        action = Action(
            primitive=compiled.primitive,
            target=str(compiled.params.get("target") or "") or None,
            params={k: v for k, v in compiled.params.items() if not k.startswith("_")},
            source=_source(decision, compiled),
            intent=decision.intent,
            fidelity=r6(compiled.fidelity),
            degraded=compiled.degraded,
            chain=tuple(compiled.chain),
            candidates=tuple(
                {"primitive": p, "score": r6(decision.scores.get(p, 0.0))} for p in decision.shortlist
            ),
        )
        if compiled.denials:
            action.params["_denials"] = list(compiled.denials)

        return Episode(
            id=ep_id,
            ts=iso,
            age_days=day,
            stage=stage,
            kind="awake",
            context={
                "location": "crib",
                "others": (
                    (["mother"] if self.world.caregiver_present else [])
                    + (["stranger"] if self.world.stranger_present else [])
                ),
                "objects": bundle.salient_ids(),
                "world_time": iso,
            },
            observation=bundle.observation(),
            appraisal=appraisal,
            drives=drives,
            action=action,
            outcome=Outcome(
                success=success,
                events=events,
                reward=r6(reward),
                unexpected=r6(appraisal.pe),
            ),
            language=Language(
                mode=verdict.utterance and compiled.mode or compiled.mode,
                utterance=verdict.utterance,
                syntax_level=compiled.level,
            ),
            memory_refs=memory_ids,
            llm=LLMDebug(
                model=decision.prior.model,
                tokens_in=decision.prior.tokens_in,
                tokens_out=decision.prior.tokens_out,
                degraded=decision.prior.degraded,
                degraded_reason=decision.prior.degraded_reason,
            ),
            salience=r6(salience),
            catch_up=False,
        )

    def _night_episode(self, *, day: int, stage: str, ts: datetime, seq: int) -> Episode:
        iso = ts.isoformat()
        ep_id = f"ep-{ts:%Y%m%d}-{seq:04d}"
        self.body.sleep(NIGHT_SLEEP_MINUTES)
        self.world.advance(NIGHT_SLEEP_MINUTES / 60.0)
        # 夜间不是「一觉到天亮」：新生儿每 2–3 小时被喂一次，否则第二天一睁眼就饿到 1.0
        self.body.eat(0.30)
        self.body.eat(0.30)
        self.world.counters["suck_streak"] = 0
        self.world.bump("night_fed", 2)
        return Episode(
            id=ep_id,
            ts=iso,
            age_days=day,
            stage=stage,
            kind="sleep",
            context={"location": "crib", "others": [], "objects": [], "world_time": iso},
            observation={"modality": [], "summary": "睡着", "salient": [], "interoception": {}},
            appraisal=Appraisal(valence=0.0, arousal=0.0, threat=0.0, control=0.0, pe=0.0, novelty=0.0),
            drives=self.body.core_drives(),
            action=Action(primitive="A0.SLEEP", params={"duration_min": NIGHT_SLEEP_MINUTES}, source="heuristic"),
            outcome=Outcome(success=True, events=["night_sleep"], reward=0.0, unexpected=0.0),
            language=Language(mode="pre_verbal", utterance=None, syntax_level="L0"),
            memory_refs=[],
            llm=LLMDebug(model="none", degraded=False),
            salience=0.1,
            catch_up=False,
        )

    # -- 后果 ---------------------------------------------------------------
    def _world_tick(self, *, day: int, ep_id: str, stage: str, hour: float) -> list[str]:
        """世界自己会发生的事：摇铃被盖住、陌生人来访。都不是他决定的。

        这一层刻意放在**感知之前**：他接下来看到的世界，已经不是上一格那个世界了。
        """
        events: list[str] = []
        if stage == "S0":
            self.world.stranger_present = False
            self.world.counters["stranger_slots_left"] = 0
            return events
        rng = derive("world-tick", self.individual_id, day, ep_id)

        # 摇铃被毯子盖住：从视野里消失，但世界还记得它在毯子下面。
        if day >= OCCLUDE_FIRST_DAY:
            if self.world.occluded.get("rattle"):
                slots = int(self.world.counters.get("rattle_covered_slots", 0)) + 1
                self.world.counters["rattle_covered_slots"] = slots
                if slots >= OCCLUDE_MAX_SLOTS:
                    self.world.occluded["rattle"] = False
                    self.world.counters["rattle_covered_slots"] = 0
                    events.append("uncovered:rattle")
            elif rng.random() < 0.18:
                self.world.occluded["rattle"] = True
                self.world.counters["rattle_covered_slots"] = 0
                events.append("covered:rattle")

        # 陌生人：白天偶尔来一次，待两格就走。
        left = int(self.world.counters.get("stranger_slots_left", 0))
        if left > 0:
            self.world.counters["stranger_slots_left"] = left - 1
            self.world.stranger_present = True
            if left == STRANGER_STAY_SLOTS:
                events.append("stranger_arrived")
        else:
            self.world.stranger_present = False
            if (
                day >= STRANGER_FIRST_DAY
                and 9.0 <= hour <= 17.0
                and rng.random() < STRANGER_ARRIVAL_PROB
            ):
                self.world.counters["stranger_slots_left"] = STRANGER_STAY_SLOTS - 1
                self.world.stranger_present = True
                events.append("stranger_arrived")

        # 她指着某样东西给他看：共同注意的「指向」是大人发起的，不是他。
        # 新生儿不会跟随指向（M1.7 要测的正是这件事从不会到会）。
        if (
            self.world.caregiver_present
            and 8.0 <= hour <= 19.0
            and rng.random() < CAREGIVER_POINT_PROB
        ):
            shown = [
                o.id
                for o in self.world.visible_objects()
                # 会走之后，她指的东西不再限于床边那几样：镜子、杯子、积木都指
                if o.id
                in (
                    "mobile",
                    "window",
                    "blanket",
                    "rattle",
                    "mirror",
                    "cup",
                    "blocks",
                    "ball",
                )
            ]
            if shown:
                target = shown[int(rng.random() * len(shown)) % len(shown)]
                events.append(f"caregiver_pointed:{target}")
                self.world.joint_target = target

        # 大人把一样东西收在手里不给：想要就得等，这是延迟满足的那个情境。
        # 只有 S2 之后才出现——更早的时候「等」不是一个可选动作，抢是唯一的表达。
        if stage >= "S2" and self.world.caregiver_present:
            held = str(self.world.withheld or "")
            if held:
                self.world.withheld = ""
                events.append(f"released:{held}")
            elif 8.0 <= hour <= 18.0 and rng.random() < WITHHOLD_PROB:
                near = [
                    o.id
                    for o in self.world.visible_objects()
                    if o.id in ("ball", "rattle", "blocks", "cup", "book") and o.reachable
                ]
                if near:
                    target = near[int(rng.random() * len(near)) % len(near)]
                    self.world.withheld = target
                    events.append(f"withheld:{target}")

        # 卡片：她摆出来让他挑。只在此刻「挑」才有意义，平时那几张纸只是纸。
        if stage >= "S2" and self.world.caregiver_present:
            if self.world.shown_cards:
                self.world.shown_cards = False
            elif 8.0 <= hour <= 18.0 and rng.random() < SHOW_CARDS_PROB:
                self.world.shown_cards = True
                events.append("showed_cards")
        return events

    def _social_smile(self, stage: str, bundle: Bundle, appraisal: Appraisal) -> bool:
        """社会性微笑：看见一张熟悉的脸，而且此刻感觉不坏 → 自动笑出来。

        S0 没有面孔通道，所以这一条在 S0 恒为假。它不是「决定要笑」——
        docs/03 的 M1.1 说的是 6–8 周起的那种自动微笑：先有表情，后有社交意图。
        """
        if stage == "S0" or appraisal.valence < SMILE_MIN_VALENCE:
            return False
        for s in bundle.stimuli:
            fac = s.channels.get("FAC")
            if fac and fac.get("present") and fac.get("familiar", False):
                return True
        return False

    def _execute(self, compiled: Any, appraisal: Appraisal, *, ts: str, ep_id: str) -> tuple[bool, list[str], float]:
        prim = compiled.primitive
        params = compiled.params
        events: list[str] = []
        success = True
        # 今天这类事做过几次：一天的总量有限，谁也不能把八小时全用在同一个动作上
        self.world.bump(f"done_today:{prim}")

        if prim == "A0.NONE":
            success = False
            events.append("unexpressed")
        elif prim == "A0.CRY":
            resp = self.caregiver.respond_to_cry(
                self.world,
                day=int(self.world.day),
                episode_id=ep_id,
                reason=str(params.get("reason", "discomfort")),
                stage=self.stage,
            )
            if resp is not None and resp.kind != "none":
                events.append(f"caregiver_{resp.kind}")
                self.body.be_held()
                if resp.speech:
                    # 被抱着的时候听见的那句话，是最容易和「舒服」绑在一起的那句
                    self._hear(
                        resp.speech,
                        events,
                        day=int(self.world.day),
                        situation="feed" if resp.kind == "feed" else "soothe",
                    )
                if resp.kind == "feed":
                    # 被喂到：这是唯一真正降低饥饿的路径
                    self.body.eat(0.35)
                    self.world.counters["suck_streak"] = 0
                success = True
            else:
                events.append("no_response")
                success = False
        elif prim == "A0.FIXATE":
            target = str(params.get("target", ""))
            self.world.fixation_counts[target] = self.world.fixation_counts.get(target, 0) + 1
            obj = self.world.obj(target)
            success = obj is not None
            if obj is not None:
                events.append(f"looked_at_{obj.id}")
        elif prim == "A0.GRASP":
            target = str(params.get("target", ""))
            obj = self.world.obj(target)
            success = obj is not None and obj.reachable
            if success and obj is not None:
                events.append(f"held_{obj.id}")
        elif prim == "A0.SUCK":
            # 非营养性吮吸：安抚自己，但不顶饿——饥饿只能靠被喂解决
            self.world.bump("suck_streak")
            self.body.arousal = clip01(self.body.arousal * 0.72)
            needs_signal = "self_soothed" if self.body.hunger < 0.6 else "sucked_but_hungry"
            events.append(needs_signal)
        elif prim == "A1.BABBLE":
            # 咿呀是**练习**，不是说话：把还没说利索的那个词拿出来反复发音
            form = self._practice_form_for(params)
            events.append("babbled")
            self.world.bump("babble_today")
            if form:
                entry = self.lexicon.produce(form, day=int(self.world.day))
                events.append(f"practiced:{form}")
                if entry.status == "productive":
                    events.append(f"word_ready:{form}")
        elif prim == "A1.WORD":
            form = str(compiled.utterance or params.get("lexeme") or "")
            if form:
                self.lexicon.produce(form, day=int(self.world.day), as_word=True)
                self.world.bump("words_said")
                self.world.bump("said_today")
                events.append(f"said:{form}")
        elif prim == "A1.POINT":
            # 指给别人看（原陈述性指）。「跟随别人指」是另一回事：
            # 那要由大人的指向发起，不能自己给自己制造一个共同注意。
            target = str(params.get("target", ""))
            events.append(f"pointed:{target}")
            success = bool(target)
        elif prim == "A1.PEEK":
            # 去找那个看不见的东西。找到了，客体就重新回到视野里。
            target = str(params.get("target", ""))
            if target and self.world.occluded.get(target):
                self.world.occluded[target] = False
                self.world.counters["rattle_covered_slots"] = 0
                events.append(f"revealed:{target}")
                self.world.bump("searched_ok")
            else:
                events.append("searched_empty")
                success = False
        elif prim == "A1.CRAWL":
            events.append("crawled")
            self._crawl(params, events)
        elif prim == "A2.WALK":
            events.append("walked")
            self._walk(params, events)
        elif prim == "A2.MANIPULATE":
            self._manipulate(params, events)
        elif prim == "A2.IMITATE":
            # 模仿不需要懂意图：她做了什么，他就照着做一遍。
            # 这一格算不算「会模仿」，由 evaluator 去数，不由他自己声明。
            action = str(params.get("action") or "")
            if action:
                events.append(f"imitated:{action}")
                self.world.bump("imitations")
            else:
                success = False
        elif prim == "A2.TWO_WORD":
            text = str(compiled.utterance or "")
            a = str(params.get("a") or "")
            b = str(params.get("b") or "")
            if text and a and b:
                events.append(f"two_word:{text}")
                for form in (a, b):
                    self.lexicon.produce(form, day=int(self.world.day), as_word=True)
                self.world.bump("two_word_actions")
                self.world.bump("said_today")
                self.world.bump("two_word_kinds:" + "|".join(sorted((a, b))))
            else:
                success = False
                events.append("no_two_words")
        elif prim == "A2.TELEGRAPH":
            action = str(params.get("action") or "")
            obj = str(params.get("object") or "")
            if action:
                events.append(f"telegraphed:{action}:{obj}")
            else:
                success = False
        elif prim == "A2.SHOW":
            target = str(params.get("target") or "")
            to = str(params.get("to") or "mother")
            if target:
                events.append(f"showed:{target}:{to}")
                self.world.bump("shows")
            else:
                success = False
        elif prim == "A2.REFUSE":
            target = str(params.get("target") or "")
            events.append(f"refused:{target}")
            self.world.bump("refusals")
        elif prim == "A2.PRETEND":
            # 假装：拿真东西当别的用。证据是两个词——什么道具、演的是哪一出。
            target = str(params.get("target") or "")
            script = str(params.get("script") or "喂饭")
            events.append(f"pretended:{script}")
            if target:
                events.append(f"pretend_with:{target}")
            self.world.bump(f"pretend_script:{script}")
            self.world.bump("pretend_actions")
            self.world.bump("pretend_today")
        elif prim == "A2.WAIT":
            # 等：东西被收着，他没去抢。等过一格就是等过了（一格 30 分钟，比 30 秒严）。
            target = str(params.get("target") or "") or str(self.world.withheld or "")
            wait_s = int(float(params.get("wait_s", 60) or 60))
            events.append(f"waited:{target}:{wait_s}")
            self.world.bump("waited")
        elif prim == "A2.CHOOSE":
            target = str(params.get("target") or "")
            if target:
                events.append(f"chose:{target}")
                self.world.bump(f"chose:{target}")
                self.world.bump("choices")
            else:
                success = False
        elif prim == "A3.PRETEND":
            events.append("pretended")
            self.world.bump("pretend_slots")
        elif prim == "A3.COOPERATE":
            events.append("cooperated")
            self.world.bump("cooperate_slots")
        elif prim == "A3.ASK":
            events.append(f"asked:{compiled.utterance or ''}")
            self.world.bump("questions")
        elif prim == "A3.HELP":
            events.append("helped")
            self.world.bump("helps")
        elif prim == "A0.STARTLE":
            self.body.arousal = clip01(self.body.arousal + 0.2)
            events.append("startled")
        elif prim == "A0.TURN_HEAD":
            events.append("turned")
        elif prim == "A0.SLEEP":
            self.body.sleep(float(params.get("minutes", 60)))
            events.append("slept")
        else:
            events.append(f"did_{prim.lower().replace('.', '_')}")

        if compiled.denials:
            events.append("denied:" + ",".join(compiled.denials))
        if compiled.degraded:
            events.append("degraded_action")

        reward = clip01(0.5 * (appraisal.valence + 1.0) / 2.0 + (0.5 if success else 0.0))
        return success, events, reward

    def _crawl(self, params: Mapping[str, Any], events: list[str]) -> None:
        """爬：朝目标挪一段；挪到床沿就停住。

        「停住」不是他没力气，而是他看见了那道落差——这就是 M1.3 要测的东西。
        我们不让婴儿真的掉下去（那会毁掉后面的所有数据），只让他在边缘停下并发怵。
        """
        target = str(params.get("target") or "")
        obj = self.world.obj(target)
        dist = float(params.get("dist_m", 1.0) or 1.0)
        bx, by, bz = self.world.baby_pos
        if obj is not None:
            dx, dy = obj.pos[0] - bx, obj.pos[1] - by
        else:
            dx, dy = 0.0, -1.0
        norm = math.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        if str(params.get("dir", "toward")) == "away":
            ux, uy = -ux, -uy
        nx, ny = bx + ux * dist, by + uy * dist
        cx = max(-CRIB_HALF_X, min(CRIB_HALF_X, nx))
        cy = max(-CRIB_HALF_Y, min(CRIB_HALF_Y, ny))
        # 停在床沿：到边缘就算停住。不是「越过去被拦住」——他根本不会往外爬第二下。
        if edge_distance((cx, cy, bz)) <= 0.02:
            events.append("edge_stopped")
            self.world.bump("edge_stops")
            # 悬空感让人发怵：这是唯一一处「身体先于认知」的恐惧
            self.body.arousal = clip01(self.body.arousal + 0.25)
        self.world.baby_pos = (cx, cy, bz)
        self.world.bump("crawls")

    def _walk(self, params: Mapping[str, Any], events: list[str]) -> None:
        """走：在地板上朝目标移动一段，走到墙为止。

        和爬的区别不只是速度：走是**连续的**，所以「走了多少步」这件事本身
        可以被计数（M2.1 要的是连走 10 步）。爬是够取动作的延长，走是位移。
        """
        target = str(params.get("target") or "")
        obj = self.world.obj(target)
        dist = float(params.get("dist_m", 1.0) or 1.0)
        speed = float(params.get("speed", 1.0) or 1.0)
        bx, by, bz = self.world.baby_pos
        if obj is not None:
            dx, dy = obj.pos[0] - bx, obj.pos[1] - by
        else:
            dx, dy = 0.0, -1.0
        norm = math.hypot(dx, dy)
        if norm < 0.05:
            # 已经站在它面前了，再走就是原地踏步——不算一步
            events.append("already_there")
            return
        ux, uy = dx / norm, dy / norm
        if str(params.get("dir", "toward")) == "away":
            ux, uy = -ux, -uy
        step = min(dist, 0.15 * speed)
        nx, ny = bx + ux * step, by + uy * step
        cx = max(-ROOM_HALF_X, min(ROOM_HALF_X, nx))
        cy = max(-ROOM_HALF_Y, min(ROOM_HALF_Y, ny))
        travelled = math.hypot(cx - bx, cy - by)
        self.world.baby_pos = (cx, cy, bz)
        if travelled >= 0.05:
            self.world.bump("walk_steps")
            self.world.bump("walk_today")
            events.append(f"stepped:{travelled:.2f}m")
            if room_distance((cx, cy, bz)) <= 0.02:
                events.append("wall_stopped")

    def _manipulate(self, params: Mapping[str, Any], events: list[str]) -> None:
        """摆弄：推、拉、塞、叠、转。这是「手上有东西」和「脑子在试」的分界。

        只有真的够得着的对象才谈得上摆弄——S2 的手还离不开眼前半米。
        """
        target = str(params.get("target") or "")
        op = str(params.get("op") or "push")
        obj = self.world.obj(target)
        if obj is None:
            events.append("manipulated_nothing")
            return
        if not obj.reachable:
            # 够不着就是够不着：不是失败，是「还走不过去」
            events.append(f"too_far:{target}")
            return
        events.append(f"manipulated:{op}:{target}")
        self.world.bump(f"manip_{op}")
        if op == "stack":
            self.world.bump("stacked")
            self.world.counters["stack_height"] = min(
                6, self.world.counters.get("stack_height", 0) + 1
            )
        elif op == "insert":
            self.world.bump("inserted")
        elif op == "turn":
            self.world.bump("turned_over")

    # -- 语言：听进去的话，才是词 -------------------------------------------
    def _caregiver_speaks(self, events: list[str], *, ep_id: str, day: int) -> None:
        """她会不会开口、说什么，不由婴儿决定。"""
        if not self.world.caregiver_present:
            return
        rng = derive("caregiver-speech", self.individual_id, day, ep_id)
        if not bernoulli(rng, CAREGIVER_SPEECH_PROB):
            return
        situation = self._speech_situation(events, rng)
        text = self.caregiver.speak(situation, day=day, episode_id=ep_id)
        if text:
            self._hear(text, events, day=day, situation=situation)

    @staticmethod
    def _speech_situation(events: Sequence[str], rng: Any) -> str:
        if any(str(e).startswith("caregiver_feed") for e in events):
            return "feed"
        if any(str(e).startswith("caregiver_hold") for e in events):
            return "soothe"
        # 「这是什么」只在她正指着东西的时候说——不然问句会变成空气
        if any(str(e).startswith("caregiver_pointed") for e in events):
            return "ask"
        if any(str(e).startswith("caregiver_soothe") for e in events):
            return "soothe"
        return ("play", "greet", "name", "ask")[rng.randrange(4)]

    def _hear(self, text: str, events: list[str], *, day: int, situation: str) -> None:
        events.append(f"heard:{text}")
        for entry in self.lexicon.hear_speech(text, day=day, context=situation):
            events.append(f"word_heard:{entry.form}")
            logger.info("第 %s 天听见一个新词：%s（来自「%s」）", day, entry.form, text)

    def _affordance_limit(self, stage: str) -> int:
        """自选单长度随阶段长大（docs/08）：S0 12、S1 12、S2–S3 16、S4+ 24。

        12 个槽位在两岁那年不够用：走路排在第十三，他永远排不进「今天能做的事」。
        这不是把阈值调松，是承认一个会走的孩子眼前本来就有更多可做的事。
        """
        if stage in ("S2", "S3"):
            return max(self.affordance_limit, 16)
        if stage in ("S4", "S5", "S6"):
            return max(self.affordance_limit, 24)
        return self.affordance_limit

    def _far_target(self, bundle: Any) -> float:
        """视野里有没有「看得见但够不着」的东西。

        学步期的走不是随机游走：一个在他两米外的东西，就是走的理由。
        注意这里看的是**看得见**的对象，不是「正在被注意」的那几个——
        房间那头的东西还没进注意，但它已经在拉着他往那边去了。
        """
        bx, by, _ = self.world.baby_pos
        for obj in self.world.visible_objects():
            if math.dist((bx, by), (obj.pos[0], obj.pos[1])) > 0.8:
                return 1.0
        return 0.0

    def _practice_backlog(self) -> float:
        """积压比例：听见过但还说不出来的词占多少。

        词汇爆发不是「学会了给糖」学出来的。孩子听见的词远多于会说的词，
        这个差距本身就是压力：说不出来的越多，越想说。
        """
        total = len(self.lexicon.entries)
        if total == 0:
            return 0.0
        pending = sum(1 for e in self.lexicon.entries.values() if e.status != "productive")
        return clip01(pending / float(total))

    def _practice_form_for(self, params: Mapping[str, Any]) -> str | None:
        """这一声咿呀在练哪个词：先看音节串，再退回「最该练的那个」。"""
        syllables = str(params.get("syllables") or "")
        entry = self.lexicon.by_form(syllables)
        if entry is not None:
            return entry.form
        for cand in self.lexicon.entries.values():
            if syllables and cand.form.startswith(syllables):
                return cand.form
        practice = self.lexicon.practice_form()
        return practice.form if practice else None

    def _remember(
        self,
        ep_id: str,
        iso: str,
        bundle: Bundle,
        compiled: Any,
        appraisal: Appraisal,
        salience: float,
        stage: str,
        events: Sequence[str] = (),
    ) -> list[str]:
        """什么值得记住。

        只按显著度写记忆，会得到一份几乎空白的童年（30 天两条）。真正该进记忆
        的是别的东西：有人来了、第一次做成某件事、以及被拒绝——docs/08 说拒绝
        本身是可感知的结果，那就得留下痕迹。每天有上限，否则「记住」会退化成
        「全都记」，等于什么都没记。
        """
        if self._memories_today >= MEMORY_PER_DAY_CAP:
            return []
        cared = any(str(e).startswith("caregiver_") for e in events)
        denied = any(str(e).startswith("denied:") for e in events)
        first_time = not bool(self.world.counters.get(f"once:{compiled.primitive}"))
        if not (salience >= 0.45 or (cared and salience >= 0.25) or denied or first_time):
            return []
        self._memories_today += 1
        why = "；".join(
            [
                *(["第一次做到"] if first_time else []),
                *[
                    str(e)
                    for e in events
                    if str(e).startswith("caregiver_") or str(e).startswith("denied:")
                ],
            ]
        )
        mid = self.repo.next_memory_id("episodic")
        mem = {
            "id": mid,
            "type": "episodic",
            "content": f"{bundle.summary()}；随后 {compiled.primitive}" + (f"（{why}）" if why else ""),
            "importance": r6(salience),
            "confidence": 0.6,
            "created": iso,
            "refs": [ep_id],
            "tags": [stage, compiled.primitive],
        }
        self.repo.write_memory(mem)
        self.repo.event("memory.written", ts=iso, ref=mid, actor="memory", episode=ep_id)
        return [mid]

    # -- 结算 ---------------------------------------------------------------
    def _settle_traits(self, day: int, stage: str, prev: Traits) -> dict[str, Any]:
        date_ = self._date_of(day)
        week = _iso_week(date_)
        episodes = self.repo.episodes_of_day(date_.isoformat())
        new = update_traits(
            prev,
            episodes,
            stage=stage,
            week=week,
            rng_seed=f"{self.individual_id}|{date_.isoformat()}",
        )
        self.repo.put_trait(new.to_dict())
        self.repo.event(
            "trait.updated",
            ts=datetime.combine(date_, dtime(hour=DAY_END_HOUR), tzinfo=TZ).isoformat(),
            ref=week,
            actor="traits",
            n_episodes=new.n_episodes,
        )
        return new.to_dict()

    def _judge(self, day: int, stage: str, date_str: str) -> tuple[list[dict[str, Any]], str | None]:
        history = self.repo.stage_history()
        start_day = int(history[-1]["age_days"]) if history else 0
        days_in_stage = day - start_day
        eps = self.repo.episodes_between(self._date_of(start_day).isoformat(), date_str)
        eps = [e for e in eps if e.get("stage") == stage]
        evals = milestones.evaluate(stage, eps)
        now = datetime.combine(self._date_of(day), dtime(hour=DAY_END_HOUR), tzinfo=TZ).isoformat()
        for e in evals:
            self.repo.put_milestone(
                {
                    "id": e.milestone_id,
                    "stage": e.stage,
                    "status": e.status,
                    "value": e.value,
                    "threshold": e.threshold,
                    "evaluated_at": now,
                    "evidence": e.evidence,
                }
            )
            self.repo.event("milestone.evaluated", ts=now, ref=e.milestone_id, actor="stages", status=e.status)

        min_dwell = int(self.cfg.get(f"stage.min_dwell_days.{stage}", 28) or 28)
        decision = judge.can_advance(
            stage, evals, days_in_stage=days_in_stage, min_dwell_days=min_dwell
        )
        if decision.advance and decision.target:
            self.repo.record_stage_advance(
                stage,
                decision.target,
                now,
                day,
                [e.milestone_id for e in evals if e.status == milestones.PASSED],
                list(self.registry.unlocked_at(decision.target)),
            )
            self.repo.event("milestone.evaluated", ts=now, ref=f"{stage}->{decision.target}", actor="stages")
            # 词库的产出阈值随阶段变松（S1 要练 8 次，S2 只要 2 次）。
            # 漏了这一步，两岁那年还在用一岁的标准练词，词汇量永远长不起来。
            self.lexicon.set_stage(decision.target)
            return [e.to_dict() for e in evals], decision.target
        return [e.to_dict() for e in evals], None

    # -- 辅助 ---------------------------------------------------------------
    def _current_traits(self, stage: str) -> Traits:
        rec = self.repo.latest_trait()
        if rec is None:
            t0 = t0_from_temperament(self.temperament, individual_id=self.individual_id, birth_iso=self.birth_iso)
            return Traits(week=_iso_week(self._birth_dt().date()), stage=stage, T=t0)
        return Traits(
            week=str(rec["week"]),
            stage=str(rec.get("stage", stage)),
            T={k: float(v) for k, v in dict(rec["T"]).items()},
            B={k: float(v) for k, v in dict(rec.get("B", {})).items()},
            eta=float(rec.get("eta", 0.0)),
            xi=float(rec.get("xi", 0.0)),
            n_episodes=int(rec.get("n_episodes", 0)),
        )

    def _goal_focus(self, b: Bias, history: History) -> str | None:
        # 注意：这里**不**能再消费 joint_target。上一格被指的东西在 _episode 开头
        # 就已经取走并清空了；这里若再看一眼，就会把它吃掉，
        # 于是「跟随指向」永远发生在指的那一瞬，看起来一次都没跟。
        if self.body.hunger > 0.65:
            return "social"
        if b.explore > 0.6:
            return "interesting"
        return None

    def _birth_dt(self) -> datetime:
        return parse_iso(self.birth_iso)

    def _date_of(self, day: int) -> date:
        return (self._birth_dt() + timedelta(days=day)).date()

    def _load_world(self) -> WorldState | None:
        return WorldState.load(self._world_path) if self._world_path.exists() else None

    def _load_body(self) -> BodyState:
        if not self._body_path.exists():
            return BodyState(
                awake_energy_per_hour=self.cfg.float_("body.awake_energy_per_hour", -3.7),
                sleep_energy_per_hour=self.cfg.float_("body.sleep_energy_per_hour", 6.5),
                hunger_per_hour=self.cfg.float_("body.hunger_per_hour", 0.06),
            )
        d = json.loads(self._body_path.read_text(encoding="utf-8"))
        b = BodyState.from_dict(d)
        b.awake_energy_per_hour = self.cfg.float_("body.awake_energy_per_hour", -3.7)
        b.sleep_energy_per_hour = self.cfg.float_("body.sleep_energy_per_hour", 6.5)
        b.hunger_per_hour = self.cfg.float_("body.hunger_per_hour", 0.06)
        return b

    def _load_lexicon(self) -> Lexicon:
        lex = Lexicon.load(self._lexicon_path) if self._lexicon_path.exists() else Lexicon()
        # 读档时的阶段才是准的：词库里的 stage 可能是上次推进时写下来的
        lex.set_stage(self.stage)
        return lex

    def _load_temperament(self) -> Temperament:
        if self._temperament_path.exists():
            d = json.loads(self._temperament_path.read_text(encoding="utf-8"))
            return Temperament(
                s=float(d.get("S", 0.0)),
                na=float(d.get("NA", 0.0)),
                ec=float(d.get("EC", 0.0)),
                af=float(d.get("AF", 0.0)),
            )
        return sample_temperament(self.individual_id, self.birth_iso)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        """原子写。Windows 上 os.replace 会被杀毒/索引短暂占住，所以重试几次再放弃。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        last: Exception | None = None
        for attempt in range(6):
            try:
                tmp.replace(path)
                return
            except PermissionError as exc:  # WinError 5：目标文件正被别人握着
                last = exc
                time.sleep(0.05 * (attempt + 1))
        raise RuntimeError(f"写不进去：{path}（重试 6 次仍被占用）") from last


class _BudgetedProvider:
    """包一层预算：超了就抛，由 policy 落到启发式。"""

    def __init__(self, inner: LLMProvider, budget: Budget) -> None:
        self.inner = inner
        self.budget = budget

    def prior(self, *, observation: Any, candidates: Any) -> Any:
        if not self.budget.can_call():
            from persona.cognition.policy import Prior

            return Prior(degraded=True, degraded_reason="本 episode 的 LLM 预算已用尽")
        p = self.inner.prior(observation=observation, candidates=candidates)
        self.budget.charge(p)
        return p


def _source(decision: Any, compiled: Any) -> str:
    if compiled.primitive == "A0.NONE":
        return "fallback"
    if decision.prior.degraded:
        return "heuristic"
    return "llm"


def _mean_salience(bundle: Bundle) -> float:
    if not bundle.stimuli:
        return 0.0
    return sum(s.salience for s in bundle.stimuli) / len(bundle.stimuli)


def _iso_week(d: date) -> str:
    iso = d.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def open_simulation(cfg: Config) -> Simulation:
    sim = Simulation(cfg)
    if not sim.born():
        sim.save()
    return sim
