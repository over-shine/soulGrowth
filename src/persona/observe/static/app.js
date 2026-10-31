/* 成长观察台前端：只读、轮询、不做任何写操作。
 *
 * 数据全部来自只读 JSON 接口；这里唯一的状态是「我点了哪一条」这种纯显示状态。
 */

const PRIM = {
  "A0.FIXATE": "注视",
  "A0.TURN_HEAD": "转头",
  "A0.GRASP": "抓握",
  "A0.SUCK": "吮吸",
  "A0.CRY": "哭",
  "A0.STARTLE": "惊跳",
  "A0.SLEEP": "睡",
  "A0.NONE": "什么都不做",
  "A1.REACH": "伸手",
  "A1.CRAWL": "爬",
  "A1.SIT": "坐",
  "A1.POINT": "指",
  "A1.BABBLE": "咿呀",
  "A1.WORD": "说词",
  "A1.GIVE": "给",
  "A1.TAKE": "拿",
  "A1.PEEK": "找",
};

const PRIM_COLOR = {
  "A0.FIXATE": "var(--gaze)",
  "A0.TURN_HEAD": "var(--gaze)",
  "A0.GRASP": "var(--grasp)",
  "A0.SUCK": "var(--suck)",
  "A0.CRY": "var(--cry)",
  "A0.STARTLE": "var(--startle)",
  "A0.SLEEP": "var(--sleep)",
  "A1.REACH": "var(--grasp)",
  "A1.CRAWL": "var(--grasp)",
};

const OBJECTS = { mother: "妈妈", mobile: "床铃", crib: "小床", window: "窗", blanket: "毯子", my_hand: "自己的手" };
const CRY = { hunger: "饿", discomfort: "不舒服", sleepy: "困", lonely: "孤单", pain: "疼" };

const MILESTONES = {
  "M0.1": "视觉定向",
  "M0.2": "注视移动的东西",
  "M0.3": "习惯化",
  "M0.4": "定向反射",
  "M0.5": "抓握反射",
  "M0.6": "哭声分化",
  "M0.7": "认照料者",
  "M1.1": "社会性微笑",
  "M1.2": "客体永久性",
  "M1.3": "深度知觉",
  "M1.4": "陌生人焦虑",
  "M1.5": "分离焦虑",
  "M1.6": "咿呀",
  "M1.7": "共同注意",
  "M1.8": "第一个词",
};

const OCEAN = { O: "开放性", C: "尽责性", E: "外向性", A: "宜人性", N: "神经质" };

const el = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

let selected = null;
let todayDate = null;
let lastDiaryDate = null;
let lastEpisodesKey = "";

async function api(path) {
  const res = await fetch(path, { cache: "no-store" });
  const data = await res.json().catch(() => ({ error: "返回的不是 JSON" }));
  if (!res.ok) throw new Error(data.error || ("HTTP " + res.status));
  return data;
}

function fmtTime(ts) {
  if (!ts) return "--:--";
  const d = new Date(ts);
  if (isNaN(d.getTime())) return String(ts).slice(11, 16);
  return d.toTimeString().slice(0, 8);
}

function humanGap(sec) {
  if (sec == null) return "还没有记录";
  if (sec < 60) return Math.round(sec) + " 秒前";
  if (sec < 3600) return Math.round(sec / 60) + " 分钟前";
  return (sec / 3600).toFixed(1) + " 小时前";
}

function primOf(ep) {
  if (ep.kind === "sleep") return "A0.SLEEP";
  return (ep.action && ep.action.primitive) || "A0.NONE";
}

function labelOf(ep) {
  const p = primOf(ep);
  let name = PRIM[p] || p;
  const act = ep.action || {};
  const tgt = act.target ? OBJECTS[act.target] || act.target : "";
  const params = act.params || {};
  let extra = "";
  if (p === "A0.CRY" && params.reason) extra = "（" + (CRY[params.reason] || params.reason) + "）";
  if (p === "A0.FIXATE" && tgt) extra = " " + tgt;
  else if (tgt && p !== "A0.CRY") extra = " " + tgt;
  if (p === "A0.STARTLE" && act.params && act.params.cause) extra = "（" + act.params.cause + "）";
  return { name, extra };
}

/* -- 渲染 ------------------------------------------------------------------ */

function renderStats(s) {
  const t = s.today || {};
  el("stats").innerHTML = [
    ["今日 episode", t.episodes || 0, "条"],
    ["今日哭", t.cries || 0, "次"],
    ["记忆", (s.counts || {}).memories || 0, "条"],
    ["累计 episode", (s.counts || {}).episodes || 0, "条"],
  ]
    .map(([k, v, u]) => `<div class="stat"><div class="k">${k}</div><div class="v">${v}<small>${u}</small></div></div>`)
    .join("");
  el("today-date").textContent = t.date || "";
}

function renderTimeline(episodes) {
  const box = el("timeline");
  if (!episodes || !episodes.length) {
    box.innerHTML = '<div class="empty">今天还没有记录。等他自己动一下。</div>';
    return;
  }
  const rows = episodes.slice().reverse();
  el("tl-count").textContent = rows.length + " 条 · 最新在上";
  const html = rows
    .map((ep) => {
      const p = primOf(ep);
      const { name, extra } = labelOf(ep);
      const sal = Number(ep.salience || 0);
      const appr = ep.appraisal || {};
      const val = Number(appr.valence || 0);
      const dot = val >= 0.15 ? "var(--good)" : val <= -0.15 ? "var(--bad)" : "var(--ink-faint)";
      const cls = ["ep"];
      if (ep.kind === "sleep") cls.push("sleep");
      if (p === "A0.CRY") cls.push("cry");
      if (ep.catch_up) cls.push("catchup");
      if (selected === ep.id) cls.push("sel");
      const tag = ep.catch_up
        ? '<span class="badge-catchup" title="这段时间是停机后补做的：进库、算里程碑，但权重减半">补</span>'
        : "";
      return `<div class="${cls.join(" ")}" data-id="${esc(ep.id)}">
        <span class="t">${fmtTime(ep.ts).slice(0, 5)}</span>
        <span class="mark" style="background:${dot}" title="效价 ${val.toFixed(2)}"></span>
        <span class="what"><b>${esc(name)}</b>${tag}<span class="target">${esc(extra)}</span>
          <span class="target" style="opacity:.55">${esc((ep.outcome && (ep.outcome.events || [])[0]) || "")}</span></span>
        <span class="gauge" title="显著度 ${sal.toFixed(2)}"><i style="width:${Math.max(2, sal * 100)}%;background:${PRIM_COLOR[p] || "var(--accent)"}"></i></span>
      </div>`;
    })
    .join("");
  box.innerHTML = html;
  box.querySelectorAll(".ep").forEach((node) => {
    node.addEventListener("click", () => selectEpisode(node.dataset.id));
  });
}

function renderMilestones(list) {
  const box = el("milestones");
  if (!list || !list.length) {
    box.innerHTML = '<div class="empty">还没有判定记录。</div>';
    return;
  }
  const order = { passed: 0, failed: 1, insufficient: 2 };
  const sorted = list.slice().sort((a, b) => (order[a.status] ?? 3) - (order[b.status] ?? 3) || String(a.id).localeCompare(String(b.id)));
  const passed = list.filter((m) => m.status === "passed").length;
  el("ms-count").textContent = passed + " / " + list.length + " 通过";
  box.innerHTML = sorted
    .map((m) => {
      const val = m.value == null ? "—" : Number(m.value).toFixed(2);
      const thr = m.threshold == null ? "" : " / " + Number(m.threshold).toFixed(2);
      const statusText = m.status === "passed" ? "通过" : m.status === "failed" ? "未通过" : "样本不足";
      const ratio = m.status === "passed" ? 1 : m.status === "failed" && m.threshold ? Math.min(1, Math.abs(Number(m.value) || 0) / Math.max(0.001, Number(m.threshold))) : 0.04;
      return `<div class="ms ${esc(m.status)}">
        <span class="id">${esc(m.id)}</span>
        <span><div class="name">${esc(MILESTONES[m.id] || m.id)}</div>
          <div class="bar"><i style="width:${Math.max(3, ratio * 100)}%"></i></div></span>
        <span class="st">${statusText} ${val}${thr}</span>
      </div>`;
    })
    .join("");
}

function renderTraits(traits) {
  const box = el("traits");
  const T = (traits && traits.T) || {};
  const B = (traits && traits.B) || {};
  const keys = ["O", "C", "E", "A", "N"];
  el("trait-week").textContent = traits && traits.week ? traits.week + " · n=" + (traits.n_episodes ?? 0) : "";
  const has = keys.some((k) => typeof T[k] === "number");
  if (!has) {
    box.innerHTML = '<div class="empty">还没有结算过特质（每周结算一次）。</div>';
    return;
  }
  box.innerHTML = keys
    .map((k) => {
      const t = Number(T[k] || 0);
      const b = Number(B[k] || 0);
      const seg = (v) => {
        const w = Math.min(50, Math.abs(v) * 50);
        return v >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
      };
      return `<div class="trait">
        <span class="k">${OCEAN[k]}</span>
        <span class="tbar"><span class="mid"></span><span class="t" style="${seg(t)}"></span><span class="b" style="${seg(b)}"></span></span>
        <span class="v">${t.toFixed(2)}</span>
      </div>`;
    })
    .join("");
  el("trait-note").textContent = "青色 = 特质 T（慢变量），橙色 = 本周行为投影 B（快变量）";
}

function renderSpark(trend) {
  const days = (trend && trend.days) || [];
  const box = el("spark");
  if (!days.length) {
    box.innerHTML = '<div class="empty">还没有足够的历史。</div>';
    el("axis").textContent = "";
    return;
  }
  const max = Math.max(1, ...days.map((d) => d.episodes));
  box.innerHTML = days
    .map((d) => {
      const h = Math.max(2, Math.round((d.episodes / max) * 70));
      const c = Math.min(h, Math.round((d.cries / max) * 70));
      return `<div class="col" title="${d.date}：${d.episodes} 条，哭 ${d.cries} 次">
        <i class="cry" style="height:${c}px"></i><i style="height:${Math.max(1, h - c)}px"></i>
      </div>`;
    })
    .join("");
  el("axis").innerHTML = `<span>${esc(days[0].date)}</span><span>${esc(days[days.length - 1].date)}</span>`;
}

function renderEvents(events) {
  const box = el("events");
  if (!events || !events.length) {
    box.innerHTML = '<div class="empty">还没有事件。</div>';
    return;
  }
  box.innerHTML = events
    .map((e) => `<div class="ev"><span class="t">${fmtTime(e.ts).slice(0, 5)}</span>
      <span class="ty"><b>${esc(e.type)}</b> <span style="opacity:.6">${esc(e.ref || "")}</span></span></div>`)
    .join("");
}

/* 极简 Markdown：标题、列表、粗体、段落。够用了，日记是本系统自己写的。 */
function renderMd(md) {
  const lines = String(md || "").split("\n");
  const out = [];
  let list = false;
  const inline = (s) =>
    esc(s)
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, '<span class="mono">$1</span>');
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    if (!line.trim()) {
      if (list) { out.push("</ul>"); list = false; }
      continue;
    }
    if (line.startsWith("## ")) {
      if (list) { out.push("</ul>"); list = false; }
      out.push("<h2>" + inline(line.slice(3)) + "</h2>");
    } else if (line.startsWith("# ")) {
      if (list) { out.push("</ul>"); list = false; }
      out.push("<h1>" + inline(line.slice(2)) + "</h1>");
    } else if (/^\s*[-*]\s+/.test(line)) {
      if (!list) { out.push("<ul>"); list = true; }
      out.push("<li>" + inline(line.replace(/^\s*[-*]\s+/, "")) + "</li>");
    } else if (/^\s*>\s?/.test(line)) {
      if (list) { out.push("</ul>"); list = false; }
      out.push('<p class="muted">' + inline(line.replace(/^\s*>\s?/, "")) + "</p>");
    } else {
      if (list) { out.push("</ul>"); list = false; }
      out.push("<p>" + inline(line) + "</p>");
    }
  }
  if (list) out.push("</ul>");
  return out.join("\n");
}

/* -- 交互 ------------------------------------------------------------------ */

async function selectEpisode(id) {
  selected = id;
  document.querySelectorAll(".ep").forEach((n) => n.classList.toggle("sel", n.dataset.id === id));
  const card = el("detail-card");
  card.hidden = false;
  el("detail-id").textContent = id;
  el("detail").innerHTML = '<div class="empty">读取中…</div>';
  try {
    const data = await api("/api/episode?id=" + encodeURIComponent(id));
    el("detail").innerHTML =
      `<div class="mono" style="color:var(--ink-dim);margin-bottom:6px">${esc(data.ts)} · ${esc(data.stage)} · 第 ${esc(data.age_days)} 天 · 显著度 ${Number(data.salience || 0).toFixed(2)}</div>` +
      '<pre class="json">' + esc(JSON.stringify(data, null, 2)) + "</pre>";
  } catch (err) {
    el("detail").innerHTML = '<div class="err">' + esc(err.message) + "</div>";
  }
}

async function loadDiary(date) {
  const box = el("diary");
  if (!date) { box.innerHTML = '<div class="empty">还没有日记。日记在每天结束时写入。</div>'; return; }
  box.innerHTML = '<div class="empty">读取中…</div>';
  try {
    const data = await api("/api/diary?date=" + encodeURIComponent(date));
    box.innerHTML = renderMd(data.markdown);
  } catch (err) {
    box.innerHTML = '<div class="err">' + esc(err.message) + "</div>";
  }
}

function fillDiarySelect(dates) {
  const sel = el("diary-select");
  el("diary-count").textContent = dates.length ? dates.length + " 天" : "";
  if (!dates.length) { sel.innerHTML = ""; sel.style.display = "none"; return; }
  sel.style.display = "";
  const current = sel.value;
  sel.innerHTML = dates.slice().reverse().map((d) => `<option value="${esc(d)}">${esc(d)}</option>`).join("");
  const pick = current && dates.includes(current) ? current : dates[dates.length - 1];
  sel.value = pick;
  if (pick !== lastDiaryDate) { lastDiaryDate = pick; loadDiary(pick); }
}

el("diary-select").addEventListener("change", (ev) => {
  lastDiaryDate = ev.target.value;
  loadDiary(ev.target.value);
});

/* -- 轮询 ------------------------------------------------------------------ */

async function tick() {
  try {
    const s = await api("/api/state");
    el("name").textContent = (s.individual && s.individual.name) || s.individual.id;
    el("sub").textContent = (s.individual && s.individual.id) + " · 生于 " + String((s.individual || {}).birth || "").slice(0, 16).replace("T", " ");
    el("stage").textContent = s.stage;
    el("age").textContent = "第 " + s.age_days + " 天";
    const live = el("live");
    live.classList.remove("stale", "dead");
    const gap = s.seconds_since_last;
    if (gap == null) el("live-text").textContent = "还没有记录";
    else if (gap < 120) el("live-text").textContent = "活着 · " + humanGap(gap);
    else if (gap < 900) { live.classList.add("stale"); el("live-text").textContent = "慢下来 · " + humanGap(gap); }
    else { live.classList.add("dead"); el("live-text").textContent = "很久没动 · " + humanGap(gap); }

    renderStats(s);
    renderMilestones(s.milestones);
    renderTraits(s.traits);
    fillDiarySelect(s.diary_dates || []);

    todayDate = (s.today || {}).date;
    if (todayDate) {
      const day = await api("/api/day?date=" + encodeURIComponent(todayDate));
      const key = (day.episodes || []).map((e) => e.id).join(",");
      if (key !== lastEpisodesKey) {
        lastEpisodesKey = key;
        renderTimeline(day.episodes || []);
      }
    }
  } catch (err) {
    el("live-text").textContent = "断开了：" + err.message;
    el("live").classList.add("dead");
  }
}

async function slowTick() {
  try {
    const [trend, events] = await Promise.all([api("/api/trend"), api("/api/events?limit=60")]);
    renderSpark(trend);
    renderEvents(events.events);
  } catch (err) {
    /* 慢频道失败不该把页面弄脏 */
  }
}

function clock() {
  const d = new Date();
  el("clock").textContent = d.toTimeString().slice(0, 8) + " · 浏览器本地时间";
}

clock();
tick();
slowTick();
setInterval(clock, 1000);
setInterval(tick, 2000);
setInterval(slowTick, 8000);
