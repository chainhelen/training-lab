import { lessons, glossary, references } from "./lessons.mjs";
import {
  ids,
  tokens,
  start,
  draw,
  config,
  preview,
  update,
  stochastic,
} from "./lab.mjs";
import { probabilities } from "./trainer.mjs";

const $ = (id) => document.getElementById(id);
const key = "training-lab-v1";
const esc = (x) =>
  String(x).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const row = (v) =>
  Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);
function read() {
  try {
    return JSON.parse(localStorage.getItem(key)) || {};
  } catch {
    return {};
  }
}
const saved = read();
let preferredLanguage;
try {
  preferredLanguage = localStorage.getItem("training-lab-language");
} catch {}
let lang = (preferredLanguage || saved.lang) === "zh" ? "zh" : "en";
const t = (en, zh) => (lang === "en" ? en : zh);
const txt = (o) => o[lang];
const name = (id) =>
  ({
    pretrain: t("Pretraining", "预训练"),
    sft: "SFT",
    dpo: "DPO",
    rl: "RL · REINFORCE",
    grpo: "GRPO",
  })[id];
const checkpoints = Object.fromEntries(
  ids
    .filter((id) => row(saved.checkpoints?.[id]))
    .map((id) => [id, saved.checkpoints[id]]),
);
const sessions = {};
function valid(id, s) {
  if (
    !s ||
    s.id !== id ||
    !row(s.logits) ||
    !row(s.reference) ||
    !row(s.counts) ||
    !Array.isArray(s.trace) ||
    !s.trace.length ||
    !s.trace.every(Number.isFinite) ||
    !Number.isSafeInteger(s.step) ||
    s.step < 0 ||
    ![s.target, s.chosen, s.rejected].every((a) => [0, 1, 2].includes(a))
  )
    return false;
  if (!s.counts.every((x) => Number.isSafeInteger(x) && x >= 0 && x <= 99))
    return false;
  if (
    stochastic(id) &&
    (!row(s.old) ||
      !Array.isArray(s.actions) ||
      s.actions.length !== 8 ||
      !s.actions.every((a) => [0, 1, 2].includes(a)))
  )
    return false;
  if (
    s.last &&
    (!row(s.last.before) ||
      !row(s.last.after) ||
      !Number.isFinite(s.last.loss) ||
      !s.last.metrics)
  )
    return false;
  return true;
}
for (const id of ids)
  if (valid(id, saved.sessions?.[id])) sessions[id] = saved.sessions[id];
const requested = location.hash.slice(1);
let id = ids.includes(requested)
  ? requested
  : ids.includes(saved.id)
    ? saved.id
    : "pretrain";
function persist() {
  try {
    localStorage.setItem(
      key,
      JSON.stringify({ lang, id, checkpoints, sessions }),
    );
  } catch {}
}
const state = () => (sessions[id] ??= start(id, checkpoints));
let openNote = false;
const fixed = (x) => Number(x).toFixed(2);
const pct = (x) => (100 * x).toFixed(1) + "%";
const strings = {
  ready: [
    "Choose a training signal and apply one update.",
    "选好训练信号，再更新一次。",
  ],
  empty: [
    "Add at least one log line to the batch.",
    "先在这批语料里加入至少一行日志。",
  ],
  duplicate: ["Choose two different replies.", "请选择两个不同的回答。"],
  draw: [
    "Draw a fresh rollout from the updated policy before the next REINFORCE step.",
    "策略已经更新，先用新策略采样，再做下一步 REINFORCE。",
  ],
  sampled: [
    "Eight new replies are ready to score.",
    "已从当前策略采样八份回答，可以打分了。",
  ],
  updated: [
    "Check the probabilities. You can apply another update.",
    "看一下概率变化，还可以再更新一次。",
  ],
  mix: [
    "The model learned from your batch. Try exactly two HOLD lines and one GO line.",
    "模型已经按这批语料更新。这一关需要恰好两行 HOLD 和一行 GO。",
  ],
  demo: [
    "That demonstration taught a different command. Pick the command for stopping.",
    "这个示范教给了模型另一个命令。换成停车指令再试试。",
  ],
  pair: [
    "Your preference moved the odds. Which reply should beat GO for this instruction?",
    "这对偏好改变了概率比。按这条指令，哪个回答应该胜过 GO？",
  ],
  shortSeen: [
    "The short-answer reward made GO more likely. Now reward the correct command and draw again.",
    "奖励短回答让 GO 更容易出现。现在换成奖励正确指令，再采一组。",
  ],
  noReward: [
    "This rollout earned no reward. Draw a new group and try again.",
    "这组所有回答的奖励都是 0。再采一组试试。",
  ],
  needShort: [
    "First observe what happens when the shortest answer gets the reward.",
    "先观察只奖励最短回答时，GO 的概率会怎么变。",
  ],
  flatSeen: [
    "All eight advantages are zero. Keep the group and switch to the HOLD reward.",
    "八个优势值全是 0。保留这组回答，切换成只奖励 HOLD。",
  ],
  needFlat: [
    "First run one equal-reward update with KL off.",
    "先关闭 KL 惩罚，让所有回答同分，更新一次。",
  ],
  homogeneous: [
    "All eight replies received the same reward. Draw again for a comparison within the group.",
    "这八个回答全都同分，再采一组含不同得分的回答。",
  ],
  redraw: [
    "Draw a new group from the current policy.",
    "请用当前策略重新采一组。",
  ],
  pass: [
    "Goal reached. Your checkpoint is saved.",
    "目标已达成，通关时的模型参数已保存。",
  ],
};
function goal() {
  return {
    pretrain: t(
      "Use 2 HOLD lines and 1 GO line. Lift HOLD to 45% and keep GO above CHECK.",
      "用 2 行 HOLD 和 1 行 GO，把 HOLD 提到 45%，并让 GO 高于 CHECK。",
    ),
    sft: t(
      "Teach the stop command until HOLD reaches 60%.",
      "用停车指令作示范，让 HOLD 的概率达到 60%。",
    ),
    dpo: t(
      "Prefer HOLD over GO until their odds are at least 4.3× the starting odds.",
      "让 HOLD 胜过 GO，直到两者的概率比达到起点的 4.3 倍。",
    ),
    rl: t(
      "First make GO more likely with a brevity reward. Then change the reward to raise HOLD.",
      "先奖励最短回答，让 GO 概率上升；再改奖励，让 HOLD 概率上升。",
    ),
    grpo: t(
      "Observe an equal-reward update with KL off. Then raise HOLD using differences within a group.",
      "先关掉 KL，观察全组同分时的一次更新。再利用组内分差提高 HOLD 概率。",
    ),
  }[id];
}
function select(key, label, opts, value) {
  return `<label>${esc(label)} <select id="${key}">${opts.map(([v, n]) => `<option value="${v}" ${String(v) === String(value) ? "selected" : ""}>${esc(n)}</option>`).join("")}</select></label>`;
}
function renderInputs(s) {
  const options = tokens.map((v, i) => [i, v]);
  if (id === "pretrain") {
    return `<h3>${t("Add a line to the corpus", "往语料里加一行日志")}</h3><p class="small muted">${t("Each click adds the prompt followed by that token.", "每点一次，就加入一行「前文 + 所选 token」。")}</p><div class="control-row">${tokens.map((v, i) => `<button data-count="${i}">+ ${v} <b>· ${s.counts[i]}</b></button>`).join("")}</div><p class="small muted">${t("Batch size", "当前样本数")} · ${s.counts.reduce((a, b) => a + b, 0)} &nbsp; <button id="empty">${t("Clear batch", "清空语料")}</button></p>`;
  }
  if (id === "sft")
    return select(
      "target",
      t("Demonstration answer", "示范回答"),
      options,
      s.target,
    );
  if (id === "dpo")
    return `<div class="control-row">${select("chosen", t("Preferred", "更喜欢"), options, s.chosen)}${select("rejected", t("Rejected", "更不喜欢"), options, s.rejected)}</div><p class="small muted">${t("The reference keeps the probabilities from the start of this experiment.", "参考模型固定使用本关开始时的概率。")}</p>`;
  const rules =
    id === "rl"
      ? [["short", t("1 for the shortest reply", "最短回答得 1 分")]]
      : [["flat", t("1 for every reply", "所有回答都得 1 分")]];
  rules.push([
    "exact",
    t("1 for HOLD, otherwise 0", "HOLD 得 1 分，其余得 0 分"),
  ]);
  const c = config(s),
    p = preview(s);
  return `<div class="control-row">${select("rule", t("Reward rule", "奖励规则"), rules, s.rule)}${
    id === "grpo"
      ? select(
          "kl",
          t("KL penalty", "KL 惩罚"),
          [
            [0, t("Off", "关闭")],
            [0.1, "0.1"],
          ],
          s.kl,
        )
      : ""
  }</div><h3>${t("Sampled replies", "采样回答")} · 8</h3><p class="small muted">${id === "grpo" ? t("Each chip shows the reward and group-relative advantage for the selected rule.", "每格显示当前规则下的奖励 R 与组内优势 A。") : t("Each chip shows the reward for one sampled reply.", "每格显示一个采样回答与它获得的奖励 R。")}</p><div class="rollout">${s.actions.map((a, i) => `<span class="sample ${c.rewards[i] ? "plus" : "minus"}" data-action="${a}"><strong>${tokens[a]}</strong><br>R ${c.rewards[i]}${id === "grpo" ? " · A " + fixed(p.metrics.advantages[i]) : ""}</span>`).join("")}</div>`;
}
function trace(s) {
  const a = s.trace,
    w = 400,
    h = 58;
  const x = (i) => 8 + ((w - 16) * i) / Math.max(8, a.length - 1),
    y = (v) => 4 + (h - 8) * (1 - v);
  const points = a.map((v, i) => x(i) + "," + y(v)).join(" ");
  return `<p class="small muted">${t("P(HOLD) across updates", "历次更新后的 P(HOLD)")} · ${Math.max(0, s.step - a.length + 1)}–${s.step}</p><svg class="trace" viewBox="0 0 ${w} ${h}" role="img" aria-label="P(HOLD) ${pct(a.at(-1))}"><path d="M8 29H392 M8 54H392" stroke="#d7d6ca" stroke-dasharray="2 4"/><polyline points="${points}" fill="none" stroke="currentColor" stroke-width="2.5"/><circle cx="${x(a.length - 1)}" cy="${y(a.at(-1))}" r="3.5" fill="currentColor"/></svg>`;
}
function stats(s) {
  const last = s.last,
    m = last?.metrics;
  const vals = [
    [t("Last loss¹", "上一步损失¹"), last ? fixed(last.loss) : "—"],
  ];
  if (id === "dpo")
    vals.push([
      t("Odds / reference", "概率比 / 参考值"),
      fixed(
        Math.exp(
          Math.min(
            60,
            s.logits[0] - s.logits[1] - (s.reference[0] - s.reference[1]),
          ),
        ),
      ) + "×",
    ]);
  if (stochastic(id))
    vals.push([
      t("Last mean reward", "上一步平均奖励"),
      m ? fixed(m.meanReward) : "—",
    ]);
  if (id === "grpo" && m)
    vals.push(
      [
        t("Clipped samples¹", "受裁剪样本¹"),
        m.clipped.filter(Boolean).length + "/8",
      ],
      ["KL¹", fixed(m.kl)],
    );
  return vals
    .map(([k, v]) => `<span class="stat">${k} <b>${v}</b></span>`)
    .join("");
}
const equations = {
  pretrain: "L = −Σᵢ (countᵢ / N) log pᵢ",
  sft: "L = −log p(demonstration | prompt)",
  dpo: "L = −log σ(β [log(p⁺ / p⁻) − log(ref⁺ / ref⁻)])",
  rl: "Δθ ∝ mean[(R − baseline) ∇ log π(answer | prompt)]",
  grpo: "Aᵢ = (Rᵢ − mean(R)) / (std(R) + ε)\nrᵢ = π(answerᵢ) / π_old(answerᵢ)\nJ = mean[min(rᵢAᵢ, clip(rᵢ, 0.8, 1.2) Aᵢ)] − λ KL(π ‖ ref)",
};
function explainLastStep(s) {
  const last = s.last;
  if (!last?.logitsBefore) return "";
  const gradient = last.metrics.gradient[0];
  const formula = `HOLD: ${fixed(last.logitsBefore[0])} + ${last.lr} × (${fixed(gradient)}) ≈ ${fixed(s.logits[0])}`;
  return `<h3>${t("Your last update", "你刚才的这一步")}</h3><p>${t("New score = old score + learning rate × update direction. Softmax then uses all three new scores to get the probability.", "新分数 = 旧分数 + 学习率 × 更新方向。三个分数一起过 softmax，才得到新的概率。")}</p><pre>${formula}
P(HOLD): ${pct(last.before[0])} → ${pct(last.after[0])}</pre>`;
}
function updateGuide() {
  const intro = t(
    "A three-token policy makes the updates small enough to inspect. The probabilities are computed from three trainable logits.",
    "这个小模型只有三个可训练的分数（logit），经 softmax 变成三个回答的概率。每一步都会用梯度更新这些分数。",
  );
  $("guide-body").innerHTML =
    `<h2>${t("Field notes", "训练手册")}</h2><p>${intro}</p><p>${t("SFT can start from your pretraining checkpoint. DPO, REINFORCE and GRPO each branch from SFT. Opening a stage early loads a starter policy. Resetting a stage reloads its latest available starting checkpoint.", "SFT 可以从你的预训练通关点开始。DPO、REINFORCE 和 GRPO 分别从 SFT 分支出发。提前跳关会使用内置起点。本关重置时，会重新读取已有的上游通关点。")}</p>${glossary.map((g) => `<h3>${esc(g.term)}</h3><p>${esc(txt(g.text))}</p>`).join("")}<h3>${t("Read the implementations", "阅读实现")}</h3><ul>${references.map((r) => `<li><a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.label)}</a></li>`).join("")}</ul>`;
}
function render() {
  const s = state(),
    idx = ids.indexOf(id),
    lesson = lessons.find((l) => l.id === id),
    probs = probabilities(s.logits);
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.title = txt(lesson.title) + " · Training Lab";
  const returnCourse = $("return-course");
  if (returnCourse) {
    returnCourse.textContent = t("Guided lessons", "拆解练习");
    returnCourse.href = "./#learn/" + id;
  }
  $("games").textContent = t("Games", "游戏大厅");
  $("guide").textContent = t("Field notes", "训练手册");
  $("language").textContent = t("中文", "English");
  $("chapters").innerHTML = ids
    .map(
      (k, i) =>
        `<button data-lesson="${k}" class="${checkpoints[k] ? "done" : ""} ${id === k ? "active" : ""}" ${id === k ? 'aria-current="step"' : ""}>${checkpoints[k] ? "✓" : String(i + 1).padStart(2, "0")} &nbsp; ${name(k)}</button>`,
    )
    .join("");
  $("kicker").textContent =
    (id === "pretrain"
      ? t("Foundation", "基础训练")
      : t("Post-training", "后训练")) +
    " / " +
    name(id);
  $("title").textContent = txt(lesson.title);
  $("brief").textContent = txt(lesson.brief);
  $("progress").textContent =
    Object.keys(checkpoints).length + " / 5 " + t("completed", "已通关");
  $("method").textContent = name(id);
  $("inputs-title").textContent = t("Training bench", "训练工作台");
  $("prompt-label").textContent =
    id === "pretrain"
      ? t("LOG PREFIX", "日志前文")
      : t("USER INSTRUCTION", "用户指令");
  $("prompt").textContent =
    id === "pretrain"
      ? "The signal says …"
      : "Stop the rover. Reply with one command.";
  $("inputs").innerHTML = renderInputs(s);
  $("sample").hidden = !stochastic(id);
  $("sample").textContent = t("Draw 8 replies", "重新采样 8 份");
  $("reset").textContent = t("Reset stage", "本关重置");
  $("step").textContent = t("Train one step", "训练一步");
  $("policy-title").textContent = t("Output probabilities", "回答概率");
  $("step-num").textContent =
    t("UPDATE ", "更新 ") + String(s.step).padStart(2, "0");
  $("chart-caption").textContent = t(
    "Bar = current policy   ·   Tick = before the last update",
    "色条表示当前概率，黑线标记上一步更新前的概率。",
  );
  $("chart").innerHTML = tokens
    .map(
      (token, i) =>
        `<div class="bar-row"><div class="bar-head"><b>${token}</b><span>${pct(probs[i])}</span></div><div class="bar-track"><span class="bar-fill" style="width:${pct(probs[i])}"></span><span class="old-mark" style="left:${pct(s.last?.before[i] ?? probs[i])}"></span></div></div>`,
    )
    .join("");
  $("history").innerHTML =
    trace(s) +
    `<p class="muted small">¹ ${t("Measured before applying the last gradient.", "上次应用梯度前计算。")}</p>`;
  $("stats").innerHTML = stats(s);
  $("goal-label").textContent = t("Experiment goal", "本关目标");
  $("goal-text").textContent = goal();
  const status = s.status || "ready";
  $("feedback").textContent = t(...(strings[status] || strings.ready));
  $("feedback").className = s.won ? "won" : "";
  document.querySelector(".goal").classList.toggle("won", s.won);
  $("advance").innerHTML = s.won
    ? idx < ids.length - 1
      ? `<button id="next">${t("Continue to", "继续到")} ${name(ids[idx + 1])}</button>`
      : `<strong>${Object.keys(checkpoints).length === ids.length ? t("All five experiments complete.", "五项训练实验全部完成。") : t("Checkpoint saved. Choose another experiment above.", "通关点已保存，可从上方选择其他实验。")}</strong>`
    : "";
  $("notes").innerHTML =
    `<details ${openNote ? "open" : ""}><summary>${t("How this update works", "这一步具体怎么算")}</summary><p>${esc(txt(lesson.learn))}</p><pre>${esc(equations[id])}</pre>${explainLastStep(s)}<p>${esc(txt(lesson.hint))}</p><p>${s.parent ? t("Initialized from your checkpoint", "初始化来自你的通关点") + " · " + name(s.parent) : t("Initialized from the built-in starting policy.", "本关从内置模型参数开始。")}</p><a href="${esc(lesson.source.url)}" target="_blank" rel="noopener">${esc(lesson.source.label)}</a></details>`;
  updateGuide();
  persist();
}
function navigate(target) {
  if (id === target) return;
  location.hash = target;
}
window.addEventListener("hashchange", () => {
  const target = location.hash.slice(1);
  if (ids.includes(target)) {
    id = target;
    openNote = false;
    render();
    document
      .querySelector(".chapters .active")
      ?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }
});
$("notes").addEventListener(
  "toggle",
  (ev) => {
    if (ev.target.tagName === "DETAILS") openNote = ev.target.open;
  },
  true,
);
document.addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (!b) return;
  const s = state();
  if (b.dataset.lesson) {
    navigate(b.dataset.lesson);
    return;
  }
  if (b.dataset.count !== undefined) {
    const k = Number(b.dataset.count);
    s.counts[k] = Math.min(99, s.counts[k] + 1);
    s.status = "ready";
    render();
    document
      .querySelector('[data-count="' + k + '"]')
      .focus({ preventScroll: true });
    return;
  }
  switch (b.id) {
    case "guide":
      $("manual").showModal();
      return;
    case "close":
      $("manual").close();
      return;
    case "language":
      lang = lang === "en" ? "zh" : "en";
      try {
        localStorage.setItem("training-lab-language", lang);
      } catch {}
      break;
    case "step":
      s.status = update(s);
      if (s.status === "pass" && !checkpoints[id])
        checkpoints[id] = [...s.logits];
      break;
    case "reset":
      sessions[id] = start(id, checkpoints);
      break;
    case "sample":
      draw(s);
      s.status = "sampled";
      break;
    case "empty":
      s.counts = [0, 0, 0];
      s.status = "ready";
      break;
    case "next": {
      const next = ids[ids.indexOf(id) + 1];
      if (new URLSearchParams(location.search).has("course")) {
        location.assign("./#learn/" + next);
      } else {
        navigate(next);
      }
      return;
    }
    default:
      return;
  }
  render();
});
$("inputs").addEventListener("change", (ev) => {
  const k = ev.target.id;
  if (!["target", "chosen", "rejected", "kl", "rule"].includes(k)) return;
  const s = state();
  s[k] = k === "rule" ? ev.target.value : Number(ev.target.value);
  s.status = "ready";
  render();
  $(k)?.focus({ preventScroll: true });
});
render();
