import {
  stations,
  missions,
  missionUrl,
  freshDeal,
  readDeal,
  replay,
} from "./arcade-engine.mjs";

import { probabilities } from "./trainer.mjs";

const get = (id) => document.getElementById(id);
const game = get("game");
const memory = new Map();
function load(key) {
  try {
    return localStorage.getItem(key) ?? memory.get(key);
  } catch {
    return memory.get(key);
  }
}
function save(key, value) {
  memory.set(key, value);
  try {
    localStorage.setItem(key, value);
  } catch {}
}
function savedDeal(id) {
  try {
    return readDeal(JSON.parse(load("training-arcade-v1/" + id)));
  } catch {
    return freshDeal();
  }
}
const cleared = (m) => load("training-arcade-cleared/" + m.id) === "1";
let language = load("training-lab-language") === "zh" ? "zh" : "en";
const t = (en, zh) => (language === "zh" ? zh : en);
const label = (pair) => t(...pair);
const escape = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        ch
      ],
  );
let mission, station, deal;

function words() {
  document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  get("language").textContent = t("中文", "English");
  get("lessons").textContent = t("Guided lessons", "拆解练习");
  get("games").textContent = t("Games", "游戏大厅");
}
function progress(items) {
  return items
    .map((m) => `<i class="${cleared(m) ? "earned" : ""}"></i>`)
    .join("");
}
function overview() {
  const total = missions.filter(cleared).length;
  const recent = missions.find((m) => m.id === load("training-arcade-last"));
  const next =
    (recent && !cleared(recent) ? recent : missions.find((m) => !cleared(m))) ??
    missions[0];
  const title = t("Corpus Arcade", "语料挑战场");
  document.title = title + " · Training Lab";
  game.innerHTML = `<div class="arc-hero"><div><p class="arc-eyebrow">${t("PRETRAINING", "预训练")}</p><h1 tabindex="-1">${title}</h1><p>${t("Collect records, inspect a split, and train a rover on a token budget. Each mission gives you a new decision to make.", "给小车收集日志、整理训练集、安排每批数据。每关都有一份任务单，交卷时就能看到自己的决定带来了什么。")}</p></div><div class="arc-summary"><span>${t("Missions cleared", "已通关")}</span><strong>${total} <small>/ ${missions.length}</small></strong><div class="arc-stamps" aria-hidden="true">${progress(missions)}</div></div></div>
  <a class="arc-resume" href="${missionUrl(next)}"><span>${t(total === missions.length ? "Play again" : recent ? "Continue playing" : "Start playing", total === missions.length ? "再次挑战" : recent ? "继续挑战" : "开始挑战")}</span><strong>${escape(label(next.title))} <b aria-hidden="true">›</b></strong></a>
  <div class="station-grid">${stations.map((s) => `<section class="station ${s.id}"><div class="station-cover" aria-hidden="true">${stationArt(s.id)}</div><div class="station-body"><h2>${escape(label(s.name))}</h2><p>${escape(label(s.brief))}</p><ol>${s.missions.map((m, i) => `<li><a data-mission="${escape(m.id)}" href="${missionUrl(m)}"><span class="mission-n">${cleared(m) ? "✓" : String(i + 1).padStart(2, "0")}</span>${escape(label(m.title))}<span class="arrow" aria-hidden="true">›</span></a></li>`).join("")}</ol></div></section>`).join("")}</div>`;
}
function stationArt(id) {
  if (id === "budget")
    return '<div class="paper-samples"><b>HOLD</b><b>GO</b><b>CHECK</b></div>';
  if (id === "audit")
    return '<div class="audit-sheets"><i>≡<small>01</small></i><i>≡<small>01</small></i><i>≣<small>02</small></i></div>';
  if (id === "trials")
    return '<div class="trial-cover"><span>A</span><i></i><i></i><i></i><span>B</span><i></i><i></i><i></i></div>';
  return (
    '<div class="mix-stack">' +
    [3, 5, 2, 4, 3, 5, 2, 3].map((n) => `<i style="--h:${n}"></i>`).join("") +
    "</div>"
  );
}
function boardView(board) {
  if (!board?.columns?.length) return "";
  return `<div class="board-scroll" tabindex="0"><table><thead><tr>${board.columns.map((h) => `<th scope="col">${escape(h)}</th>`).join("")}</tr></thead><tbody>${board.rows.map((row) => `<tr>${row.map((cell, i) => (i === 0 ? `<th scope="row">${escape(cell)}</th>` : `<td>${escape(cell)}</td>`)).join("")}</tr>`).join("")}</tbody></table></div>`;
}
function boardGraphic(id, state, frame) {
  if (id === "collect-three") {
    const words = ["HOLD", "GO", "CHECK"];
    return `<div class="cargo-slots">${[0, 1, 2]
      .map((i) => {
        const record = state.taken[i];
        return `<div class="${record === undefined ? "vacant" : ""}"><small>${t("Seat", "座位")} ${i + 1}</small><b>${record === undefined ? "—" : words[state.deck[record]]}</b></div>`;
      })
      .join("")}</div>`;
  }
  if (id === "split-by-trip") {
    return `<div class="trip-chart">${frame.board.rows.map(([family, train, test]) => `<div><b>${escape(family)}</b><span class="trip-marks" aria-hidden="true">${"<i></i>".repeat(train)}${'<i class="held"></i>'.repeat(test)}</span></div>`).join("")}<small>${t("Purple · train　Teal · validation", "紫色 · 训练　绿色 · 验证")}</small></div>`;
  }
  if (Array.isArray(state.logits) && state.logits.length === 3) {
    const p = probabilities(state.logits);
    return `<div class="model-chart"><small>${t("Current next-token probabilities", "当前模型的下一个词概率")}</small>${["HOLD", "GO", "CHECK"].map((name, i) => `<div><b>${name}</b><span><i style="width:${p[i] * 100}%"></i></span><em>${(p[i] * 100).toFixed(1)}%</em></div>`).join("")}</div>`;
  }
  return "";
}
function playView(focusCard) {
  const { state, moves } = replay(mission, deal, t);
  const frame = mission.view(state, t);
  const result = deal.submitted ? mission.grade(state, t) : null;
  const next = missions[missions.indexOf(mission) + 1];
  const title = label(mission.title);
  document.title = title + " · " + t("Corpus Arcade", "语料挑战场");
  game.innerHTML = `<article class="mission ${station.id}"><div class="arc-breadcrumb"><a href="#menu">${t("All missions", "全部挑战")}</a> / ${escape(label(station.name))}<span>${t("Deal", "牌局")} ${deal.seed}</span></div>
  <h1 tabindex="-1">${escape(title)}</h1><p class="mission-brief">${escape(label(mission.brief))}</p>
  <div class="goal-strip"><strong>${t("Your mission", "本关目标")}</strong><span>${escape(label(mission.target))}</span></div>
  <div class="mission-work"><section class="move-panel"><h2>${t("Your moves", "轮到你了")}</h2><div class="move-grid">${frame.cards.map((c) => `<button type="button" class="move-card" data-move="${escape(c.id)}" ${c.disabled ? "disabled" : ""}>${c.tag ? `<small>${escape(c.tag)}</small>` : ""}<strong>${escape(c.title)}</strong>${c.detail ? `<span>${escape(c.detail)}</span>` : ""}</button>`).join("")}</div></section>
  <section class="state-panel"><h2>${t("Game board", "当前局面")}</h2><dl class="stat-grid">${frame.stats.map((r) => `<div><dt>${escape(r.label)}</dt><dd>${escape(r.value)}</dd></div>`).join("")}</dl>
  ${boardGraphic(mission.id, state, frame)}${boardView(frame.board)}${frame.note ? `<p class="frame-note">${escape(frame.note)}</p>` : ""}</section></div>
  <div class="arc-controls"><button id="submit" class="arc-primary">${t("Check result", "交卷验收")}</button><button id="undo" ${moves.length ? "" : "disabled"}>${t("Undo last move", "撤销上一步")}</button><button id="restart" ${moves.length || deal.submitted ? "" : "disabled"}>${t("Replay this deal", "本局重来")}</button><button id="redeal">${t("New deal", "换一局")}</button></div>
  <div id="verdict" aria-live="polite" ${result ? 'class="' + (result.ok ? "win" : "retry") + '"' : ""}>${result ? `<strong>${result.ok ? t("Mission cleared", "通关") : t("Try again", "再试一次")}</strong> ${escape(result.message)}` : ""}</div>
  ${result?.ok ? `<a id="next-mission" href="${next ? missionUrl(next) : "#menu"}">${next ? t("Next mission", "下一关") + " · " + escape(label(next.title)) : t("Return to mission board", "回到挑战列表")} ›</a>` : ""}
  <div class="mission-foot"><details><summary>${t("Move history", "操作记录")} · ${moves.length}</summary><ol>${moves.map((m) => `<li>${escape(m.title)}</li>`).join("")}</ol></details><a href="./#learn/pretrain/${escape(mission.guide)}">${t("See the calculation", "回教程看计算")}</a></div></article>`;
  if (focusCard)
    Array.from(game.querySelectorAll("[data-move]"))
      .find((b) => b.dataset.move === focusCard && !b.disabled)
      ?.focus({ preventScroll: true });
}
function remember() {
  save("training-arcade-v1/" + mission.id, JSON.stringify(deal));
}
function render() {
  words();
  if (mission) playView();
  else overview();
}
const related = {
  "next-target": "pack-windows",
  "count-endings": "collect-three",
  "score-to-chance": "reread-or-open",
  "token-loss": "reread-or-open",
  "batch-loss": "nine-target-race",
  "update-direction": "schedule-updates",
  "batch-tokens": "schedule-updates",
  "same-size-sample": "collect-three",
  "matching-batches": "equalize-trials",
  "copied-weight": "copy-desk",
};
function route() {
  if (location.hash.startsWith("#lesson/")) {
    const guideId = location.hash.slice("#lesson/".length);
    const match =
      missions.find((m) => m.guide === guideId) ??
      missions.find((m) => m.id === related[guideId]);
    history.replaceState(null, "", match ? missionUrl(match) : "#menu");
  }
  mission = missions.find((m) => location.hash === missionUrl(m));
  if (mission) {
    station = stations.find((s) => s.missions.includes(mission));
    deal = savedDeal(mission.id);
    save("training-arcade-last", mission.id);
  }
  render();
}
game.addEventListener("click", (event) => {
  const button = event.target.closest("button");
  if (!button || button.disabled || !mission) return;
  const id = button.dataset.move;
  if (id !== undefined) {
    deal.actions.push(id);
    deal.submitted = false;
  } else if (button.id === "undo") {
    deal.actions.pop();
    deal.submitted = false;
  } else if (button.id === "restart") {
    deal = freshDeal(deal.seed);
  } else if (button.id === "redeal") {
    deal = freshDeal(deal.seed + 1);
  } else if (button.id === "submit") {
    deal.submitted = true;
    if (mission.grade(replay(mission, deal, t).state, t).ok)
      save("training-arcade-cleared/" + mission.id, "1");
  } else return;
  remember();
  playView(id);
  if (id === undefined) get(button.id)?.focus({ preventScroll: true });
});
get("language").onclick = () => {
  language = language === "en" ? "zh" : "en";
  save("training-lab-language", language);
  render();
};
window.addEventListener("hashchange", () => {
  route();
  game.querySelector("h1").focus({ preventScroll: true });
  window.scrollTo(0, 0);
});
window.addEventListener("pageshow", (e) => {
  if (e.persisted) {
    language = load("training-lab-language") === "zh" ? "zh" : "en";
    route();
  }
});
route();
