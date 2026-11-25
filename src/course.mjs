import {
  courses,
  valuesFor,
  lessonKey,
  lessonUrl,
  resolveRoute,
} from "./course-data.mjs";
import { renderGroupVisual } from "./group-visual.mjs";
import { renderVisual } from "./course-visuals.mjs";
import { lessons } from "./lessons.mjs";

const $ = (id) => document.getElementById(id);
const storageKey = "training-lab-guided-v1";
const html = (v) =>
  String(v).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[
        c
      ],
  );
function stored(k, fallback) {
  try {
    return JSON.parse(localStorage.getItem(k)) ?? fallback;
  } catch {
    return fallback;
  }
}
const load = stored(storageKey, {}),
  legacy = stored("training-lab-v1", {});
const validKeys = new Set(
  courses.flatMap((c) => c.steps.map((s) => lessonKey(c, s))),
);
const visited = new Set(
  Array.isArray(load.done) ? load.done.filter((id) => validKeys.has(id)) : [],
);
const store = {
  positions: { ...(load.positions || {}) },
  work: { ...(load.work || {}) },
  last: load.last,
};
let lang = legacy.lang === "zh" ? "zh" : "en";
try {
  const pref = localStorage.getItem("training-lab-language");
  if (pref === "en" || pref === "zh") lang = pref;
} catch {}
const t = (en, zh) => (lang === "zh" ? zh : en);
const text = (pair) => t(...pair);
const narrow = window.matchMedia("(max-width:760px)");
$("rail").open = !narrow.matches;
let course, step, activeKey, values, pinned, review, passed;
function save() {
  store.work[activeKey] = { values, pinned, review };
  store.positions[course.id] = step.id;
  store.last = activeKey;
  try {
    localStorage.setItem(
      storageKey,
      JSON.stringify({ ...store, done: [...visited] }),
    );
    localStorage.setItem("training-lab-language", lang);
  } catch {}
}
function capstone(c) {
  return "./workbench.html?course=1#" + c.id;
}
function routing() {
  const hash = location.hash.slice(1);
  if (courses.some((c) => c.id === hash)) {
    location.replace("./workbench.html#" + hash);
    return;
  }
  const target = hash.startsWith("learn/")
    ? hash
    : "learn/" +
      (validKeys.has(store.last)
        ? store.last
        : lessonKey(courses[0], courses[0].steps[0]));
  ({ course, step } = resolveRoute(target, store));
  activeKey = lessonKey(course, step);
  const saved = store.work[activeKey] || {};
  values = valuesFor(step, saved.values);
  pinned = saved.pinned ? valuesFor(step, saved.pinned) : null;
  review = !!saved.review;
  passed = review && step.check(values);
  if (location.hash !== lessonUrl(course, step))
    history.replaceState(null, "", lessonUrl(course, step));
  render();
  save();
}
function nav() {
  const index = course.steps.indexOf(step),
    total = course.steps.length;
  $("chapters").innerHTML = courses
    .map(
      (c) =>
        `<a href="#learn/${c.id}" ${c === course ? 'aria-current="page"' : ""}>${text(c.name)}</a>`,
    )
    .join("");
  const n = course.steps.filter((s) =>
    visited.has(lessonKey(course, s)),
  ).length;
  $("rail-title").textContent =
    t("Chapter steps", "本章步骤") + " · " + n + "/" + total;
  $("route-list").innerHTML = `<nav>${course.steps
    .map((s, i) => {
      const done = visited.has(lessonKey(course, s));
      return `<a href="${lessonUrl(course, s)}" ${s === step ? 'aria-current="step"' : ""} class="${done ? "finished" : ""}"><span class="route-number">${done ? "✓" : String(i + 1).padStart(2, "0")}</span>${html(text(s.title))}</a>`;
    })
    .join(
      "",
    )}</nav><a class="capstone" href="${capstone(course)}">${t("Full experiment", "本章综合实验")} · ${text(course.name)}</a>`;
  $("position").textContent =
    text(course.name) +
    " · " +
    t(
      "Step " + (index + 1) + " of " + total,
      "第 " + (index + 1) + " 步，共 " + total + " 步",
    );
  const previous =
    course.steps[index - 1] ||
    courses[courses.indexOf(course) - 1]?.steps.at(-1);
  const priorCourse = index > 0 ? course : courses[courses.indexOf(course) - 1];
  $("back").hidden = !previous;
  if (previous) {
    $("back").href = lessonUrl(priorCourse, previous);
    $("back").textContent =
      t("Previous · ", "上一步 · ") + text(previous.title);
  }
  const ref = lessons.find((l) => l.id === course.id).source;
  $("source").href = ref.url;
  $("source").textContent = ref.label;
  $("bench-link").textContent = t("Full experiment", "综合实验");
  $("bench-link").href = capstone(course);
}
function controls() {
  $("controls").innerHTML = step.controls
    .map((c) => {
      const key = html(c.key),
        v = values[c.key],
        label = html(text(c.label));
      if (c.options)
        return `<fieldset class="control"><legend>${label}</legend><div class="choices">${c.options.map((o, i) => `<button data-key="${key}" data-choice="${i}" aria-pressed="${v === i}">${html(text(o))}</button>`).join("")}</div></fieldset>`;
      return `<div class="control"><label for="knob-${key}">${label}</label><div class="range-line"><button data-bump="-1" data-key="${key}" aria-label="${t("Decrease", "减少")} ${label}">−</button><input id="knob-${key}" data-knob="${key}" type="range" min="${c.min}" max="${c.max}" step="${c.step ?? 1}" value="${v}"><button data-bump="1" data-key="${key}" aria-label="${t("Increase", "增加")} ${label}">+</button><output for="knob-${key}">${v}</output></div></div>`;
    })
    .join("");
}
function results() {
  const now = step.view(values, t),
    before = pinned && step.view(pinned, t);
  const rows = (list, marks = []) =>
    list
      .map((r, i) => {
        const mark =
          marks[i]?.label === r.label ? marks[i].fraction : undefined;
        return `<div class="result-row"><div class="result-head"><span>${html(r.label)}</span><b>${html(r.value)}</b></div>${Number.isFinite(r.fraction) ? `<div class="result-bar" aria-hidden="true"><i style="width:${r.fraction * 100}%"></i>${Number.isFinite(mark) ? `<em style="left:${mark * 100}%"></em>` : ""}</div>` : ""}</div>`;
      })
      .join("");
  $("guided-visual").innerHTML =
    renderGroupVisual(step, values, t) || renderVisual(step, values, t);
  $("readout").innerHTML =
    rows(now.rows, before?.rows) +
    (before
      ? `<div class="compare-view"><h3>${t("Saved comparison", "保存的对照值")}</h3>${rows(before.rows)}</div>`
      : "");
  $("calculation").textContent = now.formula;
  $("pin").textContent = pinned
    ? t("Clear comparison", "清除对照")
    : t("Save for comparison", "保存作对照");
  $("pin").setAttribute("aria-pressed", String(!!pinned));
}
function feedback() {
  $("check").textContent = t("Check this step", "检查这一步");
  $("reset").textContent = t("Reset step", "重置本步");
  const final = step === course.steps.at(-1);
  $("next").hidden = !passed;
  $("next").textContent = final
    ? t("Try the full experiment", "进入本章综合实验")
    : t("Next step", "下一步");
  $("feedback").className = review ? (passed ? "success" : "retry") : "";
  $("feedback").textContent = review
    ? (passed ? t("Correct. ", "通过。") : t("Try again. ", "再试一次。")) +
      step.feedback(values, t)
    : "";
}
function render() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.title = text(step.title) + " · Training Lab";
  $("language").textContent = t("中文", "English");
  $("games").textContent = t("Games", "游戏大厅");
  $("open-map").textContent = t("Course map", "学习路线图");
  $("title").textContent = text(step.title);
  $("brief").textContent = text(step.intro);
  $("task").textContent = text(step.task);
  $("task-label").textContent = t("Try it", "动手做");
  $("results-label").textContent = t("Live calculation", "计算结果");
  nav();
  controls();
  results();
  feedback();
}
function change(k, value) {
  const valid = valuesFor(step, { ...values, [k]: value });
  values[k] = valid[k];
  review = false;
  passed = false;
  for (const btn of $("controls").querySelectorAll("[data-choice]"))
    if (btn.dataset.key === k)
      btn.setAttribute(
        "aria-pressed",
        String(+btn.dataset.choice === valid[k]),
      );
  const input = document.getElementById("knob-" + k);
  if (input) {
    input.value = valid[k];
    input.parentElement.querySelector("output").value = valid[k];
  }
  results();
  feedback();
  save();
}
$("controls").addEventListener("input", (e) => {
  if (e.target.dataset.knob)
    change(e.target.dataset.knob, Number(e.target.value));
});
$("controls").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  const k = b.dataset.key;
  if (b.dataset.choice !== undefined) change(k, Number(b.dataset.choice));
  else if (b.dataset.bump) {
    const c = step.controls.find((c) => c.key === k);
    const value = Math.max(
      c.min,
      Math.min(
        c.max,
        Number((values[k] + Number(b.dataset.bump) * (c.step ?? 1)).toFixed(8)),
      ),
    );
    change(k, value);
  }
});
$("open-map").onclick = () => {
  $("map-title").textContent = t(
    "How the chapters connect",
    "这五章怎么连起来",
  );
  $("close-map").textContent = t("Close", "关闭");
  $("map-caption").textContent = t(
    "In the full experiments, SFT can use your pretrained scores. DPO, REINFORCE and GRPO can each start from that same SFT checkpoint.",
    "在综合实验里，SFT 可以沿用你预训练出的分数。到了 DPO、REINFORCE 和 GRPO，每种方法都可以从同一份 SFT 参数出发。",
  );
  const tile = (c) => {
    const complete = c.steps.filter((s) => visited.has(lessonKey(c, s))).length;
    const next =
      c.steps.find((s) => !visited.has(lessonKey(c, s))) ?? c.steps[0];
    return `<a href="${lessonUrl(c, next)}"><strong>${html(text(c.name))}</strong><small>${complete}/${c.steps.length} ${t("steps complete", "步已通过")}</small></a>`;
  };
  $("map-route").innerHTML =
    tile(courses[0]) +
    '<span class="map-arrow" aria-hidden="true">↓</span>' +
    tile(courses[1]) +
    `<span class="map-branches">${t("Three ways to post-train from SFT", "从 SFT 出发的三种练习")}</span><div class="map-fork">${courses.slice(2).map(tile).join("")}</div>`;
  $("course-map").showModal();
};
$("close-map").onclick = () => $("course-map").close();
$("map-route").onclick = (e) => {
  if (e.target.closest("a")) $("course-map").close();
};
$("pin").onclick = () => {
  pinned = pinned ? null : { ...values };
  results();
  save();
};
$("check").onclick = () => {
  review = true;
  passed = step.check(values);
  if (passed) visited.add(activeKey);
  nav();
  feedback();
  save();
};
$("reset").onclick = () => {
  values = valuesFor(step);
  pinned = null;
  review = false;
  passed = false;
  render();
  save();
};
$("next").onclick = () => {
  if (!passed) return;
  const next = course.steps[course.steps.indexOf(step) + 1];
  if (next) location.hash = lessonUrl(course, next);
  else location.assign(capstone(course));
};
$("language").onclick = () => {
  lang = lang === "en" ? "zh" : "en";
  render();
  save();
};
window.addEventListener("hashchange", () => {
  routing();
  if (narrow.matches) $("rail").open = false;
  $("title").focus({ preventScroll: true });
  window.scrollTo(0, 0);
});
narrow.addEventListener("change", () => {
  $("rail").open = !narrow.matches;
});
routing();
