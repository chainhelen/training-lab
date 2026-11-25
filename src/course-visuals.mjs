// Small diagrams for the guides. Values in these views follow the same controls
// as the calculation alongside them, including a saved checkpoint's round.
const esc = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[c],
  );
const names = ["HOLD", "GO", "CHECK"];

function figure(title, content) {
  return `<figure class="lab-picture"><figcaption>${esc(title)}</figcaption>${content}</figure>`;
}
function key(color, text) {
  return `<span class="pic-key"><i class="${color}" aria-hidden="true"></i>${esc(text)}</span>`;
}
function prefixStrip({ prefix }, t) {
  const words = ["The", "signal", "says", "HOLD"];
  const cells = words
    .map((word, i) => {
      const kind = i < prefix ? "read" : i === prefix ? "target" : "later";
      return `<span class="pic-token ${kind}">${i === prefix ? '<span aria-hidden="true" class="pic-cut"></span>' : ""}${esc(word)}</span>`;
    })
    .join("");
  return figure(
    t("A moving reading boundary", "阅读边界"),
    `<div class="pic-strip">${cells}</div><div class="pic-legend">${key("read", t("Input", "前文"))}${key("target", t("Next target", "下一个目标"))}${key("later", t("Later", "后文"))}</div>`,
  );
}
function logStack({ lines }, t) {
  const endings = ["HOLD", "HOLD", "GO", "CHECK"];
  return figure(
    t("Four log entries", "四行日志"),
    `<div class="pic-stack">${endings.map((word, i) => `<div class="pic-log ${i < lines ? "included" : "excluded"}"><small>${String(i + 1).padStart(2, "0")}</small><span>The signal says</span><b>${word}</b><span class="pic-choice">${esc(i < lines ? t("In batch", "入选") : t("Set aside", "未选"))}</span></div>`).join("")}</div>`,
  );
}
function lossMask({ prompt, reply }, t) {
  const span = (role, body, scored) =>
    `<div class="pic-span ${scored ? "included" : "excluded"}"><small>${esc(role)}</small><b>${esc(body)}</b><span class="pic-choice">${esc(scored ? t("Loss on", "计入损失") : t("Loss off", "屏蔽损失"))} · ${scored ? 1 : 0}</span></div>`;
  return figure(
    t("One training example", "一条训练样本"),
    `<div class="pic-pair">${span(t("User", "用户"), "Stop the rover. Reply with one command.", prompt)}${span(t("Assistant", "助手"), "HOLD", reply)}</div>`,
  );
}
function checkpoints(v, t, step) {
  const { rows } = step.view(v, t);
  const rounds = [0, 1, 2]
    .map(
      (r) =>
        `<span class="pic-stop ${r === v.snapshot ? "loaded" : ""}">${r}</span>`,
    )
    .join("");
  return figure(
    t("Checkpoint history", "参数快照"),
    `<div class="pic-timeline"><small>${esc(t("Round", "训练轮次"))}</small>${rounds}</div><div class="pic-pair"><div class="pic-span saved"><small>${esc(t("Loaded copy", "载入的副本"))} · ${esc(v.snapshot)}</small><b>HOLD ${esc(rows[1].value)}</b></div><div class="pic-span live"><small>${esc(t("Live model", "在线模型"))} · 2</small><b>HOLD ${esc(rows[0].value)}</b></div></div>`,
  );
}
function preferencePair({ chosen, rejected }, t) {
  return figure(
    t("Two replies to the same instruction", "同一条指令的两份回答"),
    `<div class="pic-root">${esc(t("Stop the rover", "让小车停车"))}</div><div class="pic-fork" aria-hidden="true"></div><div class="pic-pair"><div class="pic-span saved"><small>${esc(t("Preferred", "更喜欢"))}</small><b>${esc(names[chosen])}</b></div><div class="pic-span alt"><small>${esc(t("Rejected", "更不喜欢"))}</small><b>${esc(names[rejected])}</b></div></div>`,
  );
}
function referenceOdds({ odds }, t) {
  const one = (label, ratio) =>
    `<div class="pic-odds"><small>${esc(label)}</small><div class="pic-chips" aria-hidden="true">${Array.from({ length: ratio }, () => '<i class="hold"></i>').join("")}<span>/</span><i class="go"></i></div><b>${esc(ratio)} : 1</b></div>`;
  return figure(
    t("HOLD units per GO unit", "每一份 GO 对应几份 HOLD"),
    one(t("Reference", "参考"), 2) +
      one(t("Live", "在线"), odds) +
      `<div class="pic-legend">${key("hold", "HOLD")}${key("go", "GO")}</div>`,
  );
}
function groupDraws(v, t, step) {
  const { rows } = step.view(v, t);
  const replies = rows[2].value.split(" · ");
  const draws = replies.map((reply, i) => `<li><small>#${i + 1}</small><b>${esc(reply)}</b></li>`).join("");
  return figure(
    t("One instruction, several draws", "同一道题，采样多次"),
    `<div class="pic-root">${esc(t("Stop the rover", "让小车停车"))}</div><div class="pic-branch" aria-hidden="true">↓</div><ol class="pic-draws">${draws}</ol>`,
  );
}
export function renderVisual(step, v, t) {
  switch (step.id) {
    case "next-target":
      return prefixStrip(v, t);
    case "count-endings":
      return logStack(v, t);
    case "mask-prompt":
      return lossMask(v, t);
    case "copy-checkpoint":
      return checkpoints(v, t, step);
    case "pair-replies":
      return preferencePair(v, t);
    case "compare-reference":
      return referenceOdds(v, t);
    case "one-prompt-group":
      return groupDraws(v, t, step);
    default:
      return "";
  }
}
