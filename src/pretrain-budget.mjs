// Count target positions in full windows on one device, as in nanoGPT's loader.
const loader = {
  label: "nanoGPT · get_batch",
  url: "https://github.com/karpathy/nanoGPT/blob/master/train.py#L109-L124",
};
const counter = {
  label: "nanoGPT · tokens_per_iter",
  url: "https://github.com/karpathy/nanoGPT/blob/master/train.py#L90-L98",
};
const words = ["The", "rover", "can", "say", "HOLD"];
const range = (key, label, max) => ({ key, label, min: 1, max, initial: 1 });
const row = (label, value) => ({ label, value: String(value) });
const strip = (text, style) =>
  text
    .map((word) => `<span class="pic-token ${style}">${word}</span>`)
    .join("");
const dot = '<i aria-hidden="true"></i>';
function grid(rows, cols) {
  return (
    '<div class="budget-grid" aria-hidden="true">' +
    Array.from(
      { length: rows },
      () => "<div>" + dot.repeat(cols) + "</div>",
    ).join("") +
    "</div>"
  );
}

export const budgetSteps = [
  {
    id: "count-targets",
    title: ["How many targets fit in one line?", "一行日志能出几道题？"],
    intro: [
      "Read “The rover can say HOLD”, one word per token. The first position sees The and predicts rover. The second sees The rover and predicts can. Each visible position has a next word to predict.",
      "这行日志是 The rover can say HOLD，每个词算一个 token。第一个位置看到 The，要猜 rover；第二个位置能看到 The rover，要猜 can。每个位置各出一道题。",
    ],
    task: [
      "Extend the input until all four next words contribute to the loss.",
      "拉长输入，让这一行的四个目标词都参与计算损失。",
    ],
    controls: [range("length", ["Input positions", "输入位置数"], 4)],
    view: ({ length }, t) => ({
      rows: [
        row(t("Input positions", "输入位置数"), length),
        row(t("Last target", "最后一个目标词"), words[length]),
      ],
      formula: `x = [${words.slice(0, length).join(", ")}]\ny = [${words.slice(1, length + 1).join(", ")}]`,
    }),
    visual: ({ length }, t) =>
      `<figure class="lab-picture"><figcaption>${t("Read each column from input to next word", "每一列从输入词对到下一个词")}</figcaption><div class="shift-grid" style="--columns:${length}"><small>x</small>${strip(words.slice(0, length), "read")}<small>y</small>${strip(words.slice(1, length + 1), "target")}</div></figure>`,
    check: (v) => v.length === 4,
    feedback: (v, t) =>
      v.length === 4
        ? t(
            "Five source tokens supplied four next-token targets. HOLD is the last target in this window.",
            "五个原文 token 出了四道题，HOLD 是这个窗口的最后一个目标。",
          )
        : t(
            "The last input can still move to the right until say predicts HOLD.",
            "最后一个输入还能往右移，直到 say 的目标是 HOLD。",
          ),
    source: loader,
  },
  {
    id: "batch-tokens",
    title: ["Count targets in a batch", "一个 batch 到底有几个目标？"],
    intro: [
      "Keep that window length at four positions. A batch stacks several windows before the update. Each window here is full and has four targets.",
      "沿用刚才的四个位置。把几个窗口叠成一批，再更新参数。这里每一行都填满，每行都提供四个目标。",
    ],
    task: [
      "Make one batch that scores twelve target positions.",
      "凑一批数据，让它恰好计算十二个目标位置的损失。",
    ],
    controls: [range("batch", ["Windows in this batch", "这批有几个窗口"], 4)],
    view: ({ batch }, t) => ({
      rows: [
        row(t("Targets per window", "每个窗口的目标数"), 4),
        row(t("Target positions in the batch", "这批的目标位置数"), 4 * batch),
      ],
      formula: `[B, T] = [${batch}, 4]\n${batch} × 4 = ${4 * batch}`,
    }),
    visual: ({ batch }, t) =>
      `<figure class="lab-picture"><figcaption>${t("One row per window · one dot per target", "每行一个窗口，每个点是一个目标位置")}</figcaption>${grid(batch, 4)}</figure>`,
    check: (v) => v.batch === 3,
    feedback: (v, t) =>
      v.batch === 3
        ? t(
            "Three windows give twelve loss terms. Repeated words still count at each position.",
            "三个窗口共十二项损失。即使同一个词出现两次，两个位置也各算一次。",
          )
        : t(
            "Each added row contributes four targets. Count the rows needed for twelve.",
            "每加一行多四个目标。数数凑到十二要几行。",
          ),
    source: counter,
  },
  {
    id: "update-budget",
    title: ["How many updates can 96 targets buy?", "96 个目标够更新几次？"],
    intro: [
      "One device reads twelve target positions per batch. It can collect gradients from several such batches before changing the weights. The run stops after reading 96 positions.",
      "一台设备每批读十二个目标位置。它可以先攒几批梯度，再改一次权重。这次训练一共只读 96 个目标位置。",
    ],
    task: [
      "Keep the 96-position budget and arrange exactly four weight updates.",
      "保持 96 个位置的预算，安排恰好四次参数更新。",
    ],
    controls: [
      {
        key: "accum",
        label: ["Batches before each update", "每次更新前攒几批"],
        initial: 0,
        options: [
          ["1", "1"],
          ["2", "2"],
          ["4", "4"],
          ["8", "8"],
        ],
      },
    ],
    view: ({ accum }, t) => {
      const n = 2 ** accum;
      return {
        rows: [
          row(t("Batches read in total", "总共读几批"), 8),
          row(t("Targets per weight update", "一次更新用几个目标"), 12 * n),
          row(t("Weight updates", "参数更新次数"), 8 / n),
        ],
        formula: `96 ÷ (12 × ${n}) = ${8 / n}`,
      };
    },
    visual: ({ accum }, t) =>
      `<figure class="lab-picture"><figcaption>${t("Each box ends in one update · each dot is a batch", "每框结束时更新一次，每个点是一批")}</figcaption><div class="accum-strip">${Array.from({ length: 8 / 2 ** accum }, () => grid(1, 2 ** accum)).join("")}</div></figure>`,
    check: (v) => v.accum === 1,
    feedback: (v, t) =>
      v.accum === 1
        ? t(
            "Two batches give 24 target positions per update. Four updates spend the entire budget.",
            "每两批共 24 个目标，四次更新正好用完预算。",
          )
        : t(
            "There are eight batches. Divide them into four equal groups.",
            "总共八批，把它们分成一样大的四组。",
          ),
    source: counter,
  },
];
