import { trainStep } from "./trainer.mjs";

const sizes = [3, 6, 12, 24];
const batchSize = 3;
const reads = 48;
const runs = 12;
// A record contributes its final command after the same visible prefix.
// The held-out records have different IDs but the same overall mix of targets.
const archive = Array.from({ length: 48 }, (_, i) =>
  i < 24 ? 0 : i < 42 ? 1 : 2,
);
const validationCounts = [12, 9, 3];
function shuffledArchive(seed) {
  const records = [...archive];
  let state = seed;
  for (let i = records.length - 1; i > 0; i--) {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    const j = Math.floor((state / 2 ** 32) * (i + 1));
    [records[i], records[j]] = [records[j], records[i]];
  }
  return records;
}
function counts(tokens) {
  const counts = [0, 0, 0];
  for (const i of tokens) counts[i]++;
  return counts;
}
function loss(logits, target) {
  return trainStep(logits, { method: "pretrain", counts: target, lr: 0 }).loss;
}
function trainCorpus(corpus) {
  let logits = [0, 0, 0];
  for (let start = 0; start < reads; start += batchSize) {
    const batch = Array.from(
      { length: batchSize },
      (_, j) => corpus[(start + j) % corpus.length],
    );
    logits = trainStep(logits, {
      method: "pretrain",
      counts: counts(batch),
      lr: 0.5,
    }).logits;
  }
  const trainingCounts = counts(corpus);
  return {
    trainingCounts,
    trainLoss: loss(logits, trainingCounts),
    validationLoss: loss(logits, validationCounts),
  };
}
// Matching run numbers reuse the same shuffled archive at every size.
const shuffles = Array.from({ length: runs }, (_, i) =>
  shuffledArchive(5491 + i * 1907),
);
const experiments = sizes.map((size) =>
  shuffles.map((order) => trainCorpus(order.slice(0, size))),
);
export function seedComparison({ size, run }) {
  const results = experiments[size];
  const sorted = results.map((r) => r.validationLoss).sort((a, b) => a - b);
  return {
    selected: results[run - 1],
    results,
    misses: results.filter((r) => r.trainingCounts[2] === 0).length,
    median: (sorted[5] + sorted[6]) / 2,
    low: sorted[0],
    high: sorted.at(-1),
  };
}
const fixed = (x) => x.toFixed(3);
const check = (v) =>
  sizes[v.size] === 3 && seedComparison(v).selected.trainingCounts[2] === 0;

export const replicateStep = {
  id: "compare-subsets",
  title: ["Try twelve different training sets", "换十二批训练资料再试一次"],
  intro: [
    "Pick a subset of a 48-record archive. Each record supplies one target command. Train on the subset in a loop for 16 updates of three targets each. Repeat with twelve different shuffles and evaluate every model on the same 24 held-out records.",
    "库里有 48 份日志，每份只取最后一个命令当目标。从库里抽一批，循环读，每三次预测更新一次，一共更新 16 次。换十二种抽法各训练一个模型，最后用同样的 24 份留出日志给它们打分。",
  ],
  task: [
    "Save the 24-record result for comparison. Switch to three records and inspect a run that never saw CHECK. Compare its training and held-out loss.",
    "可以先保存 24 份时的结果作对照。再缩到 3 份，选中一次完全没见过 CHECK 的训练，对照它在训练集和留出集上的损失。",
  ],
  controls: [
    {
      key: "size",
      label: ["Distinct training records per run", "每次用几份不同的训练日志"],
      initial: 3,
      options: sizes.map((n) => [String(n), String(n)]),
    },
    {
      key: "run",
      label: ["Inspect run number", "查看第几次训练"],
      min: 1,
      max: runs,
      initial: 1,
    },
  ],
  visual: (v, t) => {
    const { results } = seedComparison(v);
    return `<figure class="lab-picture"><figcaption>${t("Twelve training sets · CHECK examples in each", "十二批资料，每批抽到了几份 CHECK")}</figcaption><div class="seed-tiles">${results.map((r, i) => `<div class="${v.run === i + 1 ? "chosen " : ""}${r.trainingCounts[2] === 0 ? "missing" : ""}"><small>#${i + 1}</small><b>${r.trainingCounts[2]}</b></div>`).join("")}</div></figure>`;
  },
  view: (v, t) => {
    const { selected, misses, median, low, high } = seedComparison(v);
    return {
      rows: [
        {
          label: t("Runs that missed CHECK", "没抽到 CHECK 的训练"),
          value: misses + " / " + runs,
        },
        {
          label: t(
            "Selected training set · HOLD / GO / CHECK",
            "选中这批的 HOLD / GO / CHECK 数量",
          ),
          value: selected.trainingCounts.join(" / "),
        },
        {
          label: t(
            "Selected model · training / held-out loss",
            "选中模型的训练损失 / 留出损失",
          ),
          value: [selected.trainLoss, selected.validationLoss]
            .map(fixed)
            .join(" / "),
        },
        {
          label: t(
            "Held-out loss across runs · min / median / max",
            "十二次留出损失 · 最小 / 中位 / 最大",
          ),
          value: [low, median, high].map(fixed).join(" / "),
        },
      ],
      formula:
        t("Target reads", "读过的目标位置") +
        " = " +
        sizes[v.size] +
        " × " +
        reads / sizes[v.size] +
        " = 48\n" +
        t("Held-out targets", "留出集的目标") +
        " = 12 HOLD + 9 GO + 3 CHECK",
    };
  },
  check,
  feedback: (v, t) =>
    check(v)
      ? t(
          "This run practiced the same three records 16 times. CHECK was missing each time. Compare the range across the twelve runs, then try a larger subset with the same 48 reads.",
          "这批三份资料读了十六遍，每一遍都缺 CHECK。看看十二次的留出损失范围，再把资料加多，总读取量仍是 48。",
        )
      : t(
          "Use three records. In the grid, find a run with a zero, then select its number.",
          "选三份资料，在图里找 CHECK 数量为 0 的那一格，切换到它的编号。",
        ),
  source: {
    label: "Muennighoff et al. · Scaling Data-Constrained Language Models",
    url: "https://arxiv.org/abs/2305.16264",
  },
  finding: [
    "The study varied distinct data and repeated passes at fixed compute. The twelve scores above come from this three-command model. Each run uses the same initial scores and learning rate of 0.5.",
    "论文在固定算力下比较了不同的语料量和重复轮数。上面的十二组损失由本页的三命令模型算出，每次的参数起点相同，学习率都为 0.5。",
  ],
};
