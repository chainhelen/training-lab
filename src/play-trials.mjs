import { trainStep } from "./trainer.mjs";
import { seedComparison } from "./pretrain-replicates.mjs";

const ab = ["A", "B"];
const stat = (label, value) => ({ label, value: String(value) });
const percent = (x) => (100 * x).toFixed(1) + "%";
const fmt = (x) => x.toFixed(3);
const avg = (xs) => xs.reduce((sum, x) => sum + x, 0) / xs.length;
const loss = (scores, counts) =>
  trainStep(scores, { method: "pretrain", counts, lr: 0 }).loss;
const add = (a, b) => a + b;

// Uniform draws within one archive, without replacement. The two archives
// are sampled independently once the allocation has been chosen.
function missProbability(n, rare, draws) {
  let p = 1;
  for (let k = 0; k < draws; k++) p *= Math.max(0, n - rare - k) / (n - k);
  return p;
}
function coverageProbabilities(s) {
  return s.draws.map((draws, i) => 1 - missProbability(6, s.rare[i], draws));
}
const coverage = {
  id: "coverage-plan",
  title: ["Seven draws, two rare signals", "七次抽样，两个稀有信号"],
  brief: [
    "Each archive holds six records. An archivist counted how many end in CHECK. Reserve seven draws across them before opening any records. Each archive will be sampled uniformly without replacement.",
    "两个库各有六份日志，管理员已经数出各有几份以 CHECK 结尾。开封前先分配七个抽样名额。每个库里等概率抽取，抽过的不会再放回。",
  ],
  target: [
    "Use all seven draws. Reach at least a 52% chance of finding CHECK in both archives.",
    "用完七个名额，让两个库都能抽到 CHECK 的概率达到 52%。",
  ],
  guide: "miss-rare-target",
  start: (seed) => ({ rare: seed % 2 ? [1, 2] : [2, 1], draws: [0, 0] }),
  play(s, id) {
    const i = ab.indexOf(id);
    if (i < 0 || s.draws[i] === 6 || s.draws.reduce(add, 0) === 7) return s;
    const draws = s.draws.slice();
    draws[i]++;
    return { ...s, draws };
  },
  view(s, t) {
    const p = coverageProbabilities(s);
    return {
      stats: [
        stat(t("Draws left", "剩余名额"), 7 - s.draws.reduce(add, 0)),
        stat(t("Both covered", "两个库都抽中"), percent(p[0] * p[1])),
        stat(
          t("Miss in at least one", "至少漏掉一个库"),
          percent(1 - p[0] * p[1]),
        ),
      ],
      cards: ab.map((name, i) => ({
        id: name,
        title: t("Reserve in ", "给这个库加名额 ") + name,
        detail: t("CHECK records: ", "CHECK 记录 ") + s.rare[i] + "/6",
        tag: t("Reserved ", "已分配 ") + s.draws[i],
        disabled: s.draws.reduce(add, 0) === 7 || s.draws[i] >= 6,
      })),
      board: {
        columns: [
          t("Archive", "日志库"),
          t("Draws", "抽几条"),
          t("Miss CHECK", "漏掉 CHECK"),
          t("Find CHECK", "抽到 CHECK"),
        ],
        rows: ab.map((name, i) => [
          name,
          s.draws[i],
          percent(1 - p[i]),
          percent(p[i]),
        ]),
      },
      note:
        t("Chance of hits in both archives = ", "两库都命中的概率 = ") +
        percent(p[0]) +
        " × " +
        percent(p[1]) +
        " ≈ " +
        percent(p[0] * p[1]),
    };
  },
  grade(s, t) {
    const [a, b] = coverageProbabilities(s),
      ok = s.draws.reduce(add, 0) === 7 && a * b >= 0.52;
    return {
      ok,
      message: ok
        ? t(
            "The scarcer archive gets four draws, the other three. Your chance of covering both is ",
            "更稀少的库分四个名额，另一个库分三个。两库都命中的概率是 ",
          ) + percent(a * b)
        : t(
            "Check the miss probabilities in both rows. Reserving a draw in one archive leaves one fewer for the other.",
            "看看两行漏采的概率。往一个库多分一个名额，另一个库就少一次机会。",
          ),
    };
  },
};

const texts = [
  ["The", "signal", "says", "HOLD"],
  ["The", "road", "says", "GO"],
  ["The", "sensor", "says", "CHECK"],
];
function windowOptions(s) {
  return s.log === null
    ? [
        { id: "log-0", start: 0 },
        { id: "log-1", start: 4 },
        { id: "cross", start: 2 },
      ]
    : [
        { id: "next", start: s.log + 1 },
        { id: "same", start: s.log },
        { id: "other", start: s.log === 0 ? 5 : 1 },
      ];
}
function validRow(r) {
  return (r.x === 0 || r.x === 4) && r.y === r.x + 1;
}
const packing = {
  id: "pack-windows",
  title: ["Pack two shifted windows", "拼出两排训练窗口"],
  brief: [
    "Two separate logs contain four tokens each. The loader has two rows with three input positions apiece. For each row, load x and then its three next-token labels y. A row must stay within one log.",
    "有两份独立日志，每份四个 token。加载器能放两行，每行有三个输入位置。先选三个输入 x，再给它配三个目标 y。同一行的预测都要留在同一份日志内。",
  ],
  target: [
    "Pack all six valid next-token targets, three from each log.",
    "填满六个有效目标，两份日志各贡献三个。",
  ],
  guide: "count-targets",
  start(seed) {
    const a = (seed >>> 0) % 3,
      b = (a + 1 + ((seed >>> 0) % 2)) % 3;
    return { words: [...texts[a], ...texts[b]], log: null, rows: [] };
  },
  play(s, id) {
    const choice = windowOptions(s).find((c) => c.id === id);
    if (s.rows.length >= 2 || !choice) return s;
    if (s.log === null) return { ...s, log: choice.start };
    return {
      ...s,
      log: null,
      rows: [...s.rows, { x: s.log, y: choice.start }],
    };
  },
  view(s, t) {
    const slice = (start) => s.words.slice(start, start + 3).join(" ");
    const correct = s.rows.filter(validRow);
    return {
      stats: [
        stat(t("Completed rows", "已拼行数"), s.rows.length + " / 2"),
        stat(
          t("Logs with a valid row", "已有完整有效行的日志数"),
          new Set(correct.map((r) => r.x)).size + " / 2",
        ),
      ],
      cards: windowOptions(s).map((o) => ({
        id: o.id,
        title: (s.log === null ? "x" : "y") + " = " + slice(o.start),
        detail:
          o.start === 2 || o.start === 3
            ? t("Straddles the two logs", "横跨两份日志")
            : t("Log ", "日志 ") + (o.start < 4 ? 1 : 2),
        disabled: s.rows.length >= 2,
      })),
      board: {
        columns: [t("Row", "行"), "x", "y", t("Check", "对齐")],
        rows: [
          ...s.rows.map((r, i) => [
            i + 1,
            slice(r.x),
            slice(r.y),
            validRow(r) ? "✓" : "×",
          ]),
          ...(s.log === null
            ? []
            : [[s.rows.length + 1, slice(s.log), "…", "…"]]),
        ],
      },
      note:
        t("Source logs: ", "原始日志为 ") +
        s.words.slice(0, 4).join(" ") +
        " ｜ " +
        s.words.slice(4).join(" "),
    };
  },
  grade(s, t) {
    const ok = new Set(s.rows.filter(validRow).map((r) => r.x)).size === 2;
    return {
      ok,
      message: ok
        ? t(
            "Six genuine next-token labels, with each log supplying its own row.",
            "六个目标都指向日志里的下一个词，两份日志各出一行。",
          )
        : s.rows.length < 2
          ? t(
              "Finish both x/y rows. The target for x in column one is the following word in that log.",
              "先拼完两行 x/y。第一列的 x 要预测这份日志中紧随它的词。",
            )
          : t(
              "Check the boundaries and shift. Start once at the beginning of each log and move y one token right.",
              "检查边界和错位。每份日志各从开头取一次 x，把 y 的起点往右移一个 token。",
            ),
    };
  },
};

const labels = ["HOLD", "GO", "CHECK"];
const blankArm = () => ({ read: 0, logits: [0, 0, 0], batches: [] });
function batchOption(id) {
  return [
    ["A3", 0, 3],
    ["A6", 0, 6],
    ["B3", 1, 3],
    ["B6", 1, 6],
  ].find(([name]) => name === id);
}
function aligned(s) {
  return (
    s.arms.every((a) => a.read === 12) &&
    s.arms[0].batches.join() === s.arms[1].batches.join()
  );
}
const trials = {
  id: "equalize-trials",
  title: ["Build a fair A/B run", "自己搭一组公平对照"],
  brief: [
    "A has three records to reread. B can draw on all twelve. Both models start at zero with learning rate 0.8. You control the size and timing of their batches. Each arm has a twelve-target cap.",
    "A 只有三份资料可循环读，B 有十二份。两边的初始分数都是零，学习率都是 0.8。你来决定每次哪边更新、读多少。每边最多读十二个目标。",
  ],
  target: [
    "Give both arms twelve reads, with the same batch sizes in the same order. Then compare them on the shared validation set.",
    "让两边各读十二次，每一步的批量大小都逐一对齐，再看两者在同一份验证集上的损失。",
  ],
  guide: "fixed-read-budget",
  start(seed) {
    const shift = (seed >>> 0) % 3;
    return {
      archive: [0, 0, 1, 0, 2, 1, 1, 0, 1, 0, 2, 0].map((i) => (i + shift) % 3),
      holdout: labels.map((_, i) => [3, 2, 1][(i + 3 - shift) % 3]),
      arms: [blankArm(), blankArm()],
    };
  },
  play(s, id) {
    const opt = batchOption(id);
    if (!opt) return s;
    const [, i, n] = opt,
      a = s.arms[i];
    if (a.read + n > 12) return s;
    const counts = [0, 0, 0],
      size = i ? 12 : 3;
    for (let k = 0; k < n; k++) counts[s.archive[(a.read + k) % size]]++;
    const logits = trainStep(a.logits, {
      method: "pretrain",
      counts,
      lr: 0.8,
    }).logits;
    const arms = s.arms.slice();
    arms[i] = { logits, read: a.read + n, batches: [...a.batches, n] };
    return { ...s, arms };
  },
  view(s, t) {
    return {
      stats: ab.map((name, i) =>
        stat(
          name + t(" reads remaining", " 剩余读取次数"),
          12 - s.arms[i].read,
        ),
      ),
      cards: ["A3", "A6", "B3", "B6"].map((id) => {
        const [, i, n] = batchOption(id);
        return {
          id,
          title: ab[i] + t(" · read ", " · 读 ") + n + t(" targets", " 个目标"),
          detail: t("One optimizer update", "参数更新一次"),
          disabled: s.arms[i].read + n > 12,
        };
      }),
      board: {
        columns: [
          t("Arm", "组"),
          t("Distinct records read", "读过的资料数"),
          t("Batch sequence", "每步读几个"),
          t("Validation loss", "验证损失"),
        ],
        rows: s.arms.map((a, i) => [
          ab[i],
          Math.min(i ? 12 : 3, a.read),
          a.batches.join(" → ") || "—",
          fmt(loss(a.logits, s.holdout)),
        ]),
      },
      note:
        t(
          "B reads this archive in order. A cycles through its first three: ",
          "B 按顺序读下面的库，A 只循环其中前三份： ",
        ) + s.archive.map((i) => labels[i]).join(" · "),
    };
  },
  grade(s, t) {
    const ok = aligned(s);
    return {
      ok,
      message: ok
        ? t(
            "Matched exposure and update schedule. The remaining difference is which records the two arms saw.",
            "读取量和更新批次都对齐了。这两轮训练的区别就是读到哪些资料。",
          )
        : t(
            "Compare the two batch sequences. Each arm should spend all twelve reads and the rows should match column by column.",
            "对照两行的批次顺序。每边都要读满十二次，同一步更新用的目标数也要相等。",
          ),
    };
  },
};

function differences(s) {
  return s.opened.map((n) => {
    const a = seedComparison({ size: 2, run: n }).selected.validationLoss;
    const b = seedComparison({ size: 3, run: n }).selected.validationLoss;
    return { n, a, b, delta: a - b };
  });
}
const paired = {
  id: "paired-report",
  title: ["Audit a one-seed headline", "拆穿只报一次的结论"],
  brief: [
    "An early run favors one recipe. Three seed IDs were registered before training. For each ID, compare 12 records with 24 from the same shuffled archive. Each model gets 48 reads and 16 updates, then sits the same 24-item test.",
    "先跑出来的一组支持其中一种方案。试验开始前已经登记了三个种子。每个种子都从同一乱序库里分别取十二和二十四份资料训练，两种模型都读 48 次、更新 16 步，再做同样 24 道题。",
  ],
  target: [
    "Reveal the other two paired runs. File the recipe with lower mean held-out loss across the three registered seeds.",
    "补齐另外两组配对试验，提交三组平均验证损失更低的方案。",
  ],
  guide: "compare-subsets",
  start(seed) {
    const set = seed % 2 ? [6, 4, 10] : [5, 6, 11];
    if ((seed >>> 0) % 4 >= 2) [set[1], set[2]] = [set[2], set[1]];
    return { set, opened: [set[0]], report: null };
  },
  play(s, id) {
    if (id === "report-A" || id === "report-B") {
      const report = id.slice(-1);
      return s.report === report ? s : { ...s, report };
    }
    const n = s.set.find((n) => "run-" + n === id);
    if (!n || s.opened.includes(n)) return s;
    return { ...s, opened: [...s.opened, n], report: null };
  },
  view(s, t) {
    const comps = differences(s),
      delta = avg(comps.map((r) => r.delta));
    return {
      stats: [
        stat(
          t("Registered pairs opened", "已揭晓的登记组数"),
          s.opened.length + " / 3",
        ),
        stat(t("Mean A − B loss", "平均损失差 A − B"), fmt(delta)),
      ],
      cards: [
        ...s.set.map((n) => ({
          id: "run-" + n,
          title: t("Reveal pair ", "揭晓一对 ") + n,
          detail: t("Same seed for A and B", "A 和 B 用同一种子"),
          disabled: s.opened.includes(n),
        })),
        ...ab.map((name, i) => ({
          id: "report-" + name,
          title: t("File recipe ", "提交方案 ") + name,
          detail: (i ? 24 : 12) + t(" records per run", " 份训练资料"),
          tag: s.report === name ? t("Filed", "已提交") : undefined,
          disabled: s.report === name,
        })),
      ],
      board: {
        columns: [t("Seed ID", "种子编号"), "A · 12", "B · 24", "A − B"],
        rows: comps.map((r) => [r.n, fmt(r.a), fmt(r.b), fmt(r.delta)]),
      },
      note: t(
        "Each cell is a measured validation loss. Smaller is better. A negative difference favors A.",
        "表中是每个模型实测的验证损失，越小越好。差值为负代表 A 更低。",
      ),
    };
  },
  grade(s, t) {
    const comps = differences(s);
    if (comps.length < 3)
      return {
        ok: false,
        message: t(
          "Open every registered pair before checking the headline.",
          "目前的结果还没覆盖全部三个登记种子。把尚未揭晓的组补全，再检查结论。",
        ),
      };
    const delta = avg(comps.map((r) => r.delta)),
      best = delta < 0 ? "A" : "B",
      ok = s.report === best;
    return {
      ok,
      message: ok
        ? t(
            "The first seed favored the other recipe. All three together give mean A − B = ",
            "第一组倾向另一个方案。三组合在一起，平均损失差 A − B = ",
          ) +
          fmt(delta) +
          t(
            ". This conclusion applies to these three paired runs.",
            "。这个结论对应这三组配对试验。",
          )
        : t(
            "Compare the sign of the mean difference. Positive A − B favors B; negative favors A.",
            "看平均差的正负。A − B 为正说明 B 的损失更低，为负说明 A 的损失更低。",
          ),
    };
  },
};
export const trialMissions = [coverage, packing, trials, paired];
