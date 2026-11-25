import { trainStep } from "./trainer.mjs";

const dedup = {
  label: "Lee et al. · Deduplicating Training Data (ACL 2022)",
  url: "https://aclanthology.org/2022.acl-long.577/",
};
const doremi = {
  label: "Xie et al. · DoReMi (NeurIPS 2023)",
  url: "https://arxiv.org/abs/2305.10429",
};

const originals = [
  { prefix: "Red lamp order", target: "HOLD" },
  { prefix: "Low arch order", target: "HOLD" },
  { prefix: "Clear path order", target: "GO" },
  { prefix: "Wheel slip order", target: "CHECK" },
];
const corpus = [
  originals[0],
  originals[1],
  originals[0],
  originals[2],
  originals[3],
];
const recordText = (r) => r.prefix + " " + r.target;
const percent = (p) => (100 * p).toFixed(1) + "%";
const fixed = (x) => x.toFixed(3);

function filterCorpus(rule) {
  const seen = new Set();
  return corpus.map((record) => {
    const key = rule === 1 ? record.target : recordText(record);
    const keep = rule === 0 || !seen.has(key);
    seen.add(key);
    return keep;
  });
}

function logFigure(title, rows) {
  // All labels and log text in these figures come from the fixed lesson bank.
  const cards = rows
    .map(
      ({ n, record, keep, note }) =>
        `<div class="pic-log ${keep ? "included" : "excluded"}"><small>${n}</small><span>${record.prefix}</span><b>${record.target}</b><span class="pic-choice">${note}</span></div>`,
    )
    .join("");
  return `<figure class="lab-picture"><figcaption>${title}</figcaption><div class="pic-stack">${cards}</div></figure>`;
}

function copyUpdate(copies) {
  const counts = [copies + 1, 1, 1];
  return trainStep([0, 0, 0], { method: "pretrain", counts, lr: 1 });
}

const validation = [
  originals[0],
  { prefix: "Red light order", target: "HOLD" },
  originals[2],
  { prefix: "Open road order", target: "GO" },
];
const fresh = [
  { prefix: "Barrier down order", target: "HOLD" },
  { prefix: "Bridge open order", target: "GO" },
];
const learned = new Map(
  originals.map(({ prefix, target }) => [prefix, target]),
);
const trainLines = new Set(originals.map(recordText));

function validationRun(replaced) {
  const records = [...validation];
  if (replaced >= 1) records[0] = fresh[0];
  if (replaced >= 2) records[2] = fresh[1];
  const losses = records.map(({ prefix, target }) => {
    const cached = learned.get(prefix);
    const probability =
      cached === undefined ? 1 / 3 : cached === target ? 0.8 : 0.1;
    return -Math.log(probability);
  });
  return {
    records,
    losses,
    mean: losses.reduce((a, b) => a + b, 0) / losses.length,
  };
}

const budget = 8;
function domainUpdate(roads) {
  const counts = [budget - roads, roads, 0];
  return trainStep([0, 0, 0], { method: "pretrain", counts, lr: 1 });
}
const mixedModel = domainUpdate(6);
const roadLoss = -Math.log(mixedModel.after[1]);
const siteLoss = -Math.log(mixedModel.after[0]);
function report(siteCount) {
  return (siteCount * siteLoss + (budget - siteCount) * roadLoss) / budget;
}

export const qualitySteps = [
  {
    id: "exact-copies",
    title: ["Which log is a copy?", "哪一行是原样复制的？"],
    intro: [
      "Five logs arrived. Two different situations can both end in HOLD. Compare the whole line before deciding which record to keep.",
      "收到了五行日志。红灯要停车，遇到低矮拱门也要停车，两行都以 HOLD 结尾。先比较整行文字，再决定哪些行该保留。",
    ],
    task: [
      "Keep the four different lines and remove the extra copy of “Red lamp order HOLD”.",
      "保留四行不同的文字，把重复的那行 Red lamp order HOLD 去掉。",
    ],
    controls: [
      {
        key: "rule",
        label: ["Keep one copy of…", "保留哪种记录的第一份？"],
        initial: 0,
        options: [
          ["Keep every row", "每行都保留"],
          ["Each ending", "相同结尾留一份"],
          ["Each full line", "整行相同留一份"],
        ],
      },
    ],
    visual({ rule }, t) {
      const kept = filterCorpus(rule);
      return logFigure(
        t("Incoming log", "收到的日志"),
        corpus.map((record, i) => ({
          n: i + 1,
          record,
          keep: kept[i],
          note: kept[i] ? t("keep", "保留") : t("skip", "略过"),
        })),
      );
    },
    view({ rule }, t) {
      const keep = filterCorpus(rule);
      const kept = corpus.filter((_, i) => keep[i]);
      return {
        rows: [
          { label: t("Rows kept", "保留的行数"), value: kept.length },
          {
            label: t("Different full lines kept", "留下多少种整行文字"),
            value: new Set(kept.map(recordText)).size,
          },
          {
            label: t("Different HOLD lines kept", "留下多少种 HOLD 日志"),
            value: new Set(
              kept.filter((r) => r.target === "HOLD").map(recordText),
            ).size,
          },
        ],
        formula:
          t("5 received", "收到 5 行") +
          " − " +
          (5 - kept.length) +
          " = " +
          kept.length,
      };
    },
    check: ({ rule }) => rule === 2,
    feedback({ rule }, t) {
      if (rule === 2)
        return t(
          "Rows 1 and 3 have identical text. Keeping one of them still leaves both the red-lamp and low-arch examples.",
          "第 1、3 行逐字相同，保留第 1 行就够了。红灯和低拱门这两种停车场景都留下了。",
        );
      return rule === 1
        ? t(
            "Look at row 2. It shares a target with row 1, but its prefix describes another situation.",
            "看第 2 行。它的答案也是 HOLD，但前文描述的是另一种情况。",
          )
        : t(
            "Read rows 1 and 3 side by side. Try the rule that compares every word.",
            "把第 1、3 行从头读到尾，试试比较整行文字的规则。",
          );
    },
    source: dedup,
    finding: [
      "Lee and colleagues found repeated passages across large training corpora and measured less verbatim memorization after deduplication.",
      "Lee 等人在大型语料中找到了成段重复的文字，并在去重训练后测得逐字复现的情况减少。",
    ],
  },
  {
    id: "copied-weight",
    title: ["Give a copy another vote", "多复制一行，哪个词多算一票？"],
    intro: [
      "Train the three-score model to predict the word after “order”. Start with the four distinct logs. Keep one low-arch HOLD, one GO, and one CHECK. Only the number of red-lamp HOLD logs changes.",
      "现在只用末尾的 order 预测它后面的命令，模型仍有三个分数。低拱门的 HOLD、GO、CHECK 各保留一行，只改红灯 HOLD 的份数。",
    ],
    task: [
      "Make HOLD account for exactly two thirds of the target tokens. Look at its probability after one update from equal scores, at learning rate 1.",
      "让所有目标词里恰好有三分之二是 HOLD，再看模型从三个相同分数出发、按学习率 1 更新一次后的概率。",
    ],
    controls: [
      {
        key: "copies",
        label: [
          "Red-lamp rows, including the original",
          "红灯日志的总份数，含原件",
        ],
        min: 1,
        max: 5,
        initial: 1,
      },
    ],
    view({ copies }, t) {
      const n = copies + 3,
        holds = copies + 1;
      const after = copyUpdate(copies).after[0];
      return {
        rows: [
          {
            label: t("HOLD / total targets", "HOLD 占目标词的比例"),
            value: holds + " / " + n + " = " + percent(holds / n),
            fraction: holds / n,
          },
          { label: t("Different full lines", "不同的整行文字"), value: 4 },
          {
            label: t(
              "Model P(HOLD) after one update",
              "更新一次后的模型 P(HOLD)",
            ),
            value: percent(after),
            fraction: after,
          },
        ],
        formula:
          t("HOLD update", "HOLD 分数更新量") +
          " = " +
          holds +
          "/" +
          n +
          " − 1/3 = " +
          fixed(holds / n - 1 / 3),
      };
    },
    check: ({ copies }) => copies === 3,
    feedback({ copies }, t) {
      return copies === 3
        ? t(
            "Three red-lamp rows plus one low-arch row give four HOLD targets out of six. Each copy contributes to the average loss.",
            "三份红灯加一份低拱门，六个目标里有四个 HOLD。算整批平均损失时，每份副本都会算一次。",
          )
        : t(
            `You have ${copies + 1} HOLD targets among ${copies + 3} rows. Count the low-arch HOLD too.`,
            `当前 ${copies + 3} 行里有 ${copies + 1} 个 HOLD，低拱门那一行也算一个。`,
          );
    },
    source: dedup,
  },
  {
    id: "cross-split-overlap",
    title: [
      "Has this validation line appeared in training?",
      "这条验证日志在训练时见过吗？",
    ],
    intro: [
      "Training used the four distinct logs. Validation contains two HOLD lines and two GO lines. A shared answer word is normal. An exact copy of a training line lets a model see the example before its test.",
      "训练用过前面四行不同的日志。验证有两行 HOLD、两行 GO。答案是同一个词很常见；如果前文和答案都原样出现过，验证时就会考到原题。",
    ],
    task: [
      "Find the validation GO line whose full text also appears in training.",
      "找出整行文字曾经进入训练集的那条 GO 验证日志。",
    ],
    controls: [
      {
        key: "line",
        label: ["Inspect validation row", "检查第几条验证日志"],
        min: 1,
        max: 4,
        initial: 4,
      },
    ],
    visual({ line }, t) {
      const selected = validation[line - 1];
      return logFigure(
        t(
          "Compare the validation row with training",
          "把这行验证文字和训练集逐行对照",
        ),
        originals.map((record, i) => {
          const same = recordText(record) === recordText(selected);
          return {
            n: i + 1,
            record,
            keep: same,
            note: same ? t("match", "相同") : t("different", "不同"),
          };
        }),
      );
    },
    view({ line }, t) {
      const r = validation[line - 1],
        identical = trainLines.has(recordText(r));
      return {
        rows: [
          { label: t("Validation row", "所选验证日志"), value: recordText(r) },
          {
            label: t(
              "Any training row with this target?",
              "训练集出现过这个结尾词？",
            ),
            value: t("Yes", "有"),
          },
          {
            label: t("Entire line in training?", "训练集出现过整行？"),
            value: identical ? t("Yes", "有") : t("No", "没有"),
          },
        ],
        formula:
          t(
            "Validation rows with exact training matches",
            "与训练集逐字相同的验证行",
          ) +
          " = " +
          validation.filter((x) => trainLines.has(recordText(x))).length +
          " / 4",
      };
    },
    check: ({ line }) => line === 3,
    feedback({ line }, t) {
      return line === 3
        ? t(
            "“Clear path order GO” occurs in both splits. Row 4 also ends in GO, but its prefix is new.",
            "Clear path order GO 两边都有。第 4 行同样以 GO 结尾，但前文没出现过。",
          )
        : t(
            "Look for GO with the same words before it. The training record begins with “Clear path”.",
            "先找 GO 的两行，再比较前文。训练集那条 GO 日志以 Clear path 开头。",
          );
    },
    source: dedup,
  },
  {
    id: "clean-validation",
    title: ["Retest on unseen lines", "把原题换掉，再测一次"],
    intro: [
      "Freeze a simple lookup predictor. For a training prefix it assigns 80% to the recorded ending and 10% to each alternative. For a new prefix it gives each ending ⅓. Replace copied validation lines with fresh lines that have the same endings.",
      "用一张查表模型做对照，并固定它的参数。见过的前文给原结尾 80%、其余两个词各 10%；新前文给三个词各 ⅓。把验证集里的原题换成新前文，每题的结尾词保持原样。",
    ],
    task: [
      "Replace both copied lines. What is the four-line mean loss when all four prefixes are new?",
      "换掉两条原题。当前文全部换成没见过的内容，四行验证损失的平均值是多少？",
    ],
    controls: [
      {
        key: "replaced",
        label: ["Copied lines replaced", "换掉几条原题"],
        min: 0,
        max: 2,
        initial: 0,
      },
    ],
    visual({ replaced }, t) {
      return logFigure(
        t("Validation records", "本次验证的四行日志"),
        validationRun(replaced).records.map((record, i) => {
          const copied = trainLines.has(recordText(record));
          return {
            n: i + 1,
            record,
            keep: !copied,
            note: copied ? t("seen", "原题") : t("new", "新题"),
          };
        }),
      );
    },
    view({ replaced }, t) {
      const { records, losses, mean } = validationRun(replaced);
      return {
        rows: [
          {
            label: t(
              "Copied lines still in validation",
              "验证集中还剩几条训练原题",
            ),
            value: records.filter((r) => trainLines.has(recordText(r))).length,
          },
          {
            label: t("Targets in every version", "每种方案的目标词"),
            value: "HOLD × 2, GO × 2",
          },
          {
            label: t("Mean validation loss", "平均验证损失"),
            value: fixed(mean),
          },
        ],
        formula: "(" + losses.map(fixed).join(" + ") + ") / 4 ≈ " + fixed(mean),
      };
    },
    check: ({ replaced }) => replaced === 2,
    feedback({ replaced }, t) {
      return replaced === 2
        ? t(
            "Every fresh prefix costs −ln(⅓) ≈ 1.099. The model stayed frozen. Its lower score on the mixed test came from the two familiar prefixes.",
            "四个新前文的损失都是 −ln(⅓) ≈ 1.099。模型一直没动，原先平均损失更低，是因为验证集里有两道原题。",
          )
        : t(
            "Every copy still receives 80% on the correct ending. Replace the remaining copies to measure this predictor on fresh text.",
            "余下的原题仍在正确答案上拿到 80% 的概率。换完这些原题，就能看到查表模型面对新前文的表现。",
          );
    },
    source: dedup,
    finding: [
      "Lee et al. found overlap between training and evaluation data. The lookup probabilities here are chosen for the exercise.",
      "Lee 等人在真实语料中发现训练与评估文本有重叠。本关的查表概率是为了演算而设定的。",
    ],
  },
  {
    id: "domain-budget",
    title: [
      "Divide eight targets between two domains",
      "把八个训练目标分给两类日志",
    ],
    intro: [
      "Return to the model that sees just “order”. In this batch, road logs end in GO and worksite logs end in HOLD. Each line contributes one target token. Start from three zero scores and take one update on eight targets, at learning rate 1.",
      "回到只看 order 的模型。这批道路日志以 GO 结尾，工地日志以 HOLD 结尾，每行贡献一个目标词。三个分数都从 0 出发，用八个目标做一次更新，学习率固定为 1。",
    ],
    task: [
      "Give three quarters of the eight target tokens to road logs and read the resulting P(GO).",
      "把八个训练目标的四分之三分给道路日志，看更新后的 P(GO)。",
    ],
    controls: [
      {
        key: "roads",
        label: ["Road targets out of eight", "八个目标里分几个给道路"],
        min: 0,
        max: 8,
        initial: 4,
      },
    ],
    visual({ roads }, t) {
      const tokens = Array.from({ length: budget }, (_, i) =>
        i < roads
          ? '<span class="pic-token read">GO</span>'
          : '<span class="pic-token target">HOLD</span>',
      );
      return `<figure class="lab-picture"><figcaption>${t("One batch of eight targets", "这一批的八个目标词")}</figcaption><div class="pic-strip">${tokens.join("")}</div></figure>`;
    },
    view({ roads }, t) {
      const after = domainUpdate(roads).after;
      return {
        rows: [
          {
            label: t("Road / worksite targets", "道路 / 工地目标数"),
            value: roads + " / " + (budget - roads),
          },
          { label: "P(GO)", value: percent(after[1]), fraction: after[1] },
          { label: "P(HOLD)", value: percent(after[0]), fraction: after[0] },
        ],
        formula:
          t("GO score update", "GO 分数更新量") +
          " = " +
          roads +
          "/8 − 1/3 ≈ " +
          fixed(roads / 8 - 1 / 3),
      };
    },
    check: ({ roads }) => roads === 6,
    feedback({ roads }, t) {
      return roads === 6
        ? t(
            "Six GO targets and two HOLD targets give P(GO) ≈ 48.1% after one update. Keep this model for the evaluation.",
            "六个 GO、两个 HOLD，更新一次后 P(GO) 约为 48.1%。下一步就用这份参数做评估。",
          )
        : t(
            "The budget is fixed. Each road target takes one place that could hold a worksite target.",
            "一共有八个位置。道路日志多拿一个位置，工地日志就少拿一个。",
          );
    },
    source: doremi,
    finding: [
      "DoReMi learned training-domain weights with a smaller proxy model. Here you can set the two weights by hand.",
      "DoReMi 用较小的代理模型学习各数据来源的训练配比。在这个两类日志的例子里，你可以亲自分配。",
    ],
  },
  {
    id: "domain-report",
    title: ["What can one average hide?", "一个平均数能遮住什么？"],
    intro: [
      "Freeze the model trained on six road and two worksite targets. Test it on eight fresh logs that all have the same final prefix “order”. It assigns 48.1% to GO and 29.2% to HOLD. Change only the mix in the test.",
      "固定上一步用六条道路、两条工地日志训练出的模型。另取八条新日志，最后一词同样是 order。模型给 GO 48.1%、给 HOLD 29.2%。现在只改测试集里两类日志各占几条。",
    ],
    task: [
      "Bring the overall mean loss below 0.80 while keeping at least one worksite example. Watch the worksite loss as you change the mix.",
      "至少留一条工地日志，把总体平均损失调到 0.80 以下。调节时看一眼工地这一项的损失。",
    ],
    controls: [
      {
        key: "sites",
        label: [
          "Worksite examples in the eight-log test",
          "八条测试日志里放几条工地日志",
        ],
        min: 1,
        max: 7,
        initial: 4,
      },
    ],
    view({ sites }, t) {
      return {
        rows: [
          {
            label: t("Road loss per target · GO", "道路每个目标的损失 · GO"),
            value: fixed(roadLoss),
          },
          {
            label: t(
              "Worksite loss per target · HOLD",
              "工地每个目标的损失 · HOLD",
            ),
            value: fixed(siteLoss),
          },
          {
            label: t("Mean across all eight targets", "八个目标的总平均损失"),
            value: fixed(report(sites)),
          },
        ],
        formula:
          "(" +
          (budget - sites) +
          " × " +
          fixed(roadLoss) +
          " + " +
          sites +
          " × " +
          fixed(siteLoss) +
          ") / 8 ≈ " +
          fixed(report(sites)),
      };
    },
    check: ({ sites }) => report(sites) < 0.8,
    feedback({ sites }, t) {
      return report(sites) < 0.8
        ? t(
            "Seven road logs and one worksite log yield 0.794 overall. The worksite loss stays 1.232. A test with fewer worksite logs gives that weakness less weight in the headline number.",
            "七条道路、一条工地，总平均降到 0.794。工地损失仍是 1.232。工地样本越少，这项弱点在总分里的分量就越小。",
          )
        : t(
            "Road logs cost less per target on this model. Include more road logs and check both domain scores again.",
            "这个模型在道路日志上每题损失更低。多放几条道路日志，再看两类各自的损失变没变。",
          );
    },
    source: doremi,
  },
];
