// Each record contributes one scored token after the same context. The event
// ID is metadata for tracking exposure. It never enters the three-logit model.
import { trainStep } from "./trainer.mjs";

const tokens = ["HOLD", "GO", "CHECK"];
const logs = [0, 0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2];
const ids = logs.map((_, i) => i + 1);
const samples = [
  [1, 2, 3],
  [4, 5, 6],
  [6, 11, 12],
];
const heldOut = [3, 2, 1];
const paper = {
  label: "Muennighoff et al. · Scaling Data-Constrained Language Models",
  url: "https://papers.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html",
};
const sampling = {
  label: "SciPy · Hypergeometric sampling",
  url: "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html",
};
const experiments = {
  label: "Hugging Face · datablations",
  url: "https://github.com/huggingface/datablations",
};
const row = (label, value) => ({ label, value: String(value) });
const f = (n) => n.toFixed(3);
const percent = (p) => (p * 100).toFixed(1) + "%";
const fraction = (label, p) => ({ label, value: percent(p), fraction: p });
const numeric = (key, label, initial, min, max) => ({
  key,
  label,
  initial,
  min,
  max,
});
const options = (key, label, initial, items) => ({
  key,
  label,
  initial,
  options: items,
});
const caption = (title, html) =>
  `<figure class="lab-picture"><figcaption>${title}</figcaption>${html}</figure>`;
function counts(recordIds) {
  const result = [0, 0, 0];
  for (const id of recordIds) result[logs[id - 1]]++;
  return result;
}
function recordStrip(chosen, t) {
  return caption(
    t(
      "Included records · dashed records are set aside",
      "入选的日志有底色，虚线的未入选",
    ),
    '<div class="pic-strip">' +
      ids
        .map(
          (id) =>
            `<span class="pic-token ${chosen.includes(id) ? "target" : "later"}"><small>#${id}</small> ${tokens[logs[id - 1]]}</span>`,
        )
        .join("") +
      "</div>",
  );
}
function choose(n, k) {
  if (k > n) return 0;
  let result = 1;
  for (let i = 1; i <= Math.min(k, n - k); i++)
    result = (result * (n - i + 1)) / i;
  return Math.round(result);
}
function fit(z, c, lr = 0) {
  return trainStep(z, { method: "pretrain", counts: c, lr });
}
function batchIds(size, step) {
  return [0, 1, 2].map((i) => ((3 * step + i) % size) + 1);
}
function trajectory(size) {
  const trainCounts = counts(ids.slice(0, size));
  let z = [0, 0, 0];
  const checkpoints = [];
  for (let step = 0; step <= 4; step++) {
    checkpoints.push({
      train: fit(z, trainCounts).loss,
      validation: fit(z, heldOut).loss,
      logits: z,
    });
    if (step < 4) z = fit(z, counts(batchIds(size, step)), 1).logits;
  }
  return checkpoints;
}
const trials = Object.fromEntries([3, 6, 12].map((n) => [n, trajectory(n)]));

// Keep both axes fixed as checkpoints appear, so dragging the slider
// never changes the scale of a difference.
function validationChart(updates, t) {
  const x = (i) => 38 + i * 68;
  const y = (loss) => 98 - (loss - 1) * 800;
  const axes = [1, 1.05, 1.1]
    .map(
      (loss) =>
        `<line x1="38" x2="310" y1="${y(loss)}" y2="${y(loss)}"/><text x="34" y="${y(loss) + 3}" text-anchor="end">${loss.toFixed(2)}</text>`,
    )
    .join("");
  const ticks = Array.from(
    { length: 5 },
    (_, i) => `<text x="${x(i)}" y="113" text-anchor="middle">${i}</text>`,
  ).join("");
  const trace = (size, css) => {
    const points = trials[size]
      .slice(0, updates + 1)
      .map((p, i) => [x(i), y(p.validation)]);
    return `<g class="${css}"><polyline points="${points.map((p) => p.join(",")).join(" ")}"/>${points.map(([cx, cy]) => `<circle cx="${cx}" cy="${cy}" r="3"/>`).join("")}</g>`;
  };
  return `<svg class="loss-plot" viewBox="0 0 330 120" aria-hidden="true">${axes}${ticks}${trace(3, "curve-a")}${trace(12, "curve-b")}</svg><div class="loss-key"><span>A · ${t("3 records", "3 条日志")}</span><span>B · ${t("12 records", "12 条日志")}</span><small>${t("Horizontal axis · updates", "横轴 · 更新步数")}</small></div>`;
}
export const scarcitySteps = [
  {
    id: "shrink-corpus",
    title: ["Shrink the available corpus", "把能拿到的日志减到四分之一"],
    intro: [
      "Twelve rover events have different record IDs. From each record we train on just one ending after “The signal says”. The archive contains six HOLD, four GO and two CHECK endings.",
      "仓库里有 12 条不同事件的日志，每条有一个编号。模型只学“The signal says”后面接哪个词，每条贡献一个训练目标。整个仓库共有 6 个 HOLD、4 个 GO、2 个 CHECK。",
    ],
    task: [
      "Keep just the first quarter of the archive. Which ending disappears?",
      "只保留最前面的四分之一，看看哪个结尾从训练集里消失了。",
    ],
    controls: [
      numeric(
        "records",
        ["Records kept from the start", "从开头留下几条"],
        12,
        1,
        12,
      ),
    ],
    source: paper,
    visual: ({ records }, t) => recordStrip(ids.slice(0, records), t),
    view({ records }, t) {
      const c = counts(ids.slice(0, records));
      return {
        rows: [
          row(t("Distinct records kept", "不同日志数"), records),
          ...tokens.map((word, i) => row(word, c[i])),
        ],
        formula: `${c.join(" + ")} = ${records}`,
      };
    },
    check: ({ records }) => records === 3,
    feedback: ({ records }, t) =>
      records === 3
        ? t(
            "The first three end in HOLD, HOLD, GO. CHECK has no example left even though it occurs twice in the archive.",
            "留下的三条是 HOLD、HOLD、GO。仓库里本有两条 CHECK，这份训练集一条都没拿到。",
          )
        : t(
            "A quarter of twelve is three. Inspect the endings of records 1 through 3.",
            "12 条的四分之一是 3 条。看一下编号 1 到 3 的结尾。",
          ),
  },
  {
    id: "same-size-sample",
    title: ["Keep the size, change the records", "同样拿三条，换一批试试"],
    intro: [
      "Keep the training set at three records. These are three possible subsets of the same twelve events. Look up each selected ID in the archive.",
      "固定只取三条。下面三份小样本都来自刚才那 12 条日志，区别只是抽到哪些编号。按编号就能找出各自的结尾。",
    ],
    task: [
      "Find the subset in which two of three endings are CHECK.",
      "选出三条里有两条是 CHECK 的那一组。",
    ],
    controls: [
      options(
        "sample",
        ["Three record IDs", "这次选的三个编号"],
        0,
        samples.map((a) => [a.join(" · "), a.join(" · ")]),
      ),
    ],
    source: sampling,
    visual: ({ sample }, t) => recordStrip(samples[sample], t),
    view({ sample }, t) {
      const c = counts(samples[sample])[2];
      return {
        rows: [
          row(
            t("Drawn endings", "抽到的结尾"),
            samples[sample].map((i) => tokens[logs[i - 1]]).join(" · "),
          ),
          fraction(t("CHECK in these three", "这三条中的 CHECK 占比"), c / 3),
          fraction(
            t("CHECK in the archive", "整个仓库中的 CHECK 占比"),
            2 / 12,
          ),
        ],
        formula: `${c} ÷ 3 = ${percent(c / 3)}`,
      };
    },
    check: ({ sample }) => sample === 2,
    feedback: ({ sample }, t) =>
      sample === 2
        ? t(
            "Records 6 and 12 contain both CHECK events in the archive. Three-record samples can therefore show 0%, 33.3% or 66.7% CHECK.",
            "6 号和 12 号把全库的两个 CHECK 都抽中了。所以同样只抽三条，CHECK 占比可以是 0%、33.3% 或 66.7%。",
          )
        : t(
            "Look for both record 6 and record 12. Each contributes one CHECK.",
            "找同时含有 6 号和 12 号的那组。这两条各贡献一个 CHECK。",
          ),
  },
  {
    id: "miss-rare-target",
    title: [
      "How likely is it to miss CHECK?",
      "随机抽日志，漏掉 CHECK 的机会有多大",
    ],
    intro: [
      "Shuffle all twelve records and keep the first N without replacement. Every subset of size N is equally likely. A subset misses CHECK only when all its records come from the ten other events.",
      "把 12 条日志充分打乱，取前 N 条，不放回。每种 N 条的组合机会相同。想完全漏掉 CHECK，抽到的日志就必须全来自另外那 10 条。",
    ],
    task: [
      "Find the smallest sample size that makes the chance of missing both CHECK records less than 10%.",
      "找到最小的抽样条数，让两个 CHECK 都漏掉的机会降到 10% 以下。",
    ],
    controls: [
      numeric(
        "draws",
        ["Records drawn without replacement", "不放回地抽几条"],
        3,
        1,
        12,
      ),
    ],
    source: sampling,
    visual({ draws }, t) {
      const miss = choose(10, draws),
        all = choose(12, draws);
      return caption(
        t("Count equally likely subsets", "数一数等可能的组合"),
        `<div class="pic-pair"><div class="pic-span alt"><small>${t("No CHECK", "一条 CHECK 都没有")}</small><b>${miss}</b></div><div class="pic-span saved"><small>${t("At least one CHECK", "至少一条 CHECK")}</small><b>${all - miss}</b></div></div>`,
      );
    },
    view({ draws }, t) {
      const miss = choose(10, draws),
        all = choose(12, draws);
      return {
        rows: [
          row(t("All possible subsets", "全部组合数"), all),
          row(t("Subsets with no CHECK", "完全没抽到 CHECK 的组合数"), miss),
          fraction(
            t("Chance of missing CHECK", "漏掉 CHECK 的概率"),
            miss / all,
          ),
        ],
        formula: `${miss} ÷ ${all} = ${percent(miss / all)}`,
      };
    },
    check: ({ draws }) => draws === 8,
    feedback: ({ draws }, t) =>
      draws === 8
        ? t(
            "Seven records miss CHECK 15.2% of the time. With eight the risk is 9.1%. This probability describes the draw before you see its result.",
            "抽 7 条时，漏掉的机会还有 15.2%。抽 8 条就降到 9.1%。这个数描述的是还没抽之前的风险。",
          )
        : draws < 8
          ? t(
              "Increase the draw size. Fewer subsets can avoid both CHECK records.",
              "再多取几条，能刚好避开两个 CHECK 的组合就会变少。",
            )
          : t(
              "Your risk is below 10%. Try one fewer record to find the smallest qualifying size.",
              "现在已经低于 10%。少取一条再看，找出刚好够用的条数。",
            ),
  },
  {
    id: "repeat-the-subset",
    title: [
      "Read the same three records again",
      "手上只有三条，多读几轮会怎样",
    ],
    intro: [
      "Suppose only records 1, 2 and 3 are available. Reading all three once is one epoch. Each visit scores one target token. The next epoch starts back at record 1.",
      "假设手上只有 1、2、3 号三条日志。把这三条都读一次就叫一轮（epoch）。每读一条算一个目标词，读下一轮时又从 1 号开始。",
    ],
    task: [
      "Reach twelve target-token visits. Check how many different records supplied them.",
      "把轮数加到累计读了 12 个目标词，看看它们来自几条不同的日志。",
    ],
    controls: [numeric("passes", ["Complete passes", "完整读几轮"], 1, 1, 4)],
    source: paper,
    visual({ passes }, t) {
      return caption(
        t("Record IDs visited on each pass", "每一轮读到的日志编号"),
        Array.from(
          { length: passes },
          (_, i) =>
            `<div class="pic-timeline"><small>${t("Pass ", "第 ")}${i + 1}${t("", " 轮")}</small> ${[1, 2, 3].map((id) => `<span class="pic-stop loaded">#${id}</span>`).join("")}</div>`,
        ).join(""),
      );
    },
    view({ passes }, t) {
      return {
        rows: [
          row(t("Distinct record IDs", "读过的不同日志"), 3),
          row(t("Scored targets", "一共训练了几个目标词"), 3 * passes),
          row(t("CHECK targets encountered", "训练时见到 CHECK 几次"), 0),
        ],
        formula: `3 × ${passes} = ${3 * passes}`,
      };
    },
    check: ({ passes }) => passes === 4,
    feedback: ({ passes }, t) =>
      passes === 4
        ? t(
            "Twelve target-token visits came from three records. Repeating them gives the optimizer more updates on HOLD and GO. The missing CHECK never becomes a target.",
            "一共训练了 12 个目标词，都来自原来的 3 条。优化器多了几次学习 HOLD 和 GO 的机会，但 CHECK 始终没有当过目标词。",
          )
        : t(
            "Count three visits per pass. Add another pass and watch which IDs reappear.",
            "每轮增加三次。再加一轮，看再次出现的是哪些编号。",
          ),
  },
  {
    id: "fixed-read-budget",
    title: [
      "Compare two runs at the same budget",
      "同样读 12 个目标词，哪一种读法学得更好",
    ],
    intro: [
      "A cycles through 3 records. B reads 12 in order. Both start with zero scores, train on 3 targets per update and use learning rate 1. After each update we freeze the scores and evaluate on the same six other events (3 HOLD, 2 GO, 1 CHECK).",
      "A 循环读前 3 条，B 顺序读 12 条。两边起始分数都是 0，每次用 3 个目标词更新，学习率都是 1。每步结束后都固定参数，在另外六条日志上验一次（3 个 HOLD、2 个 GO、1 个 CHECK）。",
    ],
    task: [
      "Move both runs to the checkpoint after twelve scored targets. Compare their losses on the same six held-out records.",
      "把两边都推进到已经训练了 12 个目标词的那一步，比较它们在同一份验证集上的损失。",
    ],
    controls: [
      numeric(
        "updates",
        ["Optimizer updates in each run", "两边各更新几步"],
        0,
        0,
        4,
      ),
    ],
    source: experiments,
    visual({ updates }, t) {
      const heading = t("Step", "步"),
        a = t("A · 3 records", "A · 3 条"),
        b = t("B · 12 records", "B · 12 条");
      return caption(
        t("Held-out loss after each update", "逐步查看验证损失"),
        validationChart(updates, t) +
          `<table style="width:100%;text-align:center;font-variant-numeric:tabular-nums"><thead><tr><th>${heading}</th><th>${a}</th><th>${b}</th></tr></thead><tbody>${trials[3]
            .slice(0, updates + 1)
            .map(
              (point, i) =>
                `<tr><th scope="row">${i}</th><td>${f(point.validation)}</td><td>${f(trials[12][i].validation)}</td></tr>`,
            )
            .join("")}</tbody></table>`,
      );
    },
    view({ updates }, t) {
      const a = trials[3][updates],
        b = trials[12][updates];
      return {
        rows: [
          row(
            t("Scored targets in each run", "每边已经训练的目标词"),
            updates * 3,
          ),
          row(
            t("A · train / held-out loss", "A · 自己的训练损失 / 验证损失"),
            f(a.train) + " / " + f(a.validation),
          ),
          row(
            t("B · train / held-out loss", "B · 自己的训练损失 / 验证损失"),
            f(b.train) + " / " + f(b.validation),
          ),
        ],
        formula: `${updates} × 3 = ${updates * 3}`,
      };
    },
    check: ({ updates }) => updates === 4,
    feedback: ({ updates }, t) =>
      updates === 4
        ? t(
            "A fits its own three records more closely. On the shared validation set B finishes slightly ahead. At update 2 A was ahead, so the checkpoint matters too.",
            "A 对自己的三条日志学得更熟。最后在共同的验证集上，B 的损失略低。回看第 2 步，那时反而是 A 更低，所以还要对齐比较的时刻。",
          )
        : t(
            "Both runs spend three targets per update. Advance until each has spent twelve.",
            "两边每一步都用三个目标词。继续推进，直到每边都用满十二个。",
          ),
  },
  {
    id: "matching-batches",
    title: [
      "When can rereading give the same update?",
      "什么时候重读也会算出完全相同的更新",
    ],
    intro: [
      "Keep B at twelve records for one pass. Let A reread a smaller file until it too has used twelve targets. The model sees only the ending counts for each three-record batch. The archive’s final six endings repeat its first six, under fresh record IDs.",
      "B 固定从头读完 12 条。A 可以循环读一个较小的文件，也总共读 12 次。这个模型每步只看本批三个结尾的计数。仓库后六条虽然编号不同，结尾顺序恰好和前六条相同。",
    ],
    task: [
      "Use fewer records than B and match its ending counts in every update. Compare the final scores.",
      "让 A 用更少的日志，同时让每一步的结尾计数都跟 B 一样。再比两边最后的分数。",
    ],
    controls: [
      options("subset", ["Records available to A", "A 能拿到的日志"], 0, [
        ["First 3", "前 3 条"],
        ["First 6", "前 6 条"],
      ]),
    ],
    source: paper,
    finding: [
      "Muennighoff and colleagues also held the compute budget fixed. In their language-model experiments, up to four passes over limited data gave losses close to using fresh data. The benefit of further repeats diminished.",
      "Muennighoff 等人也固定了计算预算。在他们的语言模型实验里，有限数据读到四轮以内，损失接近使用新数据的结果。继续增加重复次数，额外收益逐渐减小。",
    ],
    visual({ subset }, t) {
      const size = subset === 0 ? 3 : 6;
      return caption(
        t(
          "Ending counts in HOLD / GO / CHECK order",
          "逐批计数，依次为 HOLD / GO / CHECK",
        ),
        validationChart(updates, t) +
          `<table style="width:100%;text-align:center;font-variant-numeric:tabular-nums"><thead><tr><th>${t("Step", "步")}</th><th>A</th><th>B</th></tr></thead><tbody>${[0, 1, 2, 3].map((i) => `<tr><th scope="row">${i + 1}</th><td>${counts(batchIds(size, i)).join(" / ")}</td><td>${counts(batchIds(12, i)).join(" / ")}</td></tr>`).join("")}</tbody></table>`,
      );
    },
    view({ subset }, t) {
      const size = subset === 0 ? 3 : 6,
        a = trials[size][4],
        b = trials[12][4];
      return {
        rows: [
          row(
            t("A · records × passes", "A · 日志数 × 轮数"),
            `${size} × ${12 / size}`,
          ),
          row(t("B · records × passes", "B · 日志数 × 轮数"), "12 × 1"),
          row(
            t("A · final scores", "A · 最终三个分数"),
            a.logits.map(f).join(" / "),
          ),
          row(
            t("B · final scores", "B · 最终三个分数"),
            b.logits.map(f).join(" / "),
          ),
        ],
        formula:
          t("Held-out loss A / B", "验证损失 A / B") +
          " = " +
          f(a.validation) +
          " / " +
          f(b.validation),
      };
    },
    check: ({ subset }) => subset === 1,
    feedback: ({ subset }, t) =>
      subset === 1
        ? t(
            "Every batch presents the same counts to the same starting scores. The four updates therefore match exactly in this three-score model. In real text, new contexts can supply different gradients even when the same words appear.",
            "两边从同样的分数出发，四个批次的计数又逐一相同，这个小模型的三个分数就会一步不差地更新。真实文本还包含上下文，同样的词出现在不同句子里，梯度也可能不同。",
          )
        : t(
            "Compare the second batch. Three recycled records give CHECK zero targets again. Try the first six records.",
            "看第二批。循环读前三条时，CHECK 又一次是 0。试试把前六条都放进 A 的文件。",
          ),
  },
];
