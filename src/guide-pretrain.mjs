// Each micro-lab fixes one prefix and changes one input to the calculation.
import { probabilities, trainStep } from "./trainer.mjs";

const commands = ["HOLD", "GO", "CHECK"];
const logs = ["HOLD", "HOLD", "GO", "CHECK"];
const trainingCounts = [2, 1, 0];
const zeroLogits = [0, 0, 0];
const prefixTokens = ["The", "signal", "says", "HOLD"];
const percent = (p) => (100 * p).toFixed(1) + "%";
const number = (x) => x.toFixed(3);
const sign = (x) =>
  Math.abs(x) < 1e-10 ? "0.000" : (x > 0 ? "+" : "") + number(x);
const labels = (items) => items.map((item) => [item, item]);

function probabilityRows(values) {
  return commands.map((label, i) => ({
    label,
    value: percent(values[i]),
    fraction: values[i],
  }));
}

function readCounts(lines) {
  const batch = logs.slice(0, lines);
  return commands.map((token) => batch.filter((item) => item === token).length);
}

function holdProb(score) {
  return probabilities([score, 0, 0])[0];
}

function pretrain(logits, counts, lr = 0) {
  return trainStep(logits, { method: "pretrain", counts, lr });
}

const policies = [
  [1 / 3, 1 / 3, 1 / 3],
  [0.6, 0.35, 0.05],
  [0.98, 0.01, 0.01],
];

const initialDirections = pretrain(zeroLogits, trainingCounts).metrics.gradient;
const validationCounts = [3, 2, 1];
const savedSteps = [0, 2, 24];
const checkpoints = [];
let logits = zeroLogits;
for (let step = 0; step <= savedSteps.at(-1); step++) {
  if (savedSteps.includes(step)) {
    checkpoints.push({
      step,
      trainLoss: pretrain(logits, trainingCounts).loss,
      validationLoss: pretrain(logits, validationCounts).loss,
      checkProbability: probabilities(logits)[2],
    });
  }
  if (step < savedSteps.at(-1))
    logits = pretrain(logits, trainingCounts, 1).logits;
}

export const pretrainSteps = [
  {
    id: "next-target",
    title: ["Find the next token", "找出下一个 token"],
    intro: [
      "Read one rover log: “The signal says HOLD”. In this toy example each space-separated word is one token. A training pair consists of a visible prefix and the one token immediately after it.",
      "读一行小车日志“The signal says HOLD”。这个例子把空格隔开的每个词算作一个 token。把原文遮住一截，露出的前文就是输入，紧跟前文的那个词就是目标。",
    ],
    task: [
      "Move the reading boundary until HOLD is the target.",
      "移动阅读边界，让 HOLD 恰好成为要预测的词。",
    ],
    controls: [
      {
        key: "prefix",
        label: ["Words visible", "露出几个词"],
        initial: 1,
        min: 1,
        max: 3,
      },
    ],
    view({ prefix }, t) {
      const context = prefixTokens.slice(0, prefix).join(" ");
      return {
        rows: [
          {
            label: t("Original log", "原始日志"),
            value: prefixTokens.join(" "),
          },
          { label: t("Visible input", "输入的前文"), value: context },
          {
            label: t("Training target", "预测目标"),
            value: prefixTokens[prefix],
          },
        ],
        formula: context + "   →   " + prefixTokens[prefix],
      };
    },
    check: ({ prefix }) => prefix === 3,
    feedback({ prefix }, t) {
      return prefix === 3
        ? t(
            "The model receives “The signal says” and the log supplies the target HOLD. We will reuse this exact prefix.",
            "把“The signal says”喂给模型，原文给出了答案 HOLD。后面几步都固定用这段前文。",
          )
        : t(
            `The next word is ${prefixTokens[prefix]}. Reveal one more word and watch the target move.`,
            `边界后面紧跟的是 ${prefixTokens[prefix]}。再露出一个词，看预测目标怎样往后移。`,
          );
    },
  },
  {
    id: "count-endings",
    title: ["Count endings for one prefix", "数一数这段前文后面接了什么"],
    intro: [
      "Four new logs begin with “The signal says”. Their endings in order are HOLD, HOLD, GO, CHECK. Taking the first few logs gives you a batch. The frequency of an ending is its count divided by the batch size.",
      "现在有四行日志，前文全是“The signal says”，结尾依次是 HOLD、HOLD、GO、CHECK。从开头取几行作为一批。一个词出现的次数除以这批总行数，就是它的频率。",
    ],
    task: [
      "Build the batch where HOLD occurs twice and GO once, with no CHECK.",
      "取出一批日志，让 HOLD 出现两次、GO 出现一次、CHECK 出现零次。",
    ],
    controls: [
      {
        key: "lines",
        label: ["Take the first N logs", "取前几行日志"],
        initial: 1,
        min: 1,
        max: 4,
      },
    ],
    view({ lines }, t) {
      const counts = readCounts(lines);
      return {
        rows: commands.map((command, i) => ({
          label: command,
          value: counts[i] + " / " + lines + " = " + percent(counts[i] / lines),
          fraction: counts[i] / lines,
        })),
        formula:
          t("Batch", "本批结尾") +
          " = [" +
          logs.slice(0, lines).join(", ") +
          "]",
      };
    },
    check: ({ lines }) => lines === 3,
    feedback({ lines }, t) {
      if (lines === 3)
        return t(
          "Two of three endings are HOLD. This batch supplies the target frequencies [⅔, ⅓, 0] used below.",
          "三行里有两行接 HOLD。下面用到的目标频率就来自这三行，依次是 [⅔, ⅓, 0]。",
        );
      return lines < 3
        ? t(
            "The GO example is the third log. Include that line in the batch.",
            "第三行才是 GO 的样本。把这一行也放进来。",
          )
        : t(
            "The fourth log adds a CHECK. Shorten the batch by one line.",
            "第四行多出来一个 CHECK。把批次缩短一行。",
          );
    },
  },
  {
    id: "score-to-chance",
    title: ["Turn scores into probabilities", "把三个分数换算成概率"],
    intro: [
      "The tiny model has one trainable score, or logit, per possible ending. HOLD, GO and CHECK start at zero. Softmax exponentiates the three scores, then divides each result by their sum.",
      "这个小模型给三个候选结尾各存一个分数，叫 logit。起初全为 0。softmax 先对每个分数取指数，再把每项除以三项之和，就得到三个概率。",
    ],
    task: [
      "Change only the HOLD score until its probability exceeds 50%.",
      "只调 HOLD 的分数，让它的概率超过 50%。",
    ],
    controls: [
      {
        key: "score",
        label: ["HOLD logit", "HOLD 分数"],
        initial: 0,
        min: 0,
        max: 2,
        step: 0.5,
      },
    ],
    view({ score }) {
      const p = probabilities([score, 0, 0]);
      return {
        rows: probabilityRows(p),
        formula: `P(HOLD) = exp(${score}) / (exp(${score}) + 1 + 1) = ${percent(p[0])}`,
      };
    },
    check: ({ score }) => holdProb(score) > 0.5,
    feedback({ score }, t) {
      return holdProb(score) > 0.5
        ? t(
            "GO and CHECK still have score zero. Their probabilities fell because all three tokens share the new denominator.",
            "GO 和 CHECK 的分数都保持 0。它们的概率却降低了，因为它们也要除以变大的那个总和。",
          )
        : t(
            "At score zero all three exponentials are 1. Increase HOLD until its exponential exceeds the other two combined.",
            "分数全为 0 时，三项指数都是 1。把 HOLD 加大到它的指数超过另外两项之和。",
          );
    },
  },
  {
    id: "token-loss",
    title: ["Score one prediction", "给一次预测算损失"],
    intro: [
      "Take just the first log. Its target is HOLD, so we look up the probability the model gave HOLD. Its loss is −ln(P(HOLD)). For example, giving the observed token a probability of 1/3 costs about 1.099.",
      "先只看第一行日志，答案是 HOLD，就从预测分布里取出 HOLD 的概率 p。用 −ln(p) 计算这一次的损失。比如取到 1/3，这条日志的损失约为 1.099。",
    ],
    task: [
      "Make this one log's loss less than 0.60.",
      "调节 HOLD 的分数，让这一行的损失小于 0.60。",
    ],
    controls: [
      {
        key: "score",
        label: ["HOLD logit", "HOLD 分数"],
        initial: 0,
        min: -1,
        max: 2,
        step: 0.5,
      },
    ],
    view({ score }, t) {
      const p = holdProb(score);
      return {
        rows: [
          { label: t("Target from log", "日志给的目标"), value: "HOLD" },
          { label: "P(HOLD)", value: percent(p), fraction: p },
          {
            label: t("Loss on this log", "这行日志的损失"),
            value: number(-Math.log(p)),
          },
        ],
        formula: `−ln(${number(p)}) ≈ ${number(-Math.log(p))}`,
      };
    },
    check: ({ score }) => -Math.log(holdProb(score)) < 0.6,
    feedback({ score }, t) {
      return -Math.log(holdProb(score)) < 0.6
        ? t(
            "The model assigned a higher probability to the observed token, so this example's loss fell.",
            "模型给正确词 HOLD 分的概率提高，这个样本的损失就降低了。",
          )
        : t(
            "Increasing HOLD’s logit raises P(HOLD). As that probability moves toward 1, its negative log moves toward 0.",
            "提高 HOLD 的分数会拉高 P(HOLD)。这个概率越接近 1，取负对数后的损失就越接近 0。",
          );
    },
  },
  {
    id: "batch-loss",
    title: ["Average three losses", "把三条日志的损失取平均"],
    intro: [
      "Return to the batch HOLD, HOLD, GO. Each line has its own loss. HOLD's loss enters twice, GO's once, and the sum is divided by three. Compare three sets of model probabilities.",
      "回到 HOLD、HOLD、GO 这三条日志。每一条都单独算损失。HOLD 的损失要算两次，GO 的算一次，加起来再除以 3。切换三种模型预测试试。",
    ],
    task: [
      "Select the prediction set with the lowest mean loss on these three logs.",
      "比较三种预测，选出让这三行平均损失最低的一组。",
    ],
    controls: [
      {
        key: "candidate",
        label: [
          "Probabilities in HOLD / GO / CHECK order",
          "按 HOLD / GO / CHECK 排列的概率",
        ],
        initial: 0,
        options: labels(["⅓ / ⅓ / ⅓", "60% / 35% / 5%", "98% / 1% / 1%"]),
      },
    ],
    view({ candidate }, t) {
      const [hold, go] = policies[candidate].map((p) => -Math.log(p));
      const avg = (2 * hold + go) / 3;
      return {
        rows: [
          {
            label: t("Loss on a HOLD log", "一条 HOLD 的损失"),
            value: number(hold),
          },
          { label: t("Loss on a GO log", "一条 GO 的损失"), value: number(go) },
          { label: t("Batch mean loss", "本批平均损失"), value: number(avg) },
        ],
        formula: `(2 × ${number(hold)} + ${number(go)}) / 3 ≈ ${number(avg)}`,
      };
    },
    check: ({ candidate }) => candidate === 1,
    feedback({ candidate }, t) {
      if (candidate === 1)
        return t(
          "This set gives HOLD 60% and still assigns GO 35%. Its three-line mean loss is about 0.690.",
          "这组给 HOLD 60%，也给了 GO 35%。三行的平均损失约为 0.690。",
        );
      return candidate === 2
        ? t(
            "GO has only 1% probability here. Its loss of 4.605 pushes the batch mean up even though both HOLD losses are tiny.",
            "这组只给 GO 1% 的概率。一条 GO 的损失达到 4.605，会把两条 HOLD 省下来的损失又抬高。",
          )
        : t(
            "Equal probabilities give equal losses on every line. Try a set that gives more probability to both observed endings.",
            "各词概率相同时，三行的损失也相同。试试把更多概率分给日志里真实出现的两个词。",
          );
    },
  },
  {
    id: "update-direction",
    title: ["Find each score's update direction", "找到每个分数的更新方向"],
    intro: [
      "Start a fresh model with three zero logits. For batch cross-entropy with softmax, the direction added in a gradient descent step is target frequency minus model probability. A positive difference raises that score.",
      "从三个分数都是 0 的模型开始。对 softmax 和这批平均损失做梯度下降时，每个分数要加的方向是「样本目标频率 − 当前预测概率」。差值为正，就要抬高这个分数。",
    ],
    task: [
      "Inspect the command whose score can stay still on this first update.",
      "切换查看三个词，找出第一次更新时分数正好可以保持原值的那个。",
    ],
    controls: [
      {
        key: "token",
        label: ["Inspect a command", "选一个词查看"],
        initial: 0,
        options: labels(commands),
      },
    ],
    view({ token }, t) {
      const target = trainingCounts[token] / 3;
      const prediction = probabilities(zeroLogits)[token];
      return {
        rows: [
          {
            label: t("Frequency in three logs", "三行样本中的频率"),
            value: percent(target),
            fraction: target,
          },
          {
            label: t("Model probability", "模型当前预测概率"),
            value: percent(prediction),
            fraction: prediction,
          },
          {
            label: t("Direction added to the logit", "往分数上加的方向"),
            value: sign(initialDirections[token]),
          },
        ],
        formula: `d(${commands[token]}) = ${trainingCounts[token]}/3 − 1/3 = ${sign(initialDirections[token])}`,
      };
    },
    check: ({ token }) => token === 1,
    feedback({ token }, t) {
      if (token === 1)
        return t(
          "GO appears once in three logs, and the model already gives it 1/3. GO’s first score update is zero.",
          "GO 在三行中占 1/3，模型也正好给它 1/3。两数相减为 0，所以 GO 的分数第一步保持原值。",
        );
      return token === 0
        ? t(
            "HOLD needs to rise from its initial 1/3 probability toward 2/3. Which command has frequency exactly 1/3?",
            "HOLD 的目标频率是 2/3，模型只给了 1/3，分数该往上调。哪个词的目标频率正好等于 1/3？",
          )
        : t(
            "CHECK appears zero times, so its score needs to fall. Find the ending that occurs once.",
            "CHECK 一次都没出现，分数该往下调。找找正好出现一次的那个结尾。",
          );
    },
  },
  {
    id: "take-a-step",
    title: ["Set the size of one update", "控制一次更新走多远"],
    intro: [
      "The first directions for HOLD, GO and CHECK are +⅓, 0 and −⅓. Multiply each by the learning rate and add it to its old score. Softmax then turns the three updated scores into probabilities.",
      "上一步算出的三个更新方向分别是 +⅓、0、−⅓。用同一个学习率乘每个方向，再加回各自的旧分数。最后把三个新分数一起过 softmax，就能看到这一步后的概率。",
    ],
    task: [
      "Adjust the learning rate until one update gives HOLD between 44% and 46%.",
      "调学习率，让只更新一次后的 HOLD 概率落在 44% 到 46% 之间。",
    ],
    controls: [
      {
        key: "rate",
        label: ["Learning rate", "学习率"],
        initial: 0,
        min: 0,
        max: 2,
        step: 0.25,
      },
    ],
    view({ rate }, t) {
      const result = pretrain(zeroLogits, trainingCounts, rate);
      return {
        rows: commands.map((cmd, i) => ({
          label: cmd,
          value:
            t("score ", "分数 ") +
            sign(result.logits[i]) +
            "   ·   " +
            percent(result.after[i]),
          fraction: result.after[i],
        })),
        formula: `HOLD: 0 + ${rate} × (2/3 − 1/3) ≈ ${number(result.logits[0])}`,
      };
    },
    check({ rate }) {
      const p = pretrain(zeroLogits, trainingCounts, rate).after[0];
      return p >= 0.44 && p <= 0.46;
    },
    feedback({ rate }, t) {
      if (rate === 1)
        return t(
          "At learning rate 1 the new logits are about [0.333, 0, −0.333]. HOLD rises from 33.3% to 44.8%.",
          "学习率为 1 时，新分数约为 [0.333, 0, −0.333]，HOLD 从 33.3% 升到 44.8%。",
        );
      return rate < 1
        ? t(
            "The step is still too small. Increase the rate and watch HOLD's new score.",
            "这一步走得还不够。调大学习率，再看 HOLD 的新分数。",
          )
        : t(
            "The new HOLD probability is above the target range. Reduce the learning rate.",
            "新的 HOLD 概率超过目标区间了。把学习率调小一点。",
          );
    },
  },
  {
    id: "held-out",
    title: ["Check a model on fresh logs", "用留出的日志选模型"],
    intro: [
      "Save the model at updates 0, 2 and 24 while training on HOLD, HOLD, GO with learning rate 1. Evaluate each saved model on six separate logs: three HOLD, two GO, one CHECK. Evaluation reads probabilities with the parameters held fixed.",
      "仍用 HOLD、HOLD、GO 这三条日志训练，学习率为 1，在第 0、2、24 步分别保存模型。另留六条日志专门验证，其中 HOLD 三条、GO 两条、CHECK 一条。验证时固定参数，只读概率来算平均损失。",
    ],
    task: [
      "Choose the saved model with the smallest loss on the six held-out logs.",
      "切换三个保存点，选出在这六条验证日志上平均损失最小的模型。",
    ],
    controls: [
      {
        key: "checkpoint",
        label: ["Saved training update", "选择第几步的模型"],
        initial: 0,
        options: labels(["0", "2", "24"]),
      },
    ],
    view({ checkpoint }, t) {
      const point = checkpoints[checkpoint];
      return {
        rows: [
          {
            label: t("Training loss · 3 logs", "训练损失 · 三条日志"),
            value: number(point.trainLoss),
          },
          {
            label: t("Validation loss · 6 logs", "验证损失 · 六条日志"),
            value: number(point.validationLoss),
          },
          {
            label: "P(CHECK)",
            value: percent(point.checkProbability),
            fraction: point.checkProbability,
          },
        ],
        formula:
          t("Validation mean", "验证均值") +
          " = −(3 ln P(HOLD) + 2 ln P(GO) + ln P(CHECK)) / 6",
      };
    },
    check: ({ checkpoint }) => checkpoint === 1,
    feedback({ checkpoint }, t) {
      if (checkpoint === 1)
        return t(
          "Update 2 has the lowest validation loss of these three checkpoints. At update 24 the model has trained down CHECK so far that the held-out CHECK log costs more.",
          "在这三个保存点中，第 2 步的验证损失最低。到第 24 步时，模型已经把 CHECK 的概率压得很低，验证集里那条 CHECK 就要付出更高的损失。",
        );
      return checkpoint === 0
        ? t(
            "The untrained model gives every command 1/3. Try a trained checkpoint and compare the six-log loss.",
            "第 0 步对三个词各给 1/3。切到训练过的保存点，对比六条验证日志的平均损失。",
          )
        : t(
            "Update 24 has the smallest training loss, but the held-out CHECK example now has little probability. Compare with update 2.",
            "第 24 步的训练损失最低，但留出的 CHECK 样本几乎分不到概率。再看看第 2 步的验证损失。",
          );
    },
  },
];
