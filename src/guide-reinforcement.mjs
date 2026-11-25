// A single prompt and a reproducible rollout let learners follow one reply
// from sampling through scoring, normalization and the policy update.
import { sampleActions, trainStep } from "./trainer.mjs";

const tokens = ["HOLD", "GO", "CHECK"];
const oldPolicy = [0.5, 0.25, 0.25].map(Math.log);
const seed = 549;
const group = sampleActions(oldPolicy, 8, seed);
const firstFour = group.slice(0, 4);
const exact = (action) => Number(action === 0);
const shortest = (action) => Number(action === 1);
const row = (label, value) => ({ label, value: String(value) });
const percent = (p) => (100 * p).toFixed(1) + "%";
const bar = (label, p) => ({ label, value: percent(p), fraction: p });
const signed = (n) => (n > 0 ? "+" : "") + n.toFixed(2);
const sequence = (actions) => actions.map((i) => tokens[i]).join(" · ");
const slider = (key, label, min, max, initial, step = 1) => ({
  key,
  label,
  min,
  max,
  initial,
  step,
});
const choice = (key, label, initial, options) => ({
  key,
  label,
  initial,
  options,
});
const follow = (step, v, t, yes, no) => t(...(step(v) ? yes : no));
const ruleChoice = choice(
  "rule",
  ["Reward each reply for", "每份回答凭什么得分"],
  0,
  [
    ["Shortest command", "命令最短"],
    ["Correct stop command", "能让车停下"],
  ],
);
function reinforce(rule = 1, lr = 1) {
  return trainStep(oldPolicy, {
    method: "rl",
    actions: firstFour,
    rewards: firstFour.map(rule ? exact : shortest),
    baseline: 0,
    lr,
  });
}
function policyAt(holdPercent) {
  const p = holdPercent / 100;
  return [p, (1 - p) / 2, (1 - p) / 2].map(Math.log);
}
function grpo({
  logits = oldPolicy,
  rewards = group.map(exact),
  reference = oldPolicy,
  kl = 0,
  lr = 0,
} = {}) {
  return trainStep(logits, {
    method: "grpo",
    actions: group,
    rewards,
    old: oldPolicy,
    reference,
    clip: 0.2,
    kl,
    lr,
  });
}

const samplesGo = ({ ticket }) => ticket >= 50 && ticket < 75;
const fixedReward = ({ rule }) => rule === 1;
const goIsReinforced = ({ rule }) => rule === 0;
const halfBaseline = ({ b }) => b === 50;
const raiseHold = ({ rate }) => reinforce(1, rate / 100).after[0] >= 0.59;
const freshPolicy = ({ policy }) => policy === 1;
export const rlSteps = [
  {
    id: "draw-one",
    title: ["Draw one reply", "先抽一个回答"],
    intro: [
      "The instruction is “Stop the rover.” HOLD, GO and CHECK have probabilities 50%, 25% and 25%. Lay those chances out on 100 numbered tickets.",
      "指令是“停车”。HOLD、GO、CHECK 的概率分别是 50%、25%、25%。把这三个概率排成 100 张编号签，每张签对应一个回答。",
    ],
    task: ["Pick a ticket that samples GO.", "选一张会抽到 GO 的签。"],
    controls: [
      slider("ticket", ["Ticket number (0–99)", "签号（0–99）"], 0, 99, 12),
    ],
    view(v, t) {
      const a = v.ticket < 50 ? 0 : v.ticket < 75 ? 1 : 2;
      return {
        rows: [
          row("HOLD", "0–49"),
          row("GO", "50–74"),
          row("CHECK", "75–99"),
          row(t("Sampled reply", "采样回答"), tokens[a]),
        ],
        formula: `#${v.ticket} → ${tokens[a]}`,
      };
    },
    check: samplesGo,
    feedback(v, t) {
      return follow(
        samplesGo,
        v,
        t,
        [
          "GO owns 25 tickets. A random draw can pick it even though HOLD has the largest probability.",
          "GO 占 25 张签。哪怕 HOLD 概率最高，也可能抽到 GO。",
        ],
        [
          "Look for GO’s interval between the HOLD and CHECK tickets.",
          "看看哪段签号位于 HOLD 和 CHECK 之间，落到 GO。",
        ],
      );
    },
  },
  {
    id: "score-rollout",
    title: ["Score the sampled replies", "给抽出的回答打分"],
    intro: [
      "Ask “Stop the rover” four times using a fixed random seed. The sampled commands below stay on the bench while you change the scoring rule.",
      "用固定随机种子，对“停车”采样四次。下面四份回答保持原样，你来改变打分规则。",
    ],
    task: [
      "Give both HOLD replies 1 point and the other replies 0.",
      "让两份 HOLD 各得 1 分，让另两份都得 0 分。",
    ],
    controls: [ruleChoice],
    view(v, t) {
      const scores = firstFour.map(v.rule ? exact : shortest);
      return {
        rows: firstFour.map((a, i) =>
          row(`#${i + 1}  ${tokens[a]}`, `R = ${scores[i]}`),
        ),
        formula: `${t("Rewards", "奖励")} = [${scores.join(", ")}]`,
      };
    },
    check: fixedReward,
    feedback(v, t) {
      return follow(
        fixedReward,
        v,
        t,
        [
          "Both HOLD replies receive 1 because both stop the rover.",
          "两份 HOLD 都让车停车，因此各拿 1 分。",
        ],
        [
          "GO has only two letters. This rule rewards length. Try scoring the command’s meaning.",
          "GO 只有两个字母，当前规则奖励的是长度。试试按停车指令的含义打分。",
        ],
      );
    },
  },
  {
    id: "reward-steers",
    title: ["Reward a shortcut", "试一次会教错的奖励"],
    intro: [
      "Keep the same four replies. Preview one REINFORCE update under each scoring rule. Watch which command becomes more likely.",
      "沿用刚才的四份回答，预览按不同规则做一次强化学习更新的结果。看看哪条命令更容易被抽到了。",
    ],
    task: [
      "Make GO more likely, even though the instruction says to stop.",
      "指令明明是“停车”，试着让 GO 的概率反而上升。",
    ],
    controls: [{ ...ruleChoice, initial: 1 }],
    view(v, t) {
      const r = reinforce(v.rule);
      return {
        rows: r.after.map((p, i) => ({
          ...bar(tokens[i], p),
          value: `${percent(r.before[i])} → ${percent(p)}`,
        })),
        formula: `${t("Rewarded reply", "获得奖励的回答")} = ${v.rule ? "HOLD" : "GO"}`,
      };
    },
    check: goIsReinforced,
    feedback(v, t) {
      return follow(
        goIsReinforced,
        v,
        t,
        [
          "GO earns the reward for brevity and its probability rises. The score you specify defines the incentive.",
          "GO 凭最短得分，更新后概率上升。你写的打分规则决定了模型努力的方向。",
        ],
        [
          "The current scorer rewards HOLD. Which rule gives GO the best score?",
          "当前评分器奖励 HOLD。哪种规则会让 GO 得分最高？",
        ],
      );
    },
  },
  {
    id: "baseline",
    title: ["Give a score a comparison point", "给奖励找一个比较点"],
    intro: [
      "Return to the correct-command reward. HOLD earns 1 and GO earns 0. Subtract a fixed baseline b from each reward to obtain an advantage A.",
      "换回奖励停车的规则。HOLD 得 1 分，GO 得 0 分。从奖励里减去固定基线 b，得到优势值 A。",
    ],
    task: [
      "Set the baseline so HOLD has advantage +0.5 and GO has −0.5.",
      "调整基线，让 HOLD 的优势为 +0.5、GO 为 −0.5。",
    ],
    controls: [
      slider(
        "b",
        ["Baseline in hundredths", "基线（以 0.01 为单位）"],
        0,
        100,
        0,
        25,
      ),
    ],
    view(v, t) {
      const b = v.b / 100;
      return {
        rows: [
          row(t("Baseline b", "基线 b"), b),
          row("HOLD", signed(1 - b)),
          row("GO", signed(-b)),
        ],
        formula: `A(HOLD) = 1 − ${b} = ${signed(1 - b)}\nA(GO) = 0 − ${b} = ${signed(-b)}`,
      };
    },
    check: halfBaseline,
    feedback(v, t) {
      return follow(
        halfBaseline,
        v,
        t,
        [
          "A positive advantage favors a sampled response. A negative advantage pushes against it.",
          "正优势会鼓励刚抽到的回答，负优势会压低它。",
        ],
        [
          "Place b halfway between rewards 0 and 1.",
          "把基线放在 0 分和 1 分的中间。",
        ],
      );
    },
  },
  {
    id: "take-step",
    title: ["Turn reward into a parameter update", "把打分变成参数更新"],
    intro: [
      "Use the four sampled replies and the correct-command scorer. With the practice baseline back at zero, the update direction of the HOLD score is +0.25. Try different step sizes from the same starting policy.",
      "仍用那四份回答，只奖励停车。这次沿用综合关的零基线，算出的 HOLD 分数更新方向为 +0.25。从同一起点试不同步长。",
    ],
    task: [
      "Raise HOLD from 50% to at least 59% in one update.",
      "做一次更新，把 HOLD 的概率从 50% 提高到至少 59%。",
    ],
    controls: [
      slider(
        "rate",
        ["Learning rate in hundredths", "学习率（以 0.01 为单位）"],
        0,
        100,
        0,
        25,
      ),
    ],
    view(v, t) {
      const r = reinforce(1, v.rate / 100);
      return {
        rows: [
          bar(t("HOLD before", "更新前 HOLD"), r.before[0]),
          bar(t("HOLD after", "更新后 HOLD"), r.after[0]),
          row(
            t("Change in HOLD logit", "HOLD 分数的变化"),
            signed((v.rate / 100) * r.metrics.gradient[0]),
          ),
        ],
        formula: `Δscore(HOLD) = ${v.rate / 100} × ${r.metrics.gradient[0].toFixed(2)}`,
      };
    },
    check: raiseHold,
    feedback(v, t) {
      return follow(
        raiseHold,
        v,
        t,
        [
          "The new scores go through softmax. The next rollout uses the resulting probabilities.",
          "三个新分数一起经过 softmax。下一批采样就用这组新概率。",
        ],
        [
          "Increase the learning rate and watch the probability after the update.",
          "提高学习率，留意“更新后 HOLD”的概率。",
        ],
      );
    },
  },
  {
    id: "fresh-rollout",
    title: ["Sample again after training", "更新后重新采样"],
    intro: [
      "The trained policy now gives HOLD about 59.3%. Replay four random draws with the same seed, once with the old scores and once with the updated scores.",
      "刚才更新后，HOLD 的概率约为 59.3%。现在取四个相同的随机数，用旧分数和新分数各抽一批，比较回答。",
    ],
    task: [
      "Draw the next on-policy batch using the updated scores.",
      "用更新后的分数，为下一次 REINFORCE 更新采一批回答。",
    ],
    controls: [
      choice("policy", ["Sample from", "采样时使用"], 0, [
        ["Before the update", "更新前的分数"],
        ["After the update", "更新后的分数"],
      ]),
    ],
    view(v, t) {
      const r = reinforce(),
        actions = sampleActions(v.policy ? r.logits : oldPolicy, 4, 1);
      return {
        rows: [
          bar(
            t("Sampling P(HOLD)", "采样用的 P(HOLD)"),
            v.policy ? r.after[0] : r.before[0],
          ),
          row(t("Four replies", "四份回答"), sequence(actions)),
        ],
        formula: `seed 1 · ${v.policy ? "π_new" : "π_old"} → [${actions.map((i) => tokens[i]).join(", ")}]`,
      };
    },
    check: freshPolicy,
    feedback(v, t) {
      return follow(
        freshPolicy,
        v,
        t,
        [
          "The same random draw at position 3 now lands on HOLD. A fresh rollout records which policy produced the data.",
          "同一个随机数在第 3 个位置现在抽到了 HOLD。新采样的一批会记录这次用的策略。",
        ],
        [
          "Look for the policy with the new 59.3% probability.",
          "选刚得到的、HOLD 概率为 59.3% 的策略。",
        ],
      );
    },
  },
];

const fullGroup = ({ n }) => n === 8;
const inspectLast = ({ i }) => i === 8;
const correctMean = ({ divisor }) => divisor === 8;
const centreCorrect = ({ b }) => b === 50;
const normalized = ({ scale }) => scale === 1;
const allEqual = ({ rule }) => rule === 1;
const ratioCorrect = ({ p }) => p === 60;
const clippedHold = ({ p }) => grpo({ logits: policyAt(p) }).metrics.clipped[0];
const useKL = ({ strength }) => strength > 0;
const recipeGroups = ({ n }) => n === 8;
export const grpoSteps = [
  {
    id: "one-prompt-group",
    title: ["One prompt, several replies", "一道题，采好几份回答"],
    intro: [
      "Start from the same 50 / 25 / 25 policy. Repeat “Stop the rover” and keep all the sampled replies together. The first four draws match the scoring exercise.",
      "回到 50%、25%、25% 的策略。对同一条“停车”指令采样多次，把这几份回答放在一组。前四份与打分练习相同。",
    ],
    task: [
      "Grow the group to eight replies while keeping one prompt.",
      "保持只有一道题，把回答数量增到八份。",
    ],
    controls: [slider("n", ["Replies per prompt", "每题回答数"], 1, 8, 1)],
    view(v, t) {
      const a = sampleActions(oldPolicy, v.n, seed);
      return {
        rows: [
          row(t("Distinct prompts", "题目数"), 1),
          row(t("Sampled replies", "回答数"), a.length),
          row(t("Draw order", "按采样次序"), sequence(a)),
        ],
        formula: `1 × ${v.n} = ${v.n}`,
      };
    },
    check: fullGroup,
    feedback(v, t) {
      return follow(
        fullGroup,
        v,
        t,
        [
          "All eight answers belong to this one instruction. Keep this group for the following calculations.",
          "八份回答都属于这道停车指令。接下来沿用这组数据算分。",
        ],
        [
          "Increase the replies per prompt. The prompt count stays at one.",
          "把每题的回答数调大，题目数始终是 1。",
        ],
      );
    },
  },
  {
    id: "score-group",
    title: ["Pair each reply with its reward", "给每份回答配上自己的分数"],
    intro: [
      "Number the eight replies from 1 to 8. The reward function gives every HOLD a 1, every GO or CHECK a 0.",
      "给八份回答从 1 到 8 编号。评分器遇到 HOLD 给 1 分，遇到 GO 或 CHECK 给 0 分。",
    ],
    task: [
      "Inspect the last reply and find its own reward.",
      "选中最后一份回答，查看它的奖励。",
    ],
    controls: [slider("i", ["Reply number", "回答编号"], 1, 8, 1)],
    view(v, t) {
      const a = group[v.i - 1];
      return {
        rows: [
          row(t("All replies", "八份回答"), sequence(group)),
          row(t("Selected reply", "选中的回答"), `#${v.i} · ${tokens[a]}`),
          row(t("Its reward", "这份的奖励"), exact(a)),
        ],
        formula: `R[#${v.i}] = reward(${tokens[a]}) = ${exact(a)}`,
      };
    },
    check: inspectLast,
    feedback(v, t) {
      return follow(
        inspectLast,
        v,
        t,
        [
          "Reply 8 is HOLD and earns 1. Four replies in this group earned 1 and the other four earned 0.",
          "第 8 份是 HOLD，得 1 分。八份回答里四份得 1 分，四份得 0 分。",
        ],
        [
          "Move to reply 8 to inspect the end of the same group.",
          "移到第 8 份，查看同一组里的最后一份回答。",
        ],
      );
    },
  },
  {
    id: "group-mean",
    title: ["Average the eight rewards", "算八个奖励的平均数"],
    intro: [
      "The reward vector is [1,0,1,0,1,0,0,1]. The four ones sum to 4. Form the average by choosing a divisor.",
      "奖励向量是 [1,0,1,0,1,0,0,1]，四个 1 加起来等于 4。给这个总分挑一个除数，算全组平均分。",
    ],
    task: [
      "Choose the divisor that gives the average reward per reply.",
      "选对除数，算出每份回答的平均奖励。",
    ],
    controls: [slider("divisor", ["Divisor", "除数"], 1, 8, 4)],
    view(v, t) {
      return {
        rows: [
          row(t("Total reward", "总奖励"), 4),
          row(t("Replies", "回答数"), 8),
          row(t("Your quotient", "当前算出的商"), (4 / v.divisor).toFixed(3)),
        ],
        formula: `4 ÷ ${v.divisor} = ${(4 / v.divisor).toFixed(3)}`,
      };
    },
    check: correctMean,
    feedback(v, t) {
      return follow(
        correctMean,
        v,
        t,
        [
          "Four points shared across eight replies gives a group mean of 0.5.",
          "4 分分摊到 8 份回答上，全组均值为 0.5。",
        ],
        [
          "The zero-reward replies belong in the denominator too.",
          "得 0 分的回答也要计入分母。",
        ],
      );
    },
  },
  {
    id: "centre-rewards",
    title: ["Subtract the group mean", "每份奖励减去组均值"],
    intro: [
      "Choose a single comparison score b for all eight replies. Subtract b from each raw reward. A reply above the group mean gets a positive difference.",
      "八份回答共用一个比较分数 b。每份的原始奖励都减去 b，高于全组平均分的回答会得到正差值。",
    ],
    task: [
      "Use the group mean and make the eight differences sum to zero.",
      "用组均值当基线，让八个差值加起来为 0。",
    ],
    controls: [
      slider(
        "b",
        ["Comparison score in hundredths", "比较分数（以 0.01 为单位）"],
        0,
        100,
        0,
        25,
      ),
    ],
    view(v, t) {
      const b = v.b / 100;
      return {
        rows: [
          row("#1 HOLD", signed(1 - b)),
          row("#2 GO", signed(-b)),
          row(t("Sum of differences", "全组差值之和"), signed(4 - 8 * b)),
        ],
        formula: `4 × (1 − ${b}) + 4 × (0 − ${b}) = ${signed(4 - 8 * b)}`,
      };
    },
    check: centreCorrect,
    feedback(v, t) {
      return follow(
        centreCorrect,
        v,
        t,
        [
          "HOLD sits 0.5 above the group mean and the others sit 0.5 below it.",
          "HOLD 比组均值高 0.5 分，其他回答都比组均值低 0.5 分。",
        ],
        [
          "The previous step gave the group mean as 4/8. Use that same value here.",
          "上一小步算出组均值是 4/8，在这里也用这个值。",
        ],
      );
    },
  },
  {
    id: "scale-advantages",
    title: ["Scale the differences", "把差值换成优势值"],
    intro: [
      "The centered differences are ±0.5. This trainer divides those by the sample standard deviation. Each squared difference is 0.25. With eight replies, s = √(8×0.25/7) ≈ 0.535.",
      "上一步的差值为 ±0.5。这套训练器再用样本标准差来除它们。每个差值的平方都是 0.25，所以 s = √(8×0.25/7) ≈ 0.535。",
    ],
    task: [
      "Apply standard-deviation scaling and inspect the advantage of reply 1.",
      "应用标准差缩放，查看第 1 份回答的优势值。",
    ],
    controls: [
      choice("scale", ["Value to inspect", "查看的值"], 0, [
        ["Centered score", "减去均值后的差值"],
        ["Normalized advantage", "除以标准差后的优势"],
      ]),
    ],
    view(v, t) {
      const m = grpo().metrics,
        shown = (i) =>
          v.scale ? m.advantages[i] : exact(group[i]) - m.meanReward;
      return {
        rows: [
          row(
            t("Sample standard deviation", "样本标准差"),
            m.rewardStd.toFixed(3),
          ),
          row("#1 HOLD", signed(shown(0))),
          row("#2 GO", signed(shown(1))),
        ],
        formula: v.scale
          ? `A₁ = (1 − 0.5) / (√(2/7) + 0.000001) ≈ ${m.advantages[0].toFixed(3)}`
          : "R₁ − mean(R) = 1 − 0.5 = 0.5",
      };
    },
    check: normalized,
    feedback(v, t) {
      return follow(
        normalized,
        v,
        t,
        [
          "Reply 1 receives advantage about +0.94, while reply 2 receives about −0.94.",
          "第 1 份的优势约为 +0.94，第 2 份约为 −0.94。",
        ],
        [
          "Switch from the centered score to the normalized advantage.",
          "从差值切换到标准化后的优势值。",
        ],
      );
    },
  },
  {
    id: "zero-reward-spread",
    title: ["Give every reply the same score", "让全组回答都同分"],
    intro: [
      "Preview one update with KL off. Keep all eight answers, and change only the reward rule. Look at the spread and at P(HOLD) after the update.",
      "关闭 KL 惩罚，预览一次更新。八份回答保持原样，只改打分规则，观察标准差和更新后的 HOLD 概率。",
    ],
    task: [
      "Make every advantage zero while still awarding points.",
      "让每份回答都能得分，同时让八个优势值全为 0。",
    ],
    controls: [
      choice("rule", ["Reward rule", "奖励规则"], 0, [
        ["1 for HOLD only", "只有 HOLD 得 1 分"],
        ["1 for every reply", "每份都得 1 分"],
      ]),
    ],
    view(v, t) {
      const r = grpo({
        rewards: group.map((a) => (v.rule ? 1 : exact(a))),
        lr: 0.8,
      });
      return {
        rows: [
          row(
            t("Reward spread (sample s)", "奖励标准差"),
            r.metrics.rewardStd.toFixed(3),
          ),
          row(
            t("Advantages", "八个优势值"),
            r.metrics.advantages.map(signed).join("  "),
          ),
          bar(t("After update · HOLD", "更新后 HOLD"), r.after[0]),
        ],
        formula: `P(HOLD): ${percent(r.before[0])} → ${percent(r.after[0])}`,
      };
    },
    check: allEqual,
    feedback(v, t) {
      return follow(
        allEqual,
        v,
        t,
        [
          "Each 1-point answer equals the mean of 1. All eight reward gradients are zero, so this update leaves HOLD at 50%.",
          "每份奖励都是 1，组均值也是 1。八份回答的奖励梯度都为 0，这步更新让 HOLD 保持在 50%。",
        ],
        [
          "Look for a scoring rule that gives HOLD, GO and CHECK equal rewards.",
          "选一条让 HOLD、GO 和 CHECK 得相同奖励的规则。",
        ],
      );
    },
  },
  {
    id: "old-current-ratio",
    title: ["Compare now with sampling time", "对照采样当时和现在的概率"],
    intro: [
      "The eight replies were sampled when P(HOLD) was 0.50. Freeze that old probability and move the current HOLD probability. Their ratio says how much more likely the same reply has become.",
      "采这八份回答时，P(HOLD) 是 0.50。锁住这个旧概率，调节当前策略的 HOLD 概率。两者的比值表示同一个回答现在有多容易出现。",
    ],
    task: [
      "Make the current HOLD probability 1.20 times the sampling probability.",
      "让当前 HOLD 概率是采样当时的 1.20 倍。",
    ],
    controls: [
      slider("p", ["Current P(HOLD), %", "当前 P(HOLD)，%"], 40, 80, 50, 5),
    ],
    view(v, t) {
      const r = grpo({ logits: policyAt(v.p) });
      return {
        rows: [
          bar(t("Old · sampling", "采样时的旧策略"), 0.5),
          bar(t("Current · training", "训练时的当前策略"), r.before[0]),
          row(
            t("Probability ratio", "概率比值"),
            r.metrics.ratios[0].toFixed(2),
          ),
        ],
        formula: `r₁ = ${(v.p / 100).toFixed(2)} / 0.50 = ${r.metrics.ratios[0].toFixed(2)}`,
      };
    },
    check: ratioCorrect,
    feedback(v, t) {
      return follow(
        ratioCorrect,
        v,
        t,
        [
          "At 60%, the ratio is 0.60/0.50 = 1.20. The denominator stays attached to the saved rollout.",
          "调到 60% 时，比值是 0.60/0.50 = 1.20。分母一直取这批回答采样时记录的值。",
        ],
        [
          "Multiply the old 50% probability by 1.20.",
          "用旧概率 50% 乘 1.20，得到要调出的当前概率。",
        ],
      );
    },
  },
  {
    id: "clip-one-term",
    title: ["Clip one reply’s contribution", "裁剪一份回答的贡献"],
    intro: [
      "Restore the correctness rewards. Reply 1 is HOLD with positive advantage. For this reply, a clip width of 0.20 caps the favorable ratio in its objective at 1.20. Move the actual ratio across the upper edge.",
      "恢复正确性奖励。第 1 份 HOLD 是正优势。裁剪宽度为 0.20 时，这份回答在目标函数里的有利比值最多用到 1.20。试着越过上边界。",
    ],
    task: [
      "Increase P(HOLD) until reply 1’s term is clipped and stops gaining.",
      "提高 P(HOLD)，直到第 1 份回答的目标项被裁剪、停止增大。",
    ],
    controls: [
      slider("p", ["Current P(HOLD), %", "当前 P(HOLD)，%"], 40, 80, 50, 5),
    ],
    view(v, t) {
      const m = grpo({ logits: policyAt(v.p) }).metrics;
      const r = m.ratios[0],
        a = m.advantages[0];
      return {
        rows: [
          row(t("Actual ratio", "真实概率比"), r.toFixed(2)),
          row(t("Unclipped term", "未裁剪的目标项"), (r * a).toFixed(3)),
          row(
            t("Term used for this reply", "第 1 份实际贡献"),
            (Math.min(r, 1.2) * a).toFixed(3),
          ),
        ],
        formula: `A₁ > 0: min(${r.toFixed(2)}, 1.20) × ${a.toFixed(3)}`,
      };
    },
    check: clippedHold,
    feedback(v, t) {
      return follow(
        clippedHold,
        v,
        t,
        [
          "The ratio can keep rising. This positive-advantage reply’s objective term has reached a plateau. Other replies can still contribute gradients.",
          "真实概率比还可以增大，这份正优势回答的目标项进入平台。其他回答仍可能提供梯度。",
        ],
        [
          "The actual ratio must pass 1.20. Try a current probability above 60%.",
          "让真实概率比越过 1.20，试试把当前概率调到 60% 以上。",
        ],
      );
    },
  },
  {
    id: "reference-kl",
    title: ["Bring back the reference policy", "看参考模型怎样牵引策略"],
    intro: [
      "Give all eight replies 1 again. Current P(HOLD) is now 70%, while a frozen reference assigns it 50%. With zero reward advantage you can isolate the KL penalty’s effect.",
      "重新让八份回答都得 1 分。这次当前策略给 HOLD 70%，冻结的参考模型给它 50%。奖励优势全为 0，可以单看 KL 惩罚的效果。",
    ],
    task: [
      "Enable the penalty so a single step pulls HOLD toward the reference.",
      "开启惩罚，让一次更新把 HOLD 的概率拉向参考模型。",
    ],
    controls: [
      slider(
        "strength",
        ["KL weight in tenths", "KL 权重（以 0.1 为单位）"],
        0,
        10,
        0,
      ),
    ],
    view(v, t) {
      const r = grpo({
        logits: policyAt(70),
        rewards: group.map(() => 1),
        kl: v.strength / 10,
        lr: 1,
      });
      return {
        rows: [
          bar(t("Frozen reference", "冻结参考模型"), 0.5),
          bar(t("Current before step", "更新前"), r.before[0]),
          bar(t("Current after step", "更新后"), r.after[0]),
        ],
        formula: `λ = ${v.strength / 10}; KL(π || ref) = ${r.metrics.kl.toFixed(3)}`,
      };
    },
    check: useKL,
    feedback(v, t) {
      return follow(
        useKL,
        v,
        t,
        [
          "The score differences stay zero. The change comes entirely from the penalty for drift from the frozen reference.",
          "奖励差值依然全为 0，这一步的变化完全来自偏离参考模型的惩罚。",
        ],
        [
          "At weight zero the reference has no influence. Give λ a positive value.",
          "权重为 0 时参考模型不起作用，把 λ 调到一个正值。",
        ],
      );
    },
  },
  {
    id: "rollout-config",
    title: ["Map your group to a verl setting", "把这组回答对应到 verl 配置"],
    intro: [
      "In verl’s GRPO recipe, actor_rollout_ref.rollout.n sets the number of responses sampled for each prompt. Try a batch with two prompts. Each prompt gets its own reward group.",
      "在 verl 的 GRPO 示例里，actor_rollout_ref.rollout.n 设置每道题的采样回答数。这里一批给两道题，两道题分别计算自己的奖励均值。",
    ],
    task: [
      "Configure two groups with eight replies each.",
      "配置出两组，每组正好八份回答。",
    ],
    controls: [slider("n", ["rollout.n", "rollout.n"], 2, 10, 2)],
    view(v, t) {
      return {
        rows: [
          row(t("Group A · stop", "A 组 · 停车"), v.n),
          row(t("Group B · proceed", "B 组 · 前进"), v.n),
          row(t("Total sampled responses", "本批总回答数"), 2 * v.n),
        ],
        formula: `actor_rollout_ref.rollout.n=${v.n}\n2 × ${v.n} = ${2 * v.n}`,
      };
    },
    check: recipeGroups,
    feedback(v, t) {
      return follow(
        recipeGroups,
        v,
        t,
        [
          "You sampled 16 replies. The stop group uses its own eight rewards as a comparison, and the proceed group uses the other eight.",
          "总共采样 16 份回答。停车题用自己的八个奖励比较，前进题用另外八个奖励比较。",
        ],
        [
          "The setting counts replies per prompt. Set it to eight.",
          "这个配置数的是每道题的回答数，把它设为八。",
        ],
      );
    },
  },
];

export { tokens, group, firstFour, exact, shortest, grpo, policyAt };
