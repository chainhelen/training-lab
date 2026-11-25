// The round-one SFT checkpoint gives exactly P(HOLD, GO, CHECK) = (1/2, 1/4, 1/4).
// These guided examples share a fixed starting point so each can be replayed.
import { probabilities, trainStep } from "./trainer.mjs";

const names = ["HOLD", "GO", "CHECK"];
const options = names.map((value) => [value, value]);
const initialLogits = [0, 0, 0];
const rate = Math.log(2);
const pct = (value) => (100 * value).toFixed(1) + "%";
const dec = (value) => value.toFixed(3);
const selector = (key, label, initial, options) => ({
  key,
  label,
  initial,
  options,
});
const slider = (key, label, initial, min, max, step = 1) => ({
  key,
  label,
  initial,
  min,
  max,
  step,
});
const data = (label, value) => ({ label, value: String(value) });
const prob = (label, fraction) => ({ label, value: pct(fraction), fraction });
const probRows = (values) =>
  names.map((name, index) => prob("P(" + name + ")", values[index]));
function supervisedAt(rounds) {
  let logits = initialLogits;
  for (let i = 0; i < rounds; i++)
    logits = trainStep(logits, { method: "sft", target: 0, lr: rate }).logits;
  return logits;
}
const reference = supervisedAt(1);
const refProbs = probabilities(reference);
const onOff = [
  ["Off", "关"],
  ["On", "开"],
];
const command = [
  "Stop the rover. Reply with one command.",
  "让小车停车。用一个命令回答。",
];

export const sftSteps = [
  {
    id: "write-demo",
    title: ["Write one demonstration", "写一条示范回答"],
    intro: [
      "An instruction is the input. A demonstration is the answer you want the model to imitate. HOLD stops the rover, GO starts it, and CHECK reads its sensors.",
      "指令是输入。示范回答是你希望模型模仿的输出。HOLD 让车停下，GO 让车前进，CHECK 读取传感器。",
    ],
    task: [
      "Build a training example for a driver who asks the rover to stop.",
      "司机让车停下。给这条指令配上正确的示范回答。",
    ],
    controls: [
      selector(
        "answer",
        ["Assistant demonstration", "助手的示范回答"],
        1,
        options,
      ),
    ],
    view: ({ answer }, t) => ({
      rows: [
        data(t("User", "用户"), t(...command)),
        data(t("Assistant", "助手"), names[answer]),
      ],
      formula: t(
        "In the training pair, this answer becomes the target.",
        "组成训练样本后，这个回答就是模型要学的目标。",
      ),
    }),
    check: ({ answer }) => answer === 0,
    feedback: ({ answer }, t) =>
      answer === 0
        ? t(
            "You have one instruction–answer pair. Next, mark where its training target begins.",
            "这就组成了一对「指令、回答」。下一步标出从哪里开始对答案算分。",
          )
        : t(
            names[answer] +
              (answer === 1
                ? " makes the rover move. Which command applies the brakes?"
                : " reads the sensors. Which command applies the brakes?"),
            names[answer] +
              (answer === 1
                ? " 会让车前进。哪个命令能刹车？"
                : " 会读取传感器。哪个命令能刹车？"),
          ),
  },
  {
    id: "mask-prompt",
    title: ["Mark the answer span", "圈出要算损失的部分"],
    intro: [
      "The example has two spans: the driver's complete instruction and the assistant's one-token answer HOLD. In completion-only SFT, the model reads both spans during training. A mask selects which target positions contribute to the loss.",
      "这条样本分两段：司机的整条指令，以及助手回答的一个 token「HOLD」。训练时模型会读这两段，掩码决定哪些位置的预测要计入损失。",
    ],
    task: [
      "Keep the whole instruction as context. Score only the assistant's answer.",
      "保留整条指令作前文，只对助手的回答算损失。",
    ],
    controls: [
      selector(
        "prompt",
        ["Score prompt targets", "对指令位置算损失"],
        1,
        onOff,
      ),
      selector("reply", ["Score answer target", "对回答位置算损失"], 0, onOff),
    ],
    view: ({ prompt, reply }, t) => {
      const flag = (v) =>
        v ? t("included", "计入损失") : t("masked", "屏蔽损失");
      return {
        rows: [
          data(t("Context read by the model", "模型读到的前文"), t(...command)),
          data(t("Prompt targets", "指令位置的目标"), flag(prompt)),
          data("HOLD", flag(reply)),
        ],
        formula:
          t("Loss mask by span", "按两段显示的损失掩码") +
          " = [" +
          prompt +
          ", " +
          reply +
          "]",
      };
    },
    check: ({ prompt, reply }) => prompt === 0 && reply === 1,
    feedback: ({ prompt, reply }, t) =>
      prompt
        ? t(
            "The instruction supplies context. Switch off its target loss.",
            "指令给回答提供上下文。先关掉指令位置的损失。",
          )
        : !reply
          ? t(
              "Switch on the target for HOLD so there is an answer to learn.",
              "打开 HOLD 位置的损失，才有回答可学。",
            )
          : t(
              "Each prompt token stays visible as context. HOLD supplies the single scored target in this example.",
              "指令里的 token 仍然作为前文可见。这条样本中，HOLD 是唯一计入损失的目标。",
            ),
  },
  {
    id: "score-demo",
    title: ["Read the demonstration loss", "读懂这个回答的损失"],
    intro: [
      "Suppose three candidate policies give the demonstrated answer different probabilities. Its loss is −ln P(HOLD). Higher confidence in this demonstration gives a smaller loss.",
      "假设三个候选模型给示范回答的概率各不相同。这一条回答的损失等于 −ln P(HOLD)。模型给示范回答的概率越高，损失就越小。",
    ],
    task: [
      "Find the policy that gives HOLD half of the probability. Read its loss.",
      "找到把一半概率分给 HOLD 的模型，读出这条示范回答的损失。",
    ],
    controls: [
      selector("policy", ["Candidate policy", "候选模型"], 0, [
        ["A", "A"],
        ["B", "B"],
        ["C", "C"],
      ]),
    ],
    view: ({ policy }, t) => {
      const h = [1 / 3, 0.5, 0.75][policy];
      return {
        rows: [
          ...probRows([h, (1 - h) / 2, (1 - h) / 2]),
          data(t("Loss on HOLD", "HOLD 的损失"), dec(-Math.log(h))),
        ],
        formula: "L = −ln(" + dec(h) + ") = " + dec(-Math.log(h)),
      };
    },
    check: ({ policy }) => policy === 1,
    feedback: ({ policy }, t) =>
      policy === 1
        ? t(
            "At P(HOLD) = 0.5 the loss is about 0.693. Next, change a model's scores to reach that probability.",
            "当 P(HOLD) = 0.5，这一条的损失约为 0.693。下一步你会真的更新分数，让模型达到这个概率。",
          )
        : t(
            "Inspect the HOLD row. Look for 50.0%.",
            "看 HOLD 这一行，目标是 50.0%。",
          ),
  },
  {
    id: "update-demo",
    title: ["Apply one supervised update", "真正更新一次"],
    intro: [
      "Start with three scores of zero. Softmax gives each command one third. For the HOLD demonstration, each score moves by learning rate × (target − probability). The learning rate here is ln 2, about 0.693.",
      "现在从三个 0 分开始，softmax 后三个命令各占三分之一。用 HOLD 作示范，每个分数的改变量是「学习率 ×（目标值 − 当前概率）」。本例学习率取 ln 2，约 0.693。",
    ],
    task: [
      "Apply just enough updates for HOLD to reach 50.0%.",
      "调节更新次数，让 HOLD 恰好达到 50.0%。",
    ],
    controls: [
      slider(
        "rounds",
        ["Updates from the same start", "从同一起点更新几次"],
        0,
        0,
        3,
      ),
    ],
    view: ({ rounds }, t) => {
      const logits = supervisedAt(rounds);
      return {
        rows: probRows(probabilities(logits)),
        formula:
          t("Scores [HOLD, GO, CHECK]", "分数 [HOLD, GO, CHECK]") +
          " = [" +
          logits.map(dec).join(", ") +
          "]",
      };
    },
    check: ({ rounds }) => rounds === 1,
    feedback: ({ rounds }, t) =>
      rounds === 1
        ? t(
            "One step raises the HOLD score to 0.462 and lowers each other score to −0.231. The new probabilities are 50%, 25%, 25%.",
            "一步后 HOLD 的分数升到 0.462，其他两个各降到 −0.231。三个新概率分别为 50%、25%、25%。",
          )
        : rounds === 0
          ? t(
              "The scores are still tied. Apply one update using HOLD as the target.",
              "三个分数还相同。以 HOLD 为目标，先更新一次。",
            )
          : t(
              "These additional updates keep increasing P(HOLD). Move back to the step where it first reached half.",
              "继续更新会让 P(HOLD) 进一步升高。退回到它刚好占一半的那一步。",
            ),
  },
  {
    id: "copy-checkpoint",
    title: ["Keep a copy of round one", "留住第一轮的参数"],
    intro: [
      "The live model has now taken two SFT updates. We kept its earlier scores as snapshots. A checkpoint is a saved copy of the parameters at one moment.",
      "在线模型已经更新两次。我们还记录了较早的分数。把某一时刻的参数存下来，就是一个 checkpoint。",
    ],
    task: [
      "Load the snapshot that still gives HOLD 50% while the live model stays at round two.",
      "挑出仍然让 HOLD 占 50% 的快照，同时让在线模型保留第二轮的参数。",
    ],
    controls: [
      slider("snapshot", ["Saved round to load", "读取哪一轮快照"], 2, 0, 2),
    ],
    view: ({ snapshot }, t) => ({
      rows: [
        prob(
          t("Live P(HOLD) · round 2", "在线 P(HOLD) · 第 2 轮"),
          probabilities(supervisedAt(2))[0],
        ),
        prob(
          t("Saved P(HOLD)", "快照 P(HOLD)"),
          probabilities(supervisedAt(snapshot))[0],
        ),
        prob(
          t("Saved P(GO)", "快照 P(GO)"),
          probabilities(supervisedAt(snapshot))[1],
        ),
      ],
      formula:
        t("Loaded scores", "载入的分数") +
        " = [" +
        supervisedAt(snapshot).map(dec).join(", ") +
        "]",
    }),
    check: ({ snapshot }) => snapshot === 1,
    feedback: ({ snapshot }, t) =>
      snapshot === 1
        ? t(
            "The saved policy gives HOLD 50% and GO 25%. The DPO exercises use this snapshot as a frozen reference.",
            "保存下来的模型给 HOLD 50%、给 GO 25%。接下来的 DPO 小实验会用这份快照作为固定参考。",
          )
        : t(
            "Compare the saved HOLD probability with the live one. Find the earlier 50% checkpoint.",
            "比较快照和在线模型的 HOLD 概率，找回较早那个 50% 的快照。",
          ),
  },
];

function dpoAt(logits, beta = 1, lr = 0) {
  return trainStep(logits, {
    method: "dpo",
    chosen: 0,
    rejected: 1,
    reference,
    beta,
    lr,
  });
}
function policyWithOdds(odds) {
  // HOLD stays at 1/2. GO and CHECK share the remaining mass.
  const go = 0.5 / odds;
  return [0.5, go, 0.5 - go];
}
function oddsRows(probs, t) {
  return [
    prob("P(HOLD)", probs[0]),
    prob("P(GO)", probs[1]),
    data(
      t("HOLD / GO odds", "HOLD / GO 概率比"),
      (probs[0] / probs[1]).toFixed(2) + "×",
    ),
  ];
}
export const dpoSteps = [
  {
    id: "pair-replies",
    title: ["Build one preference pair", "把两个回答配成一对"],
    intro: [
      "Use the same instruction to stop the rover. This time the log includes two different answers and a human judgment about which one is better.",
      "仍然是让小车停车。这次日志里有两份不同的回答，以及人对这两份回答的取舍。",
    ],
    task: [
      "Create the comparison that prefers applying the brakes over moving forward.",
      "组出这样一对偏好：停车的回答胜过前进的回答。",
    ],
    controls: [
      selector("chosen", ["Preferred answer", "更喜欢的回答"], 1, options),
      selector("rejected", ["Rejected answer", "更不喜欢的回答"], 0, options),
    ],
    view: ({ chosen, rejected }, t) => ({
      rows: [
        data(t("Shared instruction", "共同的指令"), t(...command)),
        data(t("Preferred", "更喜欢"), names[chosen]),
        data(t("Rejected", "更不喜欢"), names[rejected]),
      ],
      formula:
        chosen === rejected
          ? t(
              "The two slots contain the same answer.",
              "两个位置填了同一个回答。",
            )
          : names[chosen] + " ≻ " + names[rejected],
    }),
    check: ({ chosen, rejected }) => chosen === 0 && rejected === 1,
    feedback: ({ chosen, rejected }, t) =>
      chosen === rejected
        ? t(
            "Pick two different answers so that the preference has a direction.",
            "选两个不同的回答，偏好才有方向。",
          )
        : chosen !== 0
          ? t(
              "Which answer would a driver want for a stop instruction? Place it in the preferred slot.",
              "司机要停车，把能停车的回答放到更喜欢一栏。",
            )
          : rejected !== 1
            ? t(
                "The comparison asks for the forward command in the rejected slot.",
                "这对偏好指定了对照项，把前进命令放进更不喜欢一栏。",
              )
            : t(
                "The pair is HOLD ≻ GO. Every following calculation uses this direction.",
                "这对记录是 HOLD 胜过 GO。后面每一个比值都沿着这个方向算。",
              ),
  },
  {
    id: "divide-odds",
    title: ["Compare two probabilities", "两个回答的概率怎么比"],
    intro: [
      "Load the SFT snapshot. HOLD has probability 0.50 and GO has probability 0.25. Divide the preferred probability by the rejected probability to get the odds for this pair.",
      "取出 SFT 的快照。HOLD 的概率是 0.50，GO 的概率是 0.25。用更喜欢的回答概率除以更不喜欢的回答概率，得到这一对的概率比。",
    ],
    task: [
      "Put the preferred command in the numerator and obtain the pair's 2× odds.",
      "把更喜欢的命令放在分子，算出这对回答的 2 倍概率比。",
    ],
    controls: [
      selector("direction", ["Order of the division", "相除的顺序"], 0, [
        ["GO / HOLD", "GO / HOLD"],
        ["HOLD / GO", "HOLD / GO"],
      ]),
    ],
    view: ({ direction }, t) => {
      const top = direction ? 0 : 1,
        bottom = 1 - top;
      return {
        rows: [
          prob("P(" + names[top] + ")", refProbs[top]),
          prob("P(" + names[bottom] + ")", refProbs[bottom]),
          data(
            t("Ratio", "比值"),
            (refProbs[top] / refProbs[bottom]).toFixed(2) + "×",
          ),
        ],
        formula:
          names[top] +
          " / " +
          names[bottom] +
          " = " +
          refProbs[top].toFixed(2) +
          " / " +
          refProbs[bottom].toFixed(2),
      };
    },
    check: ({ direction }) => direction === 1,
    feedback: ({ direction }, t) =>
      direction
        ? t(
            "HOLD is twice as likely as GO under the reference. Keep this 2× ratio for the next comparison.",
            "在参考模型里，HOLD 的概率是 GO 的 2 倍。记住这个 2，下一步拿它作对照。",
          )
        : t(
            "GO / HOLD answers the reverse question. Put HOLD first to follow the recorded judgment.",
            "GO 除以 HOLD 算了相反方向。把 HOLD 放在前面，与刚才的偏好方向一致。",
          ),
  },
  {
    id: "compare-reference",
    title: ["Compare against the frozen odds", "与固定的参考比值再比一次"],
    intro: [
      "Keep the reference at HOLD / GO = 2. Inspect policies that keep HOLD at 50% while changing how GO and CHECK share the other half.",
      "参考的 HOLD / GO 固定为 2。现在查看几种候选策略：HOLD 都占 50%，剩余一半在 GO 与 CHECK 之间的分法不同。",
    ],
    task: [
      "Make the live HOLD / GO odds twice the reference odds.",
      "让在线模型的 HOLD / GO 概率比变为参考模型的两倍。",
    ],
    controls: [
      slider("odds", ["Live HOLD / GO odds", "在线 HOLD / GO 概率比"], 2, 2, 6),
    ],
    view: ({ odds }, t) => ({
      rows: [
        ...oddsRows(policyWithOdds(odds), t),
        data(t("Frozen odds", "参考概率比"), "2.00×"),
        data(
          t("Live odds ÷ reference odds", "在线比值 ÷ 参考比值"),
          (odds / 2).toFixed(2) + "×",
        ),
      ],
      formula: odds + " / 2 = " + (odds / 2).toFixed(2),
    }),
    check: ({ odds }) => odds === 4,
    feedback: ({ odds }, t) =>
      odds === 4
        ? t(
            "The live pair is 4× and the reference pair is 2×. The relative factor is 2 even though P(HOLD) stayed at 50%.",
            "在线概率比是 4，参考概率比是 2，所以相对倍数为 2。此时 HOLD 本身的概率仍然是 50%。",
          )
        : t(
            "Divide the live ratio by the fixed 2× reference. Which setting gives a relative factor of 2?",
            "在线模型的比值还要除以参考的 2。调到哪个数，最后的相对倍数才会等于 2？",
          ),
  },
  {
    id: "scale-margin",
    title: ["Turn that factor into a preference score", "把这个倍数换成偏好分"],
    intro: [
      "The pair improved by a factor of 2. DPO takes ln(2), multiplies it by β, then applies sigmoid. The result scores the observed preference between the two replies.",
      "刚才算出相对倍数为 2。DPO 对它取 ln，再乘 β，然后过 sigmoid，得到模型对「这两个回答里更喜欢 HOLD」这一判断的分数。",
    ],
    task: [
      "Set β so that the margin equals ln(2), then read the predicted preference probability.",
      "调节 β，让间隔恰好等于 ln(2)，再读出预测的偏好概率。",
    ],
    controls: [
      selector("beta", ["Scale β", "系数 β"], 0, [
        ["0.5", "0.5"],
        ["1", "1"],
        ["2", "2"],
      ]),
    ],
    view: ({ beta }, t) => {
      const b = [0.5, 1, 2][beta],
        p = policyWithOdds(4),
        r = dpoAt(p.map(Math.log), b);
      return {
        rows: [
          data(t("Scaled margin", "缩放后的间隔"), dec(r.metrics.margin)),
          prob(
            t("P(prefer HOLD to GO)", "P(更喜欢 HOLD 而非 GO)"),
            r.metrics.preference,
          ),
          prob(t("P(emit HOLD)", "P(生成 HOLD)"), p[0]),
        ],
        formula:
          "m = " +
          b +
          " × ln(4 / 2) = " +
          dec(r.metrics.margin) +
          ";  σ(m) = " +
          dec(r.metrics.preference),
      };
    },
    check: ({ beta }) => beta === 1,
    feedback: ({ beta }, t) =>
      beta === 1
        ? t(
            "With β = 1, the preference score is 2/3. The token probability is still 1/2. The two numbers answer different questions.",
            "β = 1 时，预测的偏好概率为 2/3；单独生成 HOLD 的概率仍然是 1/2。前一个数衡量成对的偏好，后一个数用于采样。",
          )
        : t(
            "The log odds factor is already ln(2). Choose the scale that preserves it.",
            "取对数后已经是 ln(2)，选择能保留这个数值的系数。",
          ),
  },
  {
    id: "update-pair",
    title: ["Apply a DPO gradient", "把偏好真正写进分数"],
    intro: [
      "Start a fresh actor at the SFT checkpoint. Its odds and the frozen reference odds both start at 2×. Now apply HOLD ≻ GO updates with β = 1 and learning rate 1.",
      "把在线模型重新放回 SFT 的起点，两边此时的 HOLD / GO 都是 2。固定参考模型，对在线模型按「HOLD 胜过 GO」做更新，这里 β 和学习率都取 1。",
    ],
    task: [
      "Apply exactly one pair update. Watch the actor's odds and the reference side by side.",
      "恰好执行一次偏好更新，对照看在线模型和参考模型的概率比。",
    ],
    controls: [
      slider(
        "rounds",
        ["Pair updates from checkpoint", "从快照做几次偏好更新"],
        0,
        0,
        4,
      ),
    ],
    view: ({ rounds }, t) => {
      let logits = reference;
      for (let i = 0; i < rounds; i++) logits = dpoAt(logits, 1, 1).logits;
      const scored = dpoAt(logits);
      const p = probabilities(logits);
      return {
        rows: [
          ...oddsRows(p, t),
          data(t("Frozen odds", "参考概率比"), "2.00×"),
          data(
            t("DPO loss after selected updates", "所选更新次数后的 DPO 损失"),
            dec(scored.loss),
          ),
        ],
        formula:
          "−ln P(" +
          names[0] +
          " ≻ " +
          names[1] +
          ") = −ln(" +
          dec(scored.metrics.preference) +
          ") = " +
          dec(scored.loss),
      };
    },
    check: ({ rounds }) => rounds === 1,
    feedback: ({ rounds }, t) =>
      rounds === 1
        ? t(
            "The first update adds 0.5 to HOLD's score and subtracts 0.5 from GO's. The actor odds reach 5.44× while the reference stays at 2×.",
            "第一步给 HOLD 的分数加 0.5，给 GO 的分数减 0.5。在线概率比变成约 5.44，参考概率比仍为 2。",
          )
        : rounds === 0
          ? t(
              "The two policies currently match. Take one gradient step from that starting point.",
              "当前在线模型与参考模型相同。从这个起点往前更新一次。",
            )
          : t(
              "The actor has taken several steps. Return to the first one to inspect its update.",
              "在线模型已经更新多次。退回第一步，看这一对偏好第一次改变了哪些数值。",
            ),
  },
];
