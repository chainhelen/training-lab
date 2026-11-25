import { trainStep } from "./trainer.mjs";

const words = ["HOLD", "GO", "CHECK"];
const row = (label, value) => ({ label, value: String(value) });
const fmt = (loss) => loss.toFixed(3);
const zero = () => [0, 0, 0];
const valLoss = (scores, counts) =>
  trainStep(scores, { method: "pretrain", counts, lr: 0 }).loss;
const train = (scores, counts) =>
  trainStep(scores, { method: "pretrain", counts, lr: 0.8 }).logits;
const rotate = (values, offset) =>
  words.map((_, i) => values[(i + 3 - offset) % 3]);
const packetText = (counts) =>
  counts.flatMap((n, i) => Array(n).fill(words[i])).join(" · ");
const names = ["A", "B", "C"];

function shuffle(values, seed) {
  const result = [...values];
  let state = seed >>> 0;
  for (let i = result.length - 1; i > 0; i--) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const j = Math.floor((state / 4294967296) * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
function makePacks(base, seed) {
  // Change both packet order and the target the held-out set favors.
  const offset = (seed >>> 0) % 3;
  return shuffle(
    base.map((c) => rotate(c, offset)),
    seed,
  );
}
function ledger(turns, t) {
  return {
    columns: [
      t("Turn", "回合"),
      t("Packet", "资料袋"),
      t("Targets read", "读到的目标"),
      t("Validation loss", "验证损失"),
    ],
    rows: turns.map((entry, i) => [
      i + 1,
      names[entry.packet],
      packetText(entry.counts),
      fmt(entry.loss),
    ]),
  };
}
function choosePacket(s, id, limit, stock = Infinity) {
  const packet = names.indexOf(id);
  if (
    packet < 0 ||
    s.turns.length >= limit ||
    s.turns.filter((x) => x.packet === packet).length >= stock
  )
    return s;
  const counts = s.packs[packet];
  const logits = train(s.logits, counts);
  return {
    ...s,
    logits,
    turns: [...s.turns, { packet, counts, loss: valLoss(logits, s.holdout) }],
  };
}
const collection = {
  id: "collect-three",
  title: ["Three seats on the data rover", "采集车只有三个座位"],
  brief: [
    "Six sensor logs are ready to upload. The station and its one target word are printed on each card. Each record you collect takes one permanent seat for this run.",
    "六份传感器日志等着上传。每张卡列出采集站和这份日志的目标词。每收一份，就占用车上的一个座位。",
  ],
  target: [
    "Collect three different commands in exactly three uploads.",
    "用三次上传收齐三种命令。",
  ],
  guide: "shrink-corpus",
  start(seed) {
    return { deck: shuffle([0, 0, 0, 1, 1, 2], seed), taken: [] };
  },
  play(s, id) {
    const i = s.deck.findIndex((_, i) => id === "log-" + i);
    if (i < 0 || s.taken.includes(i) || s.taken.length >= 3) return s;
    return { ...s, taken: [...s.taken, i] };
  },
  view(s, t) {
    const endings = s.taken.map((i) => s.deck[i]);
    return {
      stats: [
        row(t("Seats left", "剩余座位"), 3 - s.taken.length),
        row(
          t("Commands covered", "已覆盖命令"),
          new Set(endings).size + " / 3",
        ),
      ],
      cards: s.deck.map((a, i) => ({
        id: "log-" + i,
        title: t("Station ", "采集站 ") + (i + 1),
        detail: t("Target word · ", "目标词 · ") + words[a],
        tag: s.taken.includes(i)
          ? t("On board", "已装车")
          : t("1 seat", "占 1 座"),
        disabled: s.taken.length === 3 || s.taken.includes(i),
      })),
      board: {
        columns: [
          t("Collected", "已收集"),
          t("Station", "采集站"),
          t("Target", "目标词"),
        ],
        rows: s.taken.map((i, n) => [n + 1, "#" + (i + 1), words[s.deck[i]]]),
      },
      note: t(
        "Three stations logged HOLD, two logged GO, and only one logged CHECK. Each station can upload once.",
        "三个站记下 HOLD，两个站记下 GO，只有一个站记下 CHECK。每个站只能上传一次。",
      ),
    };
  },
  grade(s, t) {
    const seen = new Set(s.taken.map((i) => s.deck[i]));
    const ok = seen.size === 3;
    const missing = words.filter((_, i) => !seen.has(i)).join(" / ");
    return {
      ok,
      message: ok
        ? t(
            "All three upload slots are full. The training set has one example of every command.",
            "三个座位都用完了。现在每种命令至少有一条训练样本。",
          )
        : s.taken.length < 3
          ? t("Still missing ", "还缺 ") + missing
          : t("The seats filled before you collected ", "座位用完了，缺少 ") +
            missing,
    };
  },
};

const rereading = {
  id: "reread-or-open",
  title: ["Open a packet or read it again?", "开新袋，还是再读一遍？"],
  brief: [
    "Each packet holds two records. Choose one packet per turn to update three scores. Opening a packet adds its two records to your collection. Reopening it trains on the same two again.",
    "每袋有两份日志。每回合选一袋，用两个目标更新模型的三个分数。新开一袋会增加两份不同的日志，再开同一袋就用原来的两份再训练一次。",
  ],
  target: [
    "Spend all four turns, read all three commands and finish at or below 1.06 validation loss.",
    "用完四回合，读到三种命令，并把验证损失降到 1.06 或以下。",
  ],
  guide: "repeat-the-subset",
  start(seed) {
    return {
      packs: makePacks(
        [
          [2, 0, 0],
          [0, 1, 1],
          [1, 1, 0],
        ],
        seed,
      ),
      holdout: rotate([4, 2, 2], (seed >>> 0) % 3),
      turns: [],
      logits: zero(),
    };
  },
  play: (s, id) => choosePacket(s, id, 4),
  view(s, t) {
    const distinct = new Set(s.turns.map((x) => x.packet));
    return {
      stats: [
        row(t("Target reads", "总读取次数"), s.turns.length * 2 + " / 8"),
        row(
          t("Distinct records read", "读过几份不同的日志"),
          distinct.size * 2,
        ),
        row(
          t("Validation loss", "验证损失"),
          fmt(valLoss(s.logits, s.holdout)),
        ),
      ],
      cards: s.packs.map((c, i) => ({
        id: names[i],
        title:
          (distinct.has(i) ? t("Reread ", "重读 ") : t("Open ", "打开 ")) +
          names[i],
        detail: packetText(c),
        tag: distinct.has(i)
          ? t("Same records", "旧样本")
          : t("2 fresh records", "2 份新样本"),
        disabled: s.turns.length >= 4,
      })),
      board: ledger(s.turns, t),
      note:
        t(
          "Fixed validation targets (HOLD / GO / CHECK): ",
          "固定验证集的目标数（HOLD / GO / CHECK）为 ",
        ) + s.holdout.join(" / "),
    };
  },
  grade(s, t) {
    const covered = words.filter((_, i) =>
      s.turns.some((q) => q.counts[i] > 0),
    ).length;
    const loss = valLoss(s.logits, s.holdout),
      ok = s.turns.length === 4 && covered === 3 && loss <= 1.06;
    return {
      ok,
      message: ok
        ? t(
            "Eight target reads and three commands covered. Final validation loss: ",
            "读了八个目标，三种命令都有覆盖。最终验证损失 ",
          ) + fmt(loss)
        : s.turns.length < 4
          ? t(
              "Use all four turns. Check the packet contents against the validation mix.",
              "先用满四回合。对照验证集配比，决定下一袋读什么。",
            )
          : t("Commands covered: ", "命令覆盖 ") +
            covered +
            "/3 · " +
            t("loss ", "损失 ") +
            fmt(loss) +
            t(
              ". Try rereading a packet containing the commands you undersampled.",
              "。可以重读一袋含有此前读得较少的命令的资料。",
            ),
    };
  },
};

const race = {
  id: "nine-target-race",
  title: ["Race on nine target tokens", "九个目标词的对照赛"],
  brief: [
    "A pilot model spends nine target reads in the order A, B, C. Start from the same zero scores, choose three packets and compare against it on the same held-out set. Each packet can be used twice.",
    "对照模型依次读 A、B、C 袋，一共读九个目标词。你也从三个零分出发，自选三袋，做同一份验证题。每袋最多读两遍。",
  ],
  target: [
    "Spend nine reads and finish with validation loss at least 0.010 below the pilot.",
    "读满九次，让验证损失比对照模型至少低 0.010。",
  ],
  guide: "fixed-read-budget",
  start(seed) {
    const offset = (seed >>> 0) % 3,
      available = makePacks(
        [
          [2, 1, 0],
          [0, 2, 1],
          [1, 0, 2],
        ],
        seed,
      ),
      holdout = rotate([5, 2, 1], offset);
    const pilot = [zero()];
    for (const packet of available) pilot.push(train(pilot.at(-1), packet));
    return {
      packs: available,
      holdout,
      pilot: pilot.map((z) => valLoss(z, holdout)),
      logits: zero(),
      turns: [],
    };
  },
  play: (s, id) => choosePacket(s, id, 3, 2),
  view(s, t) {
    return {
      stats: [
        row(
          t("Your target reads", "你的读取次数"),
          s.turns.length * 3 + " / 9",
        ),
        row(t("Your loss", "你的损失"), fmt(valLoss(s.logits, s.holdout))),
        row(
          t("Pilot at this turn", "对照模型本回合损失"),
          fmt(s.pilot[s.turns.length]),
        ),
      ],
      cards: s.packs.map((p, i) => {
        const uses = 2 - s.turns.filter((q) => q.packet === i).length;
        return {
          id: names[i],
          title: t("Train on ", "训练一袋 ") + names[i],
          detail: packetText(p),
          tag: t("Uses left: ", "还可读取 ") + uses,
          disabled: !uses || s.turns.length === 3,
        };
      }),
      board: {
        columns: [
          t("Turn", "回合"),
          t("Packet", "你的选择"),
          t("Your loss", "你的损失"),
          t("Pilot loss", "对照损失"),
        ],
        rows: s.turns.map((q, i) => [
          i + 1,
          names[q.packet],
          fmt(q.loss),
          fmt(s.pilot[i + 1]),
        ]),
      },
      note:
        t(
          "Held-out counts (HOLD / GO / CHECK): ",
          "验证集目标数（HOLD / GO / CHECK）为 ",
        ) +
        s.holdout.join(" / ") +
        t(". Target final loss ≤ ", "。通关损失上限为 ") +
        fmt(s.pilot[3] - 0.01) +
        t(" (rounded).", "（近似值）。"),
    };
  },
  grade(s, t) {
    const gain = s.pilot[3] - valLoss(s.logits, s.holdout),
      ok = s.turns.length === 3 && gain >= 0.01;
    return {
      ok,
      message: ok
        ? t(
            "Equal target budgets. You reduced held-out loss by ",
            "相同读取预算下，验证损失降低了 ",
          ) + fmt(gain)
        : s.turns.length < 3
          ? t(
              "Pick three packets. The most frequent held-out command is a clue to the mix.",
              "选满三袋。看一眼验证集中哪个目标最常见。",
            )
          : t("Your final loss was ", "你的最终损失为 ") +
            fmt(valLoss(s.logits, s.holdout)) +
            t(
              ". Compare each chosen packet with the commands in the test distribution.",
              "。把每一袋的目标和验证集的命令分布对照一下。",
            ),
    };
  },
};

const checkpoint = {
  id: "checkpoint-patrol",
  title: ["Save a checkpoint before the drift", "留住跑偏前的参数"],
  brief: [
    "Your training packet emphasizes one command. Each training turn rereads it and saves the new weights. A validation probe scores the current weights on separate records. You have ten training turns and just three probes.",
    "训练袋偏向一个命令。每训练一步就重读这一袋，保存这一步的权重。用一次评测机会可以在另外一批日志上给当前权重打分。最多训练十步，有三次评测机会。",
  ],
  target: [
    "Probe a checkpoint below the target loss. Probe a later checkpoint with higher loss, then select the best one you measured.",
    "测出一份低于目标损失的参数，再测出一份训练更久、验证损失更高的参数，最后选中已评测里损失最低的那份。",
  ],
  guide: "held-out",
  start(seed) {
    const offset = (seed >>> 0) % 3;
    const scenario = (seed >>> 0) % 3;
    const validation = [
      [3, 1, 1],
      [4, 2, 1],
      [4, 1, 1],
    ][scenario];
    return {
      data: rotate([4, 1, 0], offset),
      holdout: rotate(validation, offset),
      target: [0.97, 0.98, 0.9][scenario],
      logits: zero(),
      n: 0,
      probes: [],
      selected: null,
    };
  },
  play(s, id) {
    if (id === "train" && s.n < 10)
      return { ...s, n: s.n + 1, logits: train(s.logits, s.data) };
    if (
      id === "probe" &&
      s.probes.length < 3 &&
      !s.probes.some((p) => p.n === s.n)
    )
      return {
        ...s,
        probes: [...s.probes, { n: s.n, loss: valLoss(s.logits, s.holdout) }],
      };
    const match = s.probes.find((p) => "save-" + p.n === id);
    if (match && s.selected !== match.n) return { ...s, selected: match.n };
    return s;
  },
  view(s, t) {
    const last = s.probes.at(-1);
    return {
      stats: [
        row(t("Updates", "训练步数"), s.n + " / 10"),
        row(t("Probes left", "剩余评测次数"), 3 - s.probes.length),
        row(t("Training loss", "训练损失"), fmt(valLoss(s.logits, s.data))),
      ],
      cards: [
        {
          id: "train",
          title: t("Train one step", "再训练一步"),
          detail: t("Reread the five training targets.", "重读五个训练目标。"),
          disabled: s.n >= 10,
        },
        {
          id: "probe",
          title: t("Probe this checkpoint", "评测当前参数"),
          detail:
            t("Spend one probe on step ", "花一次机会评测第 ") +
            s.n +
            t(".", " 步。"),
          disabled: s.probes.length >= 3 || last?.n === s.n,
        },
        ...s.probes.map((p) => ({
          id: "save-" + p.n,
          title: t("Select step ", "选第 ") + p.n + t("", " 步"),
          detail: t("Measured loss ", "实测损失 ") + fmt(p.loss),
          tag: p.n === s.selected ? t("Selected", "已选") : undefined,
          disabled: p.n === s.selected,
        })),
      ],
      board: {
        columns: [
          t("Probed step", "评测步数"),
          t("Validation loss", "验证损失"),
          t("Deployment choice", "是否选用"),
        ],
        rows: s.probes.map((p) => [
          p.n,
          fmt(p.loss),
          s.selected === p.n ? "✓" : "—",
        ]),
      },
      note:
        t("Validation target < ", "验证损失目标 < ") +
        fmt(s.target) +
        t(
          ". Train / validation counts in HOLD, GO, CHECK order: ",
          "。按 HOLD、GO、CHECK 排列的训练集 / 验证集计数为 ",
        ) +
        s.data.join(", ") +
        " / " +
        s.holdout.join(", ") +
        t(". Try probing after two updates.", "。可以先训练两步再评测。"),
    };
  },
  grade(s, t) {
    const sel = s.probes.find((p) => p.n === s.selected);
    if (!sel)
      return {
        ok: false,
        message: t(
          "Probe a checkpoint, then select a measured step.",
          "先评测一份参数，再从已测的步数中选一份。",
        ),
      };
    if (sel.loss >= s.target)
      return {
        ok: false,
        message: t(
          "That checkpoint has not reached the validation target.",
          "这份参数还没有达到验证损失目标。",
        ),
      };
    if (s.probes.some((p) => p.loss < sel.loss))
      return {
        ok: false,
        message: t(
          "Another measured checkpoint has lower validation loss.",
          "你已经测出过验证损失更低的参数。",
        ),
      };
    const ok = s.probes.some((p) => p.n > sel.n && p.loss > sel.loss);
    return {
      ok,
      message: ok
        ? t(
            "You caught the turn: later updates kept fitting the training packet while held-out loss rose. The selected earlier snapshot is ready.",
            "你抓到了拐点：模型继续拟合这袋训练数据时，验证损失却升高了。你保留了更早、验证更好的参数。",
          )
        : t(
            "Train further and spend a probe to check for a rise in held-out loss.",
            "再训练几步，用一次评测看验证损失有没有回升。",
          ),
    };
  },
};
export const budgetMissions = [collection, rereading, race, checkpoint];
