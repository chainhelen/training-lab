// Every example predicts a single command after a common log prefix.
import { probabilities, trainStep } from "./trainer.mjs";

const fixed = (n) => n.toFixed(3);
const pct = (p) => (100 * p).toFixed(1) + "%";
const index = (seed, n) => (seed >>> 0) % n;
const stat = (label, value) => ({ label, value: String(value) });
const domains = (t) => [
  t("Road", "道路"),
  t("Worksite", "工地"),
  t("Workshop", "车间"),
];
const sumCols = (rows) =>
  rows.reduce((out, r) => out.map((n, i) => n + r[i]), [0, 0, 0]);
function scoreUpdate(z, counts, lr) {
  return trainStep(z, { method: "pretrain", counts, lr }).logits;
}

// Buy sealed four-token bundles. Only their source counts matter to this round.
const bundles = [
  [4, 0, 0],
  [2, 2, 0],
  [0, 3, 1],
  [1, 0, 3],
  [2, 1, 1],
  [0, 1, 3],
];
const manifests = [
  [0, 2, 3],
  [1, 2, 4],
  [0, 1, 5],
  [2, 3, 5],
  [0, 3, 4],
  [1, 3, 5],
];
const cargoTotal = (s) => sumCols(s.picks.map((i) => bundles[i]));
const cargoMission = {
  id: "pack-data",
  title: ["Pack twelve training tokens", "配齐十二个训练 token"],
  brief: [
    "You have three slots for data bundles. Each bundle contains four next-token targets collected on roads, worksites or in workshops. Compare the labels before spending a slot.",
    "你有三个数据包的位置。每包有四个要预测的词，来自道路、工地或车间日志。花掉一个位置之前，先看包里的来源数量。",
  ],
  target: [
    "Fill the three slots and match the requested counts for all three sources.",
    "装满三包，让三种来源的目标词数都符合本次订单。",
  ],
  guide: "domain-budget",
  start(seed) {
    return {
      order: index(seed, 6),
      goal: sumCols(
        manifests[index(seed, manifests.length)].map((i) => bundles[i]),
      ),
      picks: [],
    };
  },
  play(s, id) {
    const i = bundles.findIndex((_, j) => id === "pack-" + j);
    if (i < 0 || s.picks.includes(i) || s.picks.length >= 3) return s;
    return { ...s, picks: [...s.picks, i] };
  },
  view(s, t) {
    const total = cargoTotal(s),
      names = domains(t);
    return {
      stats: [
        stat(t("Cargo slots", "已装包数"), s.picks.length + " / 3"),
        stat(t("Target positions", "已装目标数"), 4 * s.picks.length + " / 12"),
      ],
      cards: bundles.map((_, k) => {
        const i = (k + s.order) % bundles.length;
        return {
          id: "pack-" + i,
          title: t("Load packet ", "装入数据包 ") + String.fromCharCode(65 + i),
          detail: bundles[i].map((n, j) => names[j] + " " + n).join(" · "),
          tag: s.picks.includes(i)
            ? t("Loaded", "已装入")
            : t("4 targets", "4 个目标"),
          disabled: s.picks.length >= 3 || s.picks.includes(i),
        };
      }),
      board: {
        columns: [
          t("Source", "来源"),
          t("Requested", "订单要求"),
          t("Loaded", "已装入"),
          t("Still needed", "还差"),
        ],
        rows: names.map((n, i) => [
          n,
          s.goal[i],
          total[i],
          s.goal[i] - total[i],
        ]),
      },
      note: t(
        "Count target tokens across all three bundles. A negative balance means that source is over quota.",
        "把三包里的目标词按来源相加。“还差”出现负数，表示这个来源装多了。",
      ),
    };
  },
  grade(s, t) {
    const total = cargoTotal(s),
      i = total.findIndex((n, i) => n !== s.goal[i]);
    if (s.picks.length < 3)
      return {
        ok: false,
        message: t(
          "There are still empty cargo slots.",
          "还有没装满的数据包位置。",
        ),
      };
    return i === -1
      ? {
          ok: true,
          message: t(
            "All twelve tokens match the order. Each source now gets its requested share of the training budget.",
            "十二个目标词都已装好，各来源正好拿到订单上要求的训练份额。",
          ),
        }
      : {
          ok: false,
          message:
            domains(t)[i] +
            t(" needs ", "需要 ") +
            s.goal[i] +
            t(" targets. Your bundles supply ", " 个目标，你装入了 ") +
            total[i] +
            t(".", "。"),
        };
  },
};

// Two unchanged categorical policies and two test domains with single targets.
const modelProbs = [
  [0.12, 0.82, 0.06],
  [0.68, 0.26, 0.06],
];
function fleetLoss(probs, road) {
  return -(road * Math.log(probs[1]) + (10 - road) * Math.log(probs[0])) / 10;
}
const auditKeys = ["road", "site", "pooled"];
const fleetMission = {
  id: "dispatch-model",
  title: ["Choose a checkpoint for the next shift", "为下一班车选模型"],
  brief: [
    "One model is better on roads (GO), the other on worksites (HOLD). The old report summarizes the previous shift. Open two reports, then dispatch a checkpoint for the incoming shift.",
    "道路日志的目标是 GO，工地日志的目标是 HOLD，两个模型各有所长。旧报表统计的是上一班车。打开两份报告，再为下一班选模型。",
  ],
  target: [
    "Inspect both domains separately. Dispatch the model with the lower mean loss for the incoming mix of ten targets.",
    "分别查看两个领域，再给下一班的十个目标选出平均损失较低的模型。",
  ],
  guide: "domain-report",
  start(seed) {
    return {
      road: [2, 8, 3, 7][index(seed, 4)],
      flipped: index(Math.floor(seed / 2), 2),
      opened: [],
      choice: null,
    };
  },
  play(s, id) {
    if (auditKeys.includes(id)) {
      if (s.opened.includes(id) || s.opened.length === 2) return s;
      return { ...s, opened: [...s.opened, id] };
    }
    if (id === "model-0" || id === "model-1") {
      const n = Number(id.at(-1));
      if (s.choice === n) return s;
      return { ...s, choice: n };
    }
    return s;
  },
  view(s, t) {
    const models = [modelProbs[s.flipped], modelProbs[1 - s.flipped]],
      labels = [
        t("Road losses", "道路损失"),
        t("Worksite losses", "工地损失"),
        t("Balanced test", "各占一半的测试"),
      ];
    return {
      stats: [
        stat(
          t("Incoming road / worksite", "下一班的道路 / 工地"),
          s.road + " / " + (10 - s.road),
        ),
        stat(t("Report slots", "已用报告名额"), s.opened.length + " / 2"),
        stat(
          t("Dispatch choice", "派出的模型"),
          s.choice === null ? "—" : "AB"[s.choice],
        ),
      ],
      cards: [
        ...auditKeys.map((k, i) => ({
          id: k,
          title: labels[i],
          detail:
            i === 2
              ? t(
                  "Five road targets and five worksite targets",
                  "五条道路日志加五条工地日志",
                )
              : t("Loss for one source", "单独查看这一类"),
          disabled: s.opened.length === 2 || s.opened.includes(k),
        })),
        ...models.map((_, i) => ({
          id: "model-" + i,
          title: t("Dispatch model ", "派出模型 ") + "AB"[i],
          tag: s.choice === i ? t("Selected", "已选择") : "",
          disabled: s.choice === i,
        })),
      ],
      board: {
        columns: [t("Mean loss per target", "每个目标的平均损失"), "A", "B"],
        rows: [
          [
            t("Previous road/worksite ", "上一班道路/工地 ") +
              (10 - s.road) +
              "/" +
              s.road,
            ...models.map((p) => fixed(fleetLoss(p, 10 - s.road))),
          ],
          ...s.opened.map((k) => [
            labels[auditKeys.indexOf(k)],
            ...models.map((p) =>
              fixed(fleetLoss(p, k === "road" ? 10 : k === "site" ? 0 : 5)),
            ),
          ]),
        ],
      },
      note: t(
        "Incoming loss = (road count × road loss + worksite count × worksite loss) ÷ 10. Lower is better.",
        "下一班平均损失 =（道路条数 × 道路损失 + 工地条数 × 工地损失）÷ 10，越低越好。",
      ),
    };
  },
  grade(s, t) {
    if (!s.opened.includes("road") || !s.opened.includes("site"))
      return {
        ok: false,
        message: t(
          "You need each source’s loss to reweight the old report. Spend a report slot on roads and one on worksites.",
          "要按下一班的构成重新加权，需要道路和工地各自的损失。把两个报告名额各留一个。",
        ),
      };
    if (s.choice === null)
      return {
        ok: false,
        message: t(
          "Choose the checkpoint to dispatch.",
          "还需要选一个要派出的模型。",
        ),
      };
    const a = fleetLoss(modelProbs[s.flipped], s.road),
      b = fleetLoss(modelProbs[1 - s.flipped], s.road),
      ok = s.choice === Number(b < a);
    return {
      ok,
      message:
        t("On this shift, A scores ", "按下一班的构成，A 的损失为 ") +
        fixed(a) +
        t(" and B scores ", "，B 为 ") +
        fixed(b) +
        t(". Select the smaller loss.", "，选择损失较低的一个。"),
    };
  },
};

const groups = [[2, 1, 3], [4, 2], [6]];
const commandNames = ["HOLD", "GO", "CHECK"];
function countsForBatch(batch) {
  return commandNames.map((_, i) => (i === batch.command ? batch.n : 0));
}
const accumMission = {
  id: "schedule-updates",
  title: ["Send three optimizer updates", "安排三次参数更新"],
  brief: [
    "A device reads one microbatch at a time. Read the next microbatch to accumulate its targets, or update the weights using everything waiting. The queue contains 18 target positions in unequal batches.",
    "设备每次只读一个小批次。点“读下一批”可以暂存本批的训练信号；点“更新参数”才用上攒下来的所有目标。队列共有 18 个目标，各批大小不同。",
  ],
  target: [
    "Read the entire queue. Make exactly three weight updates of six target positions each.",
    "读完队列，恰好更新三次，每次更新都得用六个目标位置。",
  ],
  guide: "update-budget",
  start(seed) {
    const offset = index(seed, 3),
      order = [0, 1, 2].map((_, i) => groups[(i + offset) % 3]);
    const queue = order
      .flat()
      .map((n, i) => ({ n, command: (i + index(seed, 3)) % 3 }));
    return {
      queue,
      cursor: 0,
      waiting: [0, 0, 0],
      updates: [],
      logits: [0, 0, 0],
    };
  },
  play(s, id) {
    if (id === "read" && s.cursor < s.queue.length) {
      const c = countsForBatch(s.queue[s.cursor]);
      return {
        ...s,
        cursor: s.cursor + 1,
        waiting: s.waiting.map((n, i) => n + c[i]),
      };
    }
    const count = s.waiting.reduce((a, b) => a + b);
    if (id === "update" && count) {
      return {
        ...s,
        logits: scoreUpdate(s.logits, s.waiting, 1),
        updates: [...s.updates, count],
        waiting: [0, 0, 0],
      };
    }
    return s;
  },
  view(s, t) {
    const waiting = s.waiting.reduce((a, b) => a + b),
      batch = s.queue[s.cursor];
    return {
      stats: [
        stat(t("Targets waiting", "已攒目标数"), waiting),
        stat(
          t("Targets in each update", "历次更新的目标数"),
          s.updates.join(" · ") || "—",
        ),
        stat("P(HOLD)", pct(probabilities(s.logits)[0])),
      ],
      cards: [
        {
          id: "read",
          title: t("Read the next microbatch", "读下一批"),
          detail: batch
            ? batch.n + " × " + commandNames[batch.command]
            : t("Queue empty", "队列已空"),
          disabled: !batch,
        },
        {
          id: "update",
          title: t("Update the weights", "更新参数"),
          detail: waiting + t(" targets ready", " 个目标已攒好"),
          disabled: waiting === 0,
        },
      ],
      board: {
        columns: [
          t("Microbatch", "小批次"),
          t("Target × count", "目标 × 次数"),
          t("Status", "状态"),
        ],
        rows: s.queue.map((b, i) => [
          i + 1,
          commandNames[b.command] + " × " + b.n,
          i < s.cursor
            ? t("Read", "已读")
            : i === s.cursor
              ? t("Next", "下一批")
              : t("In queue", "待处理"),
        ]),
      },
      note: t(
        "Weights stay fixed while you read. At update time each target gets one vote in the average gradient.",
        "读小批次时权重保持原样。点更新时，每个目标都以相同权重参与平均梯度。",
      ),
    };
  },
  grade(s, t) {
    const wrong = s.updates.findIndex((n) => n !== 6);
    if (wrong >= 0)
      return {
        ok: false,
        message:
          t("Update ", "第 ") +
          (wrong + 1) +
          t(" used ", " 次更新用了 ") +
          s.updates[wrong] +
          t(
            " targets. Undo to place the boundary at six.",
            " 个目标。撤回几步，把这次更新放在正好攒齐六个的位置。",
          ),
      };
    if (s.updates.length !== 3)
      return {
        ok: false,
        message: t(
          "Keep reading and apply an update each time six targets have accumulated.",
          "继续读下一批，每攒齐六个目标就更新一次。",
        ),
      };
    return {
      ok: true,
      message: t(
        "Three updates consumed 6 + 6 + 6 targets. Different numbers of microbatches can make equally large optimizer updates.",
        "三次更新各消耗六个目标。小批次的数量可以不同，每次改动权重时用到的目标数仍然相同。",
      ),
    };
  },
};

const originalProbs = [0.75, 0.18, 0.07];
const replayMission = {
  id: "keep-old-skill",
  title: ["Keep an old prediction alive", "学新数据时留住旧能力"],
  brief: [
    "At this checkpoint the common log prefix predicts HOLD 75% of the time. You have four batches from a new source ending in GO and two older batches ending in HOLD. Each batch updates the same three scores immediately.",
    "这份参数看到相同的日志前文，有 75% 的概率接 HOLD。新来源给你四批 GO，旧来源还有两批 HOLD。每用一批，就立刻更新同样的三个分数。",
  ],
  target: [
    "Spend all six batches. Finish with P(HOLD) ≥ 31% and P(GO) ≥ 50%. Choose when to revisit the old source.",
    "六批全部用完。收尾时既要 P(HOLD) ≥ 31%，又要 P(GO) ≥ 50%。由你决定哪两步复习旧日志。",
  ],
  guide: "take-a-step",
  start(seed) {
    return {
      lr: [0.75, 1, 1.25][index(seed, 3)],
      logits: originalProbs.map(Math.log),
      history: [],
      left: [2, 4],
    };
  },
  play(s, id) {
    if (id !== "old" && id !== "new") return s;
    const i = id === "old" ? 0 : 1;
    if (!s.left[i]) return s;
    const z = scoreUpdate(s.logits, i === 0 ? [3, 0, 0] : [0, 3, 0], s.lr);
    return {
      ...s,
      logits: z,
      left: s.left.map((n, j) => n - Number(i === j)),
      history: [...s.history, { i, p: probabilities(z) }],
    };
  },
  view(s, t) {
    const p = probabilities(s.logits);
    return {
      stats: [
        stat("P(HOLD)", pct(p[0])),
        stat("P(GO)", pct(p[1])),
        stat(t("Learning rate", "学习率"), s.lr),
      ],
      cards: [
        {
          id: "old",
          title: t("Revisit an old batch", "复习一批旧日志"),
          detail: "3 × HOLD",
          tag: t("Remaining ", "剩余 ") + s.left[0],
          disabled: s.left[0] === 0,
        },
        {
          id: "new",
          title: t("Train on a new batch", "学习一批新日志"),
          detail: "3 × GO",
          tag: t("Remaining ", "剩余 ") + s.left[1],
          disabled: s.left[1] === 0,
        },
      ],
      board: {
        columns: [t("Update", "更新"), t("Target", "目标"), "P(HOLD)", "P(GO)"],
        rows: [
          [0, "—", pct(originalProbs[0]), pct(originalProbs[1])],
          ...s.history.map((h, i) => [
            i + 1,
            commandNames[h.i],
            pct(h.p[0]),
            pct(h.p[1]),
          ]),
        ],
      },
      note: t(
        "Each step starts from the previous step’s scores. Try spending both replay batches early or saving one for later.",
        "每步都从上一步留下的分数继续算。可以试试开头用光两批旧日志，也可以留一批在后面复习。",
      ),
    };
  },
  grade(s, t) {
    if (s.history.length < 6)
      return {
        ok: false,
        message: t(
          "Use every batch before testing the final checkpoint.",
          "先用完六批日志，再验收最后的参数。",
        ),
      };
    const p = probabilities(s.logits);
    const ok = p[0] >= 0.31 && p[1] >= 0.5;
    return {
      ok,
      message: ok
        ? t(
            "Both targets cleared their thresholds. The position of each replay update changed how much it survived the later GO updates.",
            "两个目标的概率都达标了。复习放在第几步，会影响后来的 GO 更新还能保留多少 HOLD 概率。",
          )
        : p[0] < 0.31
          ? t(
              "HOLD fell below 31%. Move a replay batch closer to the end and train again.",
              "HOLD 低于 31%。把一批复习日志往后挪，重试这个顺序。",
            )
          : t(
              "GO finished below 50%. Leave room for a GO batch after the last replay.",
              "GO 的概率还没到 50%。最后一次复习之后，留出机会再学一批 GO。",
            ),
    };
  },
};
export const mixMissions = [
  cargoMission,
  fleetMission,
  accumMission,
  replayMission,
];
