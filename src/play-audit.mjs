// The corpus in each deal is local to that mission. IDs on cards identify
// files; an incident ID can be shared by several reports of the same event.
const cues = {
  HOLD: [
    ["Red signal", "信号灯变红"],
    ["Barrier down", "道闸落下"],
    ["Crossing occupied", "路口有人"],
    ["Platform closed", "站台关闭"],
    ["Gate locked", "大门上锁"],
    ["Bridge raised", "吊桥升起"],
  ],
  GO: [
    ["Clear track", "轨道畅通"],
    ["Bridge open", "桥梁开通"],
    ["Crossing clear", "路口已清空"],
    ["Lane reopened", "车道重开"],
    ["Green signal", "信号灯变绿"],
    ["Gate open", "大门打开"],
  ],
  CHECK: [
    ["Wheel slip", "车轮打滑"],
    ["Axle vibration", "车轴振动"],
    ["Loose cable", "线缆松动"],
    ["Sensor mismatch", "传感器读数冲突"],
    ["Low pressure", "气压偏低"],
    ["Radio dropout", "无线电掉线"],
  ],
};

function shuffled(values, seed) {
  const order = [...values];
  let state = seed >>> 0;
  for (let i = order.length - 1; i > 0; i--) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const j = Math.floor((state / 2 ** 32) * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

function sampleCues(target, seed, count) {
  return shuffled(cues[target], seed)
    .slice(0, count)
    .map((cue) => ({ cue, target }));
}
const line = (record, t) => t(...record.cue) + " → " + record.target;
const stat = (label, value) => ({ label, value: String(value) });
const recordId = (i) => "record-" + (i + 1);
function withFileIds(records, seed) {
  return shuffled(records, seed).map((r, i) => ({ ...r, id: recordId(i) }));
}
function toggleRecord(s, id) {
  if (!s.records.some((r) => r.id === id)) return s;
  return {
    ...s,
    selected: s.selected.includes(id)
      ? s.selected.filter((k) => k !== id)
      : [...s.selected, id],
  };
}
const selectedRecords = (s) =>
  s.records.filter((r) => s.selected.includes(r.id));
const statNames = {
  kept: ["Files kept", "留下的文件"],
  distinct: ["Different lines kept", "留下的不同文本"],
  extra: ["Extra copies kept", "多留的副本"],
};

function copyCounts(s) {
  const kept = selectedRecords(s);
  const distinct = new Set(kept.map((r) => r.cue[0])).size;
  return { kept: kept.length, distinct, extra: kept.length - distinct };
}

const copies = {
  id: "copy-desk",
  title: ["Clear the copy desk", "清理重复来稿"],
  brief: [
    "Seven files arrived from the rover. Two files can contain the same entire line even when their file numbers differ. Click a file to set it aside or restore it.",
    "小车送来七份文件。文件编号不同，里面的整行文字也可能完全一样。点一份文件就能移出，再点一次放回。",
  ],
  target: [
    "Keep exactly one copy of each different line. The final folder should contain four files.",
    "每种完整文本保留一份，最后文件夹里应有四份文件。",
  ],
  guide: "exact-copies",
  start(seed) {
    const lines = [
      ...sampleCues("HOLD", seed, 2),
      ...sampleCues("GO", seed + 7, 1),
      ...sampleCues("CHECK", seed + 17, 1),
    ];
    const extras = shuffled(lines, seed + 31);
    const records = withFileIds(
      [...lines, extras[0], extras[0], extras[1]],
      seed + 43,
    );
    return { records, selected: records.map((r) => r.id) };
  },
  play: toggleRecord,
  view(s, t) {
    const counts = copyCounts(s);
    return {
      stats: Object.entries(counts).map(([key, n]) =>
        stat(t(...statNames[key]), n),
      ),
      cards: s.records.map((r, i) => ({
        id: r.id,
        title: line(r, t),
        detail: t("File ", "文件 ") + (i + 1),
        tag: s.selected.includes(r.id)
          ? t("In folder", "已保留")
          : t("Set aside", "已移出"),
      })),
    };
  },
  grade(s, t) {
    const { distinct, extra } = copyCounts(s);
    if (distinct < 4)
      return {
        ok: false,
        message: t(
          "A whole line has disappeared. Restore one file for each missing line.",
          "有一种文本被全部移出了。给每种缺失的文本放回一份文件。",
        ),
      };
    if (extra)
      return {
        ok: false,
        message: t(
          "All four lines are present. A repeated line still has extra copies.",
          "四种文本都在。还有文本在文件夹里留了多份。",
        ),
      };
    return {
      ok: true,
      message: t(
        "Four different lines survived. The three extra copies will no longer give their targets extra training weight.",
        "四种文本各留了一份。三份副本被移出，它们的目标词不会再因为复制而额外多算三次。",
      ),
    };
  },
};

function eventCodes(seed, n) {
  return shuffled(
    ["K17", "B36", "N42", "G58", "C63", "D78", "V82", "P94"],
    seed,
  ).slice(0, n);
}

const rescue = {
  id: "rare-report-rescue",
  title: ["Rescue the rare incidents", "找回被误删的稀有事件"],
  brief: [
    "A filter kept one CHECK report and quarantined the rest. Each incident has a dispatch number. A relay or summary can describe the same incident in different words.",
    "过滤器只留下了一条 CHECK 日志，其余都进了待审区。调度号对应一次实际事件；转发、简报虽然换了说法，也可能来自同一次事件。",
  ],
  target: [
    "Restore two reports so the archive covers all three incidents, with one file per incident. Click a restored file to send it back.",
    "恢复两份，使档案覆盖三次事件，每次事件各有一份文件。再点已恢复的文件可以撤回。",
  ],
  guide: "exact-copies",
  start(seed) {
    const events = eventCodes(seed, 3);
    const base = sampleCues("CHECK", seed + 5, 3);
    const sources = [
      ["Dispatch", "调度原报"],
      ["Relay", "转发"],
      ["Shift summary", "交班简报"],
    ];
    const variants = [
      [0, 1],
      [0, 2],
      [1, 0],
      [1, 1],
      [2, 0],
      [2, 2],
    ];
    const records = withFileIds(
      variants.map(([i, j]) => ({
        ...base[i],
        event: events[i],
        source: sources[j],
      })),
      seed + 73,
    );
    return {
      records,
      selected: [],
      archived: { ...base[0], event: events[0] },
    };
  },
  play(s, id) {
    if (s.selected.length === 2 && !s.selected.includes(id)) return s;
    return toggleRecord(s, id);
  },
  view(s, t) {
    const restored = selectedRecords(s);
    const total = new Set([s.archived.event, ...restored.map((r) => r.event)])
      .size;
    return {
      stats: [
        stat(t("Recovery slots used", "恢复名额"), s.selected.length + " / 2"),
        stat(t("Incidents covered", "事件覆盖"), total + " / 3"),
      ],
      note:
        t("Already in the archive · ", "档案里已经有这条 · ") +
        s.archived.event +
        " · " +
        line(s.archived, t),
      cards: s.records.map((r) => {
        const picked = s.selected.includes(r.id);
        return {
          id: r.id,
          title: r.event + " · " + line(r, t),
          detail: t(...r.source),
          tag: picked ? t("Restored", "已恢复") : t("In quarantine", "待审核"),
          disabled: !picked && restored.length === 2,
        };
      }),
    };
  },
  grade(s, t) {
    const restored = selectedRecords(s);
    const incidents = new Set([
      s.archived.event,
      ...restored.map((r) => r.event),
    ]);
    if (restored.length < 2)
      return {
        ok: false,
        message: t(
          "There is still room for a missing incident. Compare dispatch numbers against the archive.",
          "还有恢复名额。对照档案里的调度号，找一次尚未覆盖的事件。",
        ),
      };
    const ok = incidents.size === 3;
    return {
      ok,
      message: ok
        ? t(
            "All three files say CHECK, and each comes from a different incident. The rare target now has three sources of evidence.",
            "三份都写 CHECK，但各自来自不同事件。现在这个稀有目标有了三个来源。",
          )
        : t(
            "A dispatch number is represented twice. Return one of those reports and recover a missing incident.",
            "一个调度号出现了两次。退回其中一份，为遗漏的事件留个位置。",
          ),
    };
  },
};

function splitCounts(s) {
  return s.families.map((f) => {
    const rows = s.records.filter((r) => r.family === f);
    const testing = rows.filter((r) => s.selected.includes(r.id)).length;
    return { family: f, training: rows.length - testing, testing };
  });
}
const familySplit = {
  id: "split-by-trip",
  title: ["Keep each trip on one side", "把同一趟车次留在同一边"],
  brief: [
    "These eight logs come from four trips. Consecutive reports from one trip share the same journey. Click a log to move it between training and validation.",
    "八条日志来自四趟车。同一趟的连续报告共享这一路的情况。点一条日志可以在训练集和验证集之间移动。",
  ],
  target: [
    "Put three logs in validation, covering both HOLD and CHECK. Every trip must stay entirely on one side.",
    "验证集恰好放三条，HOLD 和 CHECK 都要有。同一趟车的日志只能出现在一边。",
  ],
  guide: "cross-split-overlap",
  start(seed) {
    const families = eventCodes(seed + 9, 4);
    const labels = [
      ["HOLD", "HOLD"],
      ["CHECK"],
      ["HOLD", "CHECK", "CHECK"],
      ["CHECK", "CHECK"],
    ];
    const records = withFileIds(
      labels.flatMap((tokens, i) =>
        tokens.map((target, k) => ({
          family: families[i],
          seq: k + 1,
          target,
        })),
      ),
      seed + 33,
    );
    // Start with two deliberately split families and a fully held-out singleton.
    return {
      families,
      records,
      selected: records
        .filter((r) => r.seq === 1 && r.family !== families[3])
        .map((r) => r.id),
    };
  },
  play: toggleRecord,
  view(s, t) {
    const byTrip = splitCounts(s);
    return {
      stats: [
        stat(t("Validation logs", "验证日志"), s.selected.length + " / 3"),
        stat(
          t("Trips on both sides", "跨集合的车次"),
          byTrip.filter((r) => r.training && r.testing).length,
        ),
      ],
      board: {
        columns: [
          t("Trip", "车次"),
          t("Train", "训练"),
          t("Validation", "验证"),
        ],
        rows: byTrip.map((r) => [r.family, r.training, r.testing]),
      },
      cards: s.records.map((r) => ({
        id: r.id,
        title: r.family + " · " + t("report ", "报告 ") + r.seq,
        detail: r.target,
        tag: s.selected.includes(r.id)
          ? t("Validation", "验证")
          : t("Training", "训练"),
      })),
    };
  },
  grade(s, t) {
    const mixed = splitCounts(s).find((r) => r.training && r.testing);
    if (mixed)
      return {
        ok: false,
        message:
          mixed.family +
          " · " +
          t(
            "This trip still has reports on both sides. Keep its reports together.",
            "这趟车还有日志同时分在两边。把同车次的报告放在一边。",
          ),
      };
    if (s.selected.length !== 3)
      return {
        ok: false,
        message: t(
          "The trips are separated. Move complete trips until validation contains three reports.",
          "车次已经分开。按整趟移动，凑出三条验证日志。",
        ),
      };
    if (new Set(selectedRecords(s).map((r) => r.target)).size !== 2)
      return {
        ok: false,
        message: t(
          "The validation reports currently cover one command. Choose trips that also include the other command.",
          "目前验证集只覆盖一个命令。重新选几趟车，把另一个命令也放进去。",
        ),
      };
    return {
      ok: true,
      message: t(
        "Both commands appear in validation. Every validation trip is new to the model, even when several reports describe the same ride.",
        "验证集覆盖两种命令。即使一趟车有好几条报告，验证车次也都没有进过训练集。",
      ),
    };
  },
};

function auditScore(s) {
  const test = s.slots.map((id) => s.records.find((r) => r.id === id));
  const holds = test.filter((r) => r.target === "HOLD").length;
  const leaked = test.filter((r) =>
    s.training.some((seen) => seen.cue[0] === r.cue[0]),
  ).length;
  // Frozen table assigns the observed command 0.8 for a known cue (0.1 + 0.1
  // to the others); the three possible commands are equiprobable on new cues.
  return {
    leaked,
    holds,
    meanLoss:
      -(leaked * Math.log(0.8) + (test.length - leaked) * Math.log(1 / 3)) /
      test.length,
  };
}
const leakage = {
  id: "clean-scorecard",
  title: ["Repair a suspicious scorecard", "修复一张虚高的成绩单"],
  brief: [
    "A frozen lookup model gives the correct command 80% on cues it saw in training and ⅓ on new cues. Compare the training index with the four test files. Select a test file, then a reserve file to swap them.",
    "这里用一张固定查找表模拟记忆：见过的提示给正确命令 80% 概率，新提示给三种命令各 ⅓。对照训练索引和四份考题，先点一道考题，再点一份备用题来交换。",
  ],
  target: [
    "Remove all training cues from the test while keeping two HOLD and two GO. Watch the mean loss after each swap.",
    "验证题里不能出现训练时见过的提示，同时保留两道 HOLD 和两道 GO。每换一道，看看平均损失怎么变。",
  ],
  guide: "clean-validation",
  start(seed) {
    const hold = sampleCues("HOLD", seed, 6),
      go = sampleCues("GO", seed + 7, 6);
    const records = withFileIds(
      [hold[0], go[0], hold[2], go[2], hold[3], go[3], hold[4], go[4]],
      seed + 333,
    );
    const initial = new Set([hold[0].cue, go[0].cue, hold[2].cue, go[2].cue]);
    return {
      records,
      training: [hold[0], go[0], hold[1], go[1]],
      slots: records.filter((r) => initial.has(r.cue)).map((r) => r.id),
      active: null,
    };
  },
  play(s, id) {
    if (!s.records.some((r) => r.id === id)) return s;
    if (s.slots.includes(id))
      return { ...s, active: s.active === id ? null : id };
    if (!s.active) return s;
    return {
      ...s,
      slots: s.slots.map((x) => (x === s.active ? id : x)),
      active: null,
    };
  },
  view(s, t) {
    const { leaked, holds, meanLoss } = auditScore(s);
    const ordered = [
      ...s.slots.map((id) => s.records.find((r) => r.id === id)),
      ...s.records.filter((r) => !s.slots.includes(r.id)),
    ];
    return {
      stats: [
        stat(t("Seen test cues", "见过的验证提示"), leaked + " / 4"),
        stat("HOLD / GO", holds + " / " + (4 - holds)),
        stat(t("Mean test loss", "验证平均损失"), meanLoss.toFixed(3)),
      ],
      board: {
        columns: [
          t("Training index · cue", "训练索引 · 提示"),
          t("Target", "目标"),
        ],
        rows: s.training.map((r) => [t(...r.cue), r.target]),
      },
      cards: ordered.map((r) => {
        const held = s.slots.includes(r.id);
        return {
          id: r.id,
          title: line(r, t),
          tag: held ? t("Test set", "验证集") : t("Reserve", "备用题"),
          disabled: !held && !s.active,
          detail:
            s.active === r.id
              ? t("Selected for replacement", "等候交换")
              : held
                ? t("Select to replace", "选择换出")
                : t("Swap in", "换入"),
        };
      }),
      note: s.active
        ? t(
            "Pick a reserve file. Both the before and after tests have four files.",
            "选一份备用题换入。交换前后都用四份文件评分。",
          )
        : t(
            "Loss averages −ln(probability of the correct command) over the four test files.",
            "四道题各算 −ln(正确命令的概率)，然后取平均。",
          ),
    };
  },
  grade(s, t) {
    const { leaked, holds, meanLoss } = auditScore(s);
    if (leaked)
      return {
        ok: false,
        message:
          t("Training cues left in validation", "验证里还混着训练提示") +
          " · " +
          leaked +
          t(
            ". Find those cues in the index and swap in fresh files.",
            " 条。对照训练索引找出来，换成新题。",
          ),
      };
    if (holds !== 2)
      return {
        ok: false,
        message: t(
          "The test is clean. Restore the original target mix of two HOLD and two GO.",
          "验证提示都已换新。再把目标比例调回两道 HOLD、两道 GO，以便比较。",
        ),
      };
    return {
      ok: true,
      message:
        t(
          "At the same target mix, loss rose from 0.661 to ",
          "保持命令比例相同，损失从 0.661 升到 ",
        ) +
        meanLoss.toFixed(3) +
        t(
          ". The model stayed frozen while you replaced test files.",
          "。整个过程中模型参数都没有动。",
        ),
    };
  },
};

export const auditMissions = [copies, rescue, familySplit, leakage];
