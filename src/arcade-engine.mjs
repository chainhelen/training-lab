import { budgetMissions } from "./play-budget.mjs";
import { auditMissions } from "./play-audit.mjs";
import { mixMissions } from "./play-mix.mjs";
import { trialMissions } from "./play-trials.mjs";

export const stations = [
  {
    id: "budget",
    name: ["Data expedition", "采集训练数据"],
    brief: [
      "Spend a limited budget on records, updates and validation checks.",
      "分配有限的预算，决定读哪些日志、何时更新、何时验证。",
    ],
    missions: budgetMissions,
  },
  {
    id: "audit",
    name: ["Corpus inspection", "语料质检"],
    brief: [
      "Clean a dataset and build an evaluation split you can trust.",
      "清理重复资料，为模型准备一份真正没见过的考卷。",
    ],
    missions: auditMissions,
  },
  {
    id: "mix",
    name: ["Training control room", "训练调度室"],
    brief: [
      "Choose the mix of data, schedule updates and compare the outcomes.",
      "给不同来源分配训练量，安排更新，再检查结果。",
    ],
    missions: mixMissions,
  },
  {
    id: "trials",
    name: ["Experiment design", "实验复核台"],
    brief: [
      "Pack prediction windows, match A/B budgets, and check a result across runs.",
      "拼预测窗口、对齐两组预算，再换几批数据检验结论。",
    ],
    missions: trialMissions,
  },
];
export const missions = stations.flatMap((s) => s.missions);
export const missionUrl = (m) => "#play/" + m.id;
export function freshDeal(seed = 1) {
  return { seed, actions: [], submitted: false };
}
export function readDeal(raw) {
  if (
    !raw ||
    !Number.isSafeInteger(raw.seed) ||
    raw.seed < 1 ||
    !Array.isArray(raw.actions)
  )
    return freshDeal();
  return {
    seed: raw.seed,
    actions: raw.actions.filter((a) => typeof a === "string"),
    submitted: raw.submitted === true,
  };
}
// Rebuild the board and translated move log from the same initial deal.
// Replaying public action IDs also lets older saves ignore retired cards.
export function replay(mission, deal, t) {
  let state = mission.start(deal.seed);
  const moves = [];
  for (const id of deal.actions) {
    const card = mission
      .view(state, t)
      .cards.find((c) => c.id === id && !c.disabled);
    if (!card) continue;
    const next = mission.play(state, id);
    if (next === state) continue;
    moves.push({ id, title: card.title });
    state = next;
  }
  return { state, moves };
}
