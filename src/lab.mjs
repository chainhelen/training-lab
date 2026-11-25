import { probabilities, trainStep, sampleActions } from "./trainer.mjs";

export const ids = ["pretrain", "sft", "dpo", "rl", "grpo"];
export const tokens = ["HOLD", "GO", "CHECK"];
const fallbacks = {
  pretrain: [0, 0, 0],
  sft: [0.55, 0, -0.55],
  dpo: [0.7, 0, -0.3],
  rl: [0.7, 0, -0.3],
  grpo: [0.7, 0, -0.3],
};
export const stochastic = (id) => id === "rl" || id === "grpo";

export function start(id, checkpoints = {}) {
  const parent = id === "pretrain" ? null : id === "sft" ? "pretrain" : "sft";
  const inherited = parent && checkpoints[parent];
  const logits = [...(inherited || fallbacks[id])];
  const state = {
    id,
    parent: inherited ? parent : null,
    logits,
    reference: [...logits],
    counts: [0, 0, 0],
    target: 1,
    chosen: 1,
    rejected: 0,
    rule: id === "rl" ? "short" : "flat",
    kl: 0,
    seed: 31,
    step: 0,
    trace: [probabilities(logits)[0]],
    last: null,
    observed: false,
    won: false,
    used: false,
  };
  if (stochastic(id)) draw(state);
  return state;
}
export function draw(s) {
  s.seed = (s.seed + 1) >>> 0;
  s.old = [...s.logits];
  s.actions = sampleActions(s.old, 8, s.seed);
  s.used = false;
}
export function config(s) {
  const method = s.id;
  if (method === "pretrain") return { method, counts: s.counts, lr: 1 };
  if (method === "sft") return { method, target: s.target, lr: 0.8 };
  if (method === "dpo")
    return {
      method,
      chosen: s.chosen,
      rejected: s.rejected,
      reference: s.reference,
      beta: 1,
      lr: 1,
    };
  const rewards = s.actions.map((a) =>
    s.rule === "flat" ? 1 : Number(a === (s.rule === "short" ? 1 : 0)),
  );
  return {
    method,
    actions: s.actions,
    rewards,
    baseline: 0,
    old: s.old,
    reference: s.reference,
    clip: 0.2,
    kl: s.kl,
    lr: 0.8,
  };
}
export function preview(s) {
  if (s.id === "pretrain" && s.counts.every((n) => n === 0))
    return { error: "empty" };
  if (s.id === "dpo" && s.chosen === s.rejected) return { error: "duplicate" };
  return trainStep(s.logits, config(s));
}
export function update(s) {
  if (s.id === "rl" && s.used) return "draw";
  const batch = config(s),
    r = preview(s);
  if (r.error) return r.error;
  const before = s.logits;
  s.logits = r.logits;
  s.step++;
  s.trace.push(r.after[0]);
  if (s.trace.length > 33) s.trace.shift();
  s.last = {
    before: r.before,
    after: r.after,
    loss: r.loss,
    logitsBefore: [...before],
    lr: batch.lr,
    metrics: r.metrics,
    rule: s.rule,
  };
  s.used = stochastic(s.id);
  const inc = (i) => r.after[i] > r.before[i] + 1e-8;
  let pass = false,
    status = "updated";
  switch (s.id) {
    case "pretrain":
      pass =
        s.counts.join(",") === "2,1,0" &&
        r.after[0] >= 0.45 &&
        r.after[1] > r.after[2];
      if (s.counts.join(",") !== "2,1,0") status = "mix";
      break;
    case "sft":
      pass = s.target === 0 && r.after[0] >= 0.6;
      if (s.target !== 0) status = "demo";
      break;
    case "dpo": {
      const shift =
        s.logits[0] - s.logits[1] - (s.reference[0] - s.reference[1]);
      pass = s.chosen === 0 && s.rejected === 1 && shift >= Math.log(4.3);
      if (s.chosen !== 0 || s.rejected !== 1) status = "pair";
      break;
    }
    case "rl":
      if (s.rule === "short" && inc(1)) {
        s.observed = true;
        status = "shortSeen";
      }
      if (s.rule === "exact" && s.observed && inc(0)) pass = true;
      if (batch.rewards.every((r) => r === 0)) status = "noReward";
      if (s.rule === "exact" && !s.observed) status = "needShort";
      break;
    case "grpo":
      if (
        s.rule === "flat" &&
        s.kl === 0 &&
        before.every((v, i) => v === s.logits[i])
      ) {
        s.observed = true;
        status = "flatSeen";
      }
      if (s.rule === "exact") {
        if (!s.observed) status = "needFlat";
        else if (r.metrics.rewardStd === 0) status = "homogeneous";
        else if (inc(0)) pass = true;
        else status = "redraw";
      }
      break;
  }
  s.won ||= pass;
  return pass ? "pass" : status;
}
