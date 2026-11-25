// One logit row belongs to one context. Each answer in the post-training
// exercises is a single categorical action. All logarithms are natural.
function requireValue(condition, message) {
  if (!condition) throw new RangeError(message);
}

function finiteArray(values, name, length) {
  requireValue(
    Array.isArray(values) && values.length > 0,
    name + " must be a nonempty array",
  );
  requireValue(
    length === undefined || values.length === length,
    name + " has the wrong length",
  );
  for (const value of values)
    requireValue(Number.isFinite(value), name + " contains a nonfinite value");
}

function logProbabilities(logits) {
  finiteArray(logits, "logits");
  const largest = Math.max(...logits);
  const centered = logits.map((x) => x - largest);
  const logTotal = Math.log(centered.reduce((sum, x) => sum + Math.exp(x), 0));
  const result = centered.map((x) => x - logTotal);
  finiteArray(result, "log probabilities");
  return result;
}

export function probabilities(logits) {
  return logProbabilities(logits).map(Math.exp);
}

// A seeded uniform draw followed by the inverse categorical CDF lets players
// replay a rollout. Repeated action IDs represent separate sampled answers.
export function sampleActions(logits, count, seed = 1) {
  requireValue(
    Number.isSafeInteger(count) && count > 0,
    "sample count must be a positive integer",
  );
  requireValue(Number.isSafeInteger(seed), "seed must be an integer");
  const probs = probabilities(logits);
  let state = seed >>> 0;
  return Array.from({ length: count }, () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let bits = Math.imul(state ^ (state >>> 15), state | 1);
    bits ^= bits + Math.imul(bits ^ (bits >>> 7), bits | 61);
    let remainder = ((bits ^ (bits >>> 14)) >>> 0) / 4294967296;
    for (let i = 0; i < probs.length - 1; i++) {
      if (remainder < probs[i]) return i;
      remainder -= probs[i];
    }
    return probs.length - 1;
  });
}

function actionIndex(action, width) {
  requireValue(
    Number.isInteger(action) && action >= 0 && action < width,
    "action is outside this row",
  );
}

function crossEntropy(logProbs, probs, config) {
  let target;
  let total = 1;
  if (config.method === "sft") {
    actionIndex(config.target, probs.length);
    target = probs.map((_, i) => Number(i === config.target));
  } else {
    finiteArray(config.counts, "next-token counts", probs.length);
    requireValue(
      config.counts.every((v) => Number.isSafeInteger(v) && v >= 0),
      "counts must be nonnegative integers",
    );
    total = config.counts.reduce((sum, x) => sum + x, 0);
    requireValue(
      Number.isSafeInteger(total) && total > 0,
      "the corpus needs at least one next token",
    );
    target = config.counts.map((count) => count / total);
  }
  const loss = -target.reduce(
    (sum, frequency, i) => sum + frequency * logProbs[i],
    0,
  );
  const gradient = target.map((frequency, i) => frequency - probs[i]);
  return { loss, metrics: { gradient, target, count: total } };
}

function directPreference(logits, config) {
  const { chosen, rejected, reference, beta = 1 } = config;
  actionIndex(chosen, logits.length);
  actionIndex(rejected, logits.length);
  requireValue(
    chosen !== rejected,
    "the preference needs two different answers",
  );
  finiteArray(reference, "fixed reference logits", logits.length);
  requireValue(Number.isFinite(beta) && beta > 0, "beta must be positive");

  // The log normalizer cancels in log p(chosen) - log p(rejected).
  const margin =
    beta *
    (logits[chosen] -
      logits[rejected] -
      (reference[chosen] - reference[rejected]));
  const loss = Math.max(0, -margin) + Math.log1p(Math.exp(-Math.abs(margin)));
  const weight = beta / (1 + Math.exp(margin));
  const gradient = logits.map(() => 0);
  gradient[chosen] = weight;
  gradient[rejected] = -weight;
  const preference = 1 / (1 + Math.exp(-margin));
  return { loss, metrics: { gradient, preference, margin } };
}

function policyUpdate(logProbs, probs, config) {
  const { actions, rewards } = config;
  finiteArray(actions, "sampled actions");
  for (const id of actions) actionIndex(id, probs.length);
  finiteArray(rewards, "sampled rewards", actions.length);
  const count = actions.length;
  const meanReward = rewards.reduce((sum, r) => sum + r / count, 0);
  const grpo = config.method === "grpo";
  const { baseline = 0, clip = 0.2, kl: klWeight = 0 } = config;
  let rewardStd = 0;
  let oldLogProbs;
  let advantages;
  if (grpo) {
    requireValue(
      count >= 2,
      "group-relative training needs at least two responses to the same prompt",
    );
    requireValue(
      Number.isFinite(clip) && clip > 0 && clip < 1,
      "clip must be between zero and one",
    );
    requireValue(
      Number.isFinite(klWeight) && klWeight >= 0,
      "KL weight must be nonnegative",
    );
    finiteArray(config.old, "rollout policy logits", probs.length);
    oldLogProbs = logProbabilities(config.old);
    // Use the sample standard deviation, as in verl's group estimator.
    const centered = rewards.map((r) => r - meanReward);
    if (!rewards.every((r) => r === rewards[0])) {
      rewardStd = Math.sqrt(
        centered.reduce((sum, x) => sum + x * x, 0) / (count - 1),
      );
    }
    advantages = centered.map((value) =>
      rewardStd === 0 ? 0 : value / (rewardStd + 1e-6),
    );
  } else {
    requireValue(Number.isFinite(baseline), "reward baseline must be finite");
    advantages = rewards.map((reward) => reward - baseline);
  }

  const gradient = probs.map(() => 0);
  const ratios = [],
    clipped = [];
  let loss = 0;
  for (let i = 0; i < count; i++) {
    const action = actions[i],
      advantage = advantages[i];
    let coefficient = advantage;
    if (grpo) {
      const ratio = Math.exp(logProbs[action] - oldLogProbs[action]);
      requireValue(
        Number.isFinite(ratio),
        "policy ratio exceeds numerical range",
      );
      const bounded = Math.max(1 - clip, Math.min(ratio, 1 + clip));
      const saturated =
        (advantage > 0 && ratio > 1 + clip) ||
        (advantage < 0 && ratio < 1 - clip);
      loss -= Math.min(ratio * advantage, bounded * advantage) / count;
      coefficient = saturated ? 0 : ratio * advantage;
      ratios.push(ratio);
      clipped.push(saturated);
    } else {
      loss -= (advantage * logProbs[action]) / count;
    }
    // Gradient of log π(action) with respect to each trainable logit.
    for (let k = 0; k < probs.length; k++) {
      gradient[k] += (coefficient * (Number(k === action) - probs[k])) / count;
    }
  }

  let kl = 0;
  if (grpo && (config.reference !== undefined || klWeight > 0)) {
    finiteArray(config.reference, "fixed reference logits", probs.length);
    const reference = logProbabilities(config.reference);
    const logRatio = logProbs.map((logP, i) => logP - reference[i]);
    // The tiny action space lets us evaluate KL(current || reference) exactly.
    kl = probs.reduce((sum, p, i) => sum + p * logRatio[i], 0);
    loss += klWeight * kl;
    for (let i = 0; i < probs.length; i++) {
      gradient[i] -= klWeight * probs[i] * (logRatio[i] - kl);
    }
  }
  return {
    loss,
    metrics: {
      gradient,
      advantages,
      meanReward,
      rewardStd,
      ratios,
      clipped,
      kl,
    },
  };
}

// A single SGD step. The reported loss and diagnostics describe the policy
// before the step. "gradient" is the direction of increasing the objective.
export function trainStep(logits, config) {
  const logProbs = logProbabilities(logits);
  const before = logProbs.map(Math.exp);
  const { method, lr = 1 } = config;
  requireValue(
    Number.isFinite(lr) && lr >= 0,
    "learning rate must be nonnegative",
  );
  let update;
  switch (method) {
    case "pretrain":
    case "sft":
      update = crossEntropy(logProbs, before, config);
      break;
    case "dpo":
      update = directPreference(logits, config);
      break;
    case "rl":
    case "grpo":
      update = policyUpdate(logProbs, before, config);
      break;
    default:
      throw new RangeError("unknown training method: " + method);
  }
  requireValue(Number.isFinite(update.loss), "loss exceeds numerical range");
  const next = logits.map(
    (logit, i) => logit + lr * update.metrics.gradient[i],
  );
  return { logits: next, before, after: probabilities(next), ...update };
}
