import { test } from "node:test";
import assert from "node:assert/strict";
import { probabilities, sampleActions, trainStep } from "../src/trainer.mjs";

const uniform = [0, 0, 0];
function near(x, y, tolerance = 1e-8) {
  assert.ok(Math.abs(x - y) < tolerance, `expected ${x} ≈ ${y}`);
}

test("softmax survives a common large offset and sampling follows a saved seed", () => {
  assert.deepEqual(probabilities([1000, 1001, 999]), probabilities([0, 1, -1]));
  const draws = sampleActions([Math.log(7), Math.log(2), 0], 8000, 2026);
  assert.deepEqual(
    draws,
    sampleActions([Math.log(7), Math.log(2), 0], 8000, 2026),
  );
  for (const [id, expected] of [0.7, 0.2, 0.1].entries()) {
    near(draws.filter((x) => x === id).length / draws.length, expected, 0.025);
  }
});

test("next-token data have an equilibrium at empirical frequencies and SFT selects one labeled response", () => {
  const logits = [Math.log(4), Math.log(2), 0];
  const data = { method: "pretrain", counts: [4, 2, 1] };
  const fit = trainStep(logits, data);
  fit.metrics.gradient.forEach((g) => near(g, 0));
  const sft = trainStep(uniform, { method: "sft", target: 1 });
  assert.ok(sft.after[1] > sft.before[1]);
  near(sft.after[1], Math.exp(1) / (2 + Math.exp(1)));
  assert.deepEqual(uniform, [0, 0, 0]);
});

test("DPO compares log odds relative to the frozen reference", () => {
  const cfg = {
    method: "dpo",
    chosen: 0,
    rejected: 1,
    beta: 2,
    reference: [4, 2, 3],
  };
  const initial = trainStep([4, 2, 3], cfg);
  near(initial.loss, Math.log(2));
  near(initial.metrics.preference, 0.5);
  near(initial.logits[0] - initial.logits[1], 4);
  const second = trainStep(initial.logits, cfg);
  assert.ok(second.loss < initial.loss);
  assert.ok(second.metrics.gradient[0] < initial.metrics.gradient[0]);
  near(second.logits[2], 3);
  assert.deepEqual(cfg.reference, [4, 2, 3]);
});

test("external reward and baseline determine which sampled action gains probability", () => {
  const cfg = { method: "rl", actions: [0], rewards: [0.5], baseline: 0 };
  const success = trainStep(uniform, cfg);
  const disappointment = trainStep(uniform, { ...cfg, baseline: 1 });
  assert.ok(success.after[0] > success.before[0]);
  assert.ok(disappointment.after[0] < disappointment.before[0]);
  const neutral = trainStep(uniform, { ...cfg, baseline: 0.5 });
  assert.deepEqual(neutral.before, neutral.after);
});

test("GRPO computes sample-relative advantages and uses the rollout policy in each ratio", () => {
  const config = {
    method: "grpo",
    actions: [0, 1, 2],
    rewards: [1, 0, -1],
    old: uniform,
    clip: 0.2,
  };
  const first = trainStep(uniform, config);
  assert.deepEqual(first.metrics.ratios, [1, 1, 1]);
  near(first.metrics.rewardStd, 1);
  near(first.metrics.advantages[0], 1 / 1.000001);
  assert.ok(first.after[0] > first.before[0]);
  assert.ok(first.after[2] < first.before[2]);
  // Positive advantage already above the upper bound and negative already
  // below the lower bound: both surrogate terms have a flat gradient.
  const clipped = trainStep([3, 0, -3], config);
  assert.deepEqual(clipped.metrics.clipped, [true, false, true]);
  assert.deepEqual(clipped.before, clipped.after);
  // Out-of-range ratios in a loss-worsening direction still have a gradient.
  const reverse = trainStep([-3, 0, 3], config);
  assert.deepEqual(reverse.metrics.clipped, [false, false, false]);
  assert.ok(reverse.after[2] < reverse.before[2]);
});

test("a flat reward group has no policy gradient, and an independent KL term can still pull back", () => {
  const config = {
    method: "grpo",
    actions: [0, 1, 0],
    rewards: [0.1, 0.1, 0.1],
    old: uniform,
    reference: uniform,
  };
  const logits = [1.2, 0.1, -0.4];
  const flat = trainStep(logits, config);
  assert.deepEqual(flat.metrics.advantages, [0, 0, 0]);
  assert.deepEqual(flat.before, flat.after);
  const regularized = trainStep(logits, { ...config, kl: 0.5 });
  const next = trainStep(regularized.logits, config);
  assert.ok(next.metrics.kl < flat.metrics.kl);
});

test("every update direction matches a central finite difference of the displayed loss", () => {
  const configurations = [
    { method: "pretrain", counts: [0, 2, 1] },
    { method: "sft", target: 1 },
    {
      method: "dpo",
      chosen: 0,
      rejected: 2,
      reference: [-0.2, 0.1, 0.3],
      beta: 0.6,
    },
    {
      method: "rl",
      actions: [2, 1, 1, 0],
      rewards: [0, 1, 1, 0.5],
      baseline: 0.3,
    },
    {
      method: "grpo",
      actions: [2, 1, 1, 0],
      rewards: [0, 1, 1, 0.5],
      old: [0.1, -0.25, 0.15],
      reference: [-0.2, 0.1, 0.3],
      kl: 0.4,
    },
  ];
  const logits = [0.1, -0.2, 0.14],
    step = 1e-5;
  for (const cfg of configurations) {
    const { metrics } = trainStep(logits, cfg);
    for (const [i, gradient] of metrics.gradient.entries()) {
      const shiftedLoss = (offset) =>
        trainStep(
          logits.map((x, j) => x + (i === j ? offset : 0)),
          cfg,
        ).loss;
      near(
        gradient,
        -(shiftedLoss(step) - shiftedLoss(-step)) / (2 * step),
        1e-7,
      );
    }
  }
});

test("malformed labels and empty training data produce explicit errors", () => {
  assert.throws(
    () => trainStep(uniform, { method: "pretrain", counts: [0, 0, 0] }),
    /next token/,
  );
  assert.throws(
    () => trainStep(uniform, { method: "sft", target: 4 }),
    /outside/,
  );
  assert.throws(
    () => trainStep(uniform, { method: "grpo", actions: [0], rewards: [1] }),
    /two responses/,
  );
  assert.throws(
    () =>
      trainStep(uniform, {
        method: "grpo",
        actions: [0, 1],
        rewards: [1, 2],
        old: uniform,
        kl: 1,
      }),
    /reference/,
  );
});
