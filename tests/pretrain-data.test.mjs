import { test } from "node:test";
import assert from "node:assert/strict";
import { budgetSteps } from "../src/pretrain-budget.mjs";
import { scarcitySteps } from "../src/pretrain-scarcity.mjs";
import { qualitySteps } from "../src/pretrain-quality.mjs";
import { seedComparison } from "../src/pretrain-replicates.mjs";
import { probabilities, trainStep } from "../src/trainer.mjs";

const getRows = (steps, id, controls) =>
  steps.find((step) => step.id === id).view(controls, (en) => en).rows;
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const fixed = (x) => x.toFixed(3);
function meanLoss(logits, counts) {
  const p = probabilities(logits);
  return -sum(counts.map((n, i) => n * Math.log(p[i]))) / sum(counts);
}
function update(logits, counts, lr) {
  return trainStep(logits, { method: "pretrain", counts, lr }).logits;
}

test("accumulation changes updates while consuming all 96 predictions", () => {
  for (let accum = 0; accum < 4; accum++) {
    const [batches, targetsPerUpdate, updates] = getRows(
      budgetSteps,
      "update-budget",
      { accum },
    ).map((r) => Number(r.value));
    assert.equal(batches, 8);
    assert.equal(targetsPerUpdate, 12 * 2 ** accum);
    assert.equal(targetsPerUpdate * updates, 96);
  }
});

test("rare-target chance agrees with enumerating every subset without replacement", () => {
  const all = Array(13).fill(0),
    missed = Array(13).fill(0);
  // Index 5 and 11 are the two CHECK records in the twelve-record archive.
  for (let mask = 1; mask < 2 ** 12; mask++) {
    const size = mask.toString(2).replaceAll("0", "").length;
    all[size]++;
    if (!(mask & ((1 << 5) | (1 << 11)))) missed[size]++;
  }
  let firstBelowThreshold;
  for (let draws = 1; draws <= 12; draws++) {
    const rows = getRows(scarcitySteps, "miss-rare-target", { draws });
    assert.equal(Number(rows[0].value), all[draws]);
    assert.equal(Number(rows[1].value), missed[draws]);
    assert.equal(rows[2].fraction, missed[draws] / all[draws]);
    if (firstBelowThreshold === undefined && rows[2].fraction < 0.1)
      firstBelowThreshold = draws;
  }
  const check = scarcitySteps.find((s) => s.id === "miss-rare-target").check;
  for (let draws = 1; draws <= 12; draws++)
    assert.equal(check({ draws }), draws === firstBelowThreshold);
});

test("fixed-budget trials use the same held-out records and the advertised batch order", () => {
  const holdHeavy = [2, 1, 0],
    mixed = [1, 1, 1],
    validation = [3, 2, 1];
  let small = [0, 0, 0],
    large = [0, 0, 0];
  for (let updates = 0; updates <= 4; updates++) {
    const rows = getRows(scarcitySteps, "fixed-read-budget", { updates });
    assert.equal(Number(rows[0].value), updates * 3);
    for (const [row, logits, trainCounts] of [
      [rows[1], small, holdHeavy],
      [rows[2], large, validation],
    ]) {
      assert.deepEqual(
        String(row.value).split(" / "),
        [meanLoss(logits, trainCounts), meanLoss(logits, validation)].map(
          fixed,
        ),
      );
    }
    if (updates < 4) {
      small = update(small, holdHeavy, 1);
      large = update(large, updates % 2 === 0 ? holdHeavy : mixed, 1);
    }
  }
  assert.notDeepEqual(small, large);
  const same = getRows(scarcitySteps, "matching-batches", { subset: 1 });
  assert.equal(same[2].value, same[3].value);
  assert.equal(same[2].value, large.map(fixed).join(" / "));
  const different = getRows(scarcitySteps, "matching-batches", { subset: 0 });
  assert.equal(different[2].value, small.map(fixed).join(" / "));
  assert.equal(different[3].value, large.map(fixed).join(" / "));
});

test("cleaning validation replaces only copied prefixes at fixed size and target mix", () => {
  for (let replaced = 0; replaced <= 2; replaced++) {
    const rows = getRows(qualitySteps, "clean-validation", { replaced });
    const copied = 2 - replaced;
    assert.equal(Number(rows[0].value), copied);
    assert.equal(rows[1].value, "HOLD × 2, GO × 2");
    const expected =
      -(copied * Math.log(0.8) + (4 - copied) * Math.log(1 / 3)) / 4;
    assert.equal(rows[2].value, fixed(expected));
  }
});

test("changing the reporting mix reweights the mean with both domain losses fixed", () => {
  const snapshot = update([0, 0, 0], [2, 6, 0], 1);
  const road = meanLoss(snapshot, [0, 1, 0]),
    site = meanLoss(snapshot, [1, 0, 0]);
  for (let sites = 1; sites <= 7; sites++) {
    const rows = getRows(qualitySteps, "domain-report", { sites });
    assert.deepEqual(
      rows.map((r) => r.value),
      [road, site, ((8 - sites) * road + sites * site) / 8].map(fixed),
    );
  }
});

test("paired subsamples retain prior examples and share one evaluation set across twelve runs", () => {
  for (let size = 0; size < 4; size++) {
    const { results, misses, low, high, median } = seedComparison({
      size,
      run: 1,
    });
    assert.equal(results.length, 12);
    const losses = results.map((r) => r.validationLoss).sort((a, b) => a - b);
    assert.equal(low, losses[0]);
    assert.equal(high, losses[11]);
    assert.equal(median, (losses[5] + losses[6]) / 2);
    assert.equal(
      misses,
      results.filter((r) => r.trainingCounts[2] === 0).length,
    );
    for (let run = 1; run <= 12; run++) {
      const { selected } = seedComparison({ size, run });
      assert.deepEqual(selected, results[run - 1]);
      assert.equal(sum(selected.trainingCounts), 3 * 2 ** size);
      if (size > 0) {
        const previous = seedComparison({ size: size - 1, run }).selected
          .trainingCounts;
        assert.ok(selected.trainingCounts.every((n, i) => n >= previous[i]));
      } else {
        // Three samples exactly fill a batch, so all sixteen updates see these counts.
        let logits = [0, 0, 0];
        for (let step = 0; step < 16; step++)
          logits = update(logits, selected.trainingCounts, 0.5);
        assert.ok(
          Math.abs(meanLoss(logits, [12, 9, 3]) - selected.validationLoss) <
            1e-12,
        );
        assert.ok(
          Math.abs(
            meanLoss(logits, selected.trainingCounts) - selected.trainLoss,
          ) < 1e-12,
        );
      }
    }
  }
});
