import { test } from "node:test";
import assert from "node:assert/strict";
import {
  missions,
  stations,
  freshDeal,
  readDeal,
  replay,
} from "../src/arcade-engine.mjs";

const english = (en) => en;
const chinese = (en, zh) => zh;
// These are real card clicks from one reproducible deal. Each losing path
// spends resources or changes the board before the player submits it.
const plays = {
  "coverage-plan": {
    win: ["A", "A", "A", "A", "B", "B", "B"],
    lose: ["A", "B", "B", "B", "B", "B", "B"],
  },
  "pack-windows": {
    win: ["log-0", "next", "log-1", "next"],
    lose: ["log-0", "same", "log-1", "next"],
  },
  "equalize-trials": {
    win: ["A6", "B6", "A6", "B6"],
    lose: ["A6", "A6", "B3", "B3", "B6"],
  },
  "paired-report": {
    win: ["run-4", "run-10", "report-B"],
    lose: ["run-4", "run-10", "report-A"],
  },
  "collect-three": {
    win: ["log-0", "log-1", "log-4"],
    lose: ["log-0", "log-1", "log-2"],
  },
  "reread-or-open": { win: ["A", "A", "B", "C"], lose: ["A", "A", "A", "A"] },
  "nine-target-race": { win: ["A", "C", "C"], lose: ["A", "A", "B"] },
  "checkpoint-patrol": {
    win: [
      "train",
      "train",
      "probe",
      "train",
      "train",
      "train",
      "train",
      "train",
      "train",
      "train",
      "train",
      "probe",
      "save-2",
    ],
    lose: [
      "train",
      "train",
      "probe",
      "train",
      "train",
      "train",
      "train",
      "train",
      "train",
      "train",
      "train",
      "probe",
      "save-10",
    ],
  },
  "copy-desk": {
    win: ["record-4", "record-5", "record-7"],
    lose: ["record-1"],
  },
  "rare-report-rescue": {
    win: ["record-2", "record-4"],
    lose: ["record-1", "record-6"],
  },
  "split-by-trip": {
    win: ["record-4", "record-5", "record-6", "record-7"],
    lose: ["record-3"],
  },
  "clean-scorecard": {
    win: ["record-3", "record-1", "record-5", "record-2"],
    lose: ["record-3", "record-1"],
  },
  "pack-data": {
    win: ["pack-1", "pack-2", "pack-4"],
    lose: ["pack-1", "pack-2", "pack-3"],
  },
  "dispatch-model": {
    win: ["road", "site", "model-0"],
    lose: ["road", "site", "model-1"],
  },
  "schedule-updates": {
    win: [
      "read",
      "read",
      "update",
      "read",
      "update",
      "read",
      "read",
      "read",
      "update",
    ],
    lose: ["read", "update"],
  },
  "keep-old-skill": {
    win: ["new", "new", "new", "old", "old", "new"],
    lose: ["old", "old", "new", "new", "new", "new"],
  },
};

function playCards(mission, actions, seed = 1) {
  let state = mission.start(seed);
  for (const id of actions) {
    const card = mission.view(state, english).cards.find((c) => c.id === id);
    assert.ok(card && !card.disabled, mission.id + " unexpected move " + id);
    const snapshot = structuredClone(state);
    const next = mission.play(state, id);
    assert.notEqual(next, state);
    assert.deepEqual(
      state,
      snapshot,
      "move changed the state of the preceding turn",
    );
    state = next;
  }
  return state;
}

test("every playable station is covered by a reproducible pass and fail path", () => {
  assert.deepEqual(Object.keys(plays).sort(), missions.map((m) => m.id).sort());
  assert.equal(
    new Set(stations.flatMap((s) => s.missions.map((m) => m.id))).size,
    missions.length,
  );
});

for (const mission of missions) {
  test(
    mission.id +
      " grades actual card sequences and replays saved moves in either language",
    () => {
      const { win, lose } = plays[mission.id];
      for (const [actions, shouldPass] of [
        [[], false],
        [lose, false],
        [win, true],
      ]) {
        const direct = playCards(mission, actions);
        for (const t of [english, chinese]) {
          const deal = readDeal(
            JSON.parse(
              JSON.stringify({ ...freshDeal(), actions, submitted: true }),
            ),
          );
          const game = replay(mission, deal, t);
          assert.deepEqual(
            game.moves.map((m) => m.id),
            actions,
          );
          assert.deepEqual(game.state, direct);
          assert.equal(mission.grade(game.state, t).ok, shouldPass);
          assert.ok(mission.grade(game.state, t).message.length > 0);
          for (const card of mission
            .view(game.state, t)
            .cards.filter((c) => c.disabled))
            assert.strictEqual(mission.play(game.state, card.id), game.state);
        }
        assert.notEqual(
          mission.grade(direct, english).message,
          mission.grade(direct, chinese).message,
        );
      }
      const previous = win.slice(0, -1);
      assert.deepEqual(
        replay(mission, { seed: 1, actions: previous }, chinese).state,
        playCards(mission, previous),
      );
      const restored = readDeal({
        seed: 1,
        actions: ["removed-card", ...win],
        submitted: true,
      });
      assert.deepEqual(
        replay(mission, restored, english).moves.map((m) => m.id),
        win,
      );
      const initial = mission.start(2);
      assert.strictEqual(mission.play(initial, "removed-card"), initial);
      assert.deepEqual(replay(mission, freshDeal(2), english).state, initial);
      assert.deepEqual(mission.start(2), initial);
      assert.ok(
        [2, 3, 4, 5, 6].some(
          (s) =>
            JSON.stringify(mission.start(s)) !==
            JSON.stringify(mission.start(1)),
        ),
        "new deals must vary",
      );
    },
  );
}

test("unusable saved deals restart and repeated one-use cards leave budgets unchanged", () => {
  for (const record of [
    null,
    {},
    { seed: -1, actions: [] },
    { seed: 1, actions: "A" },
  ])
    assert.deepEqual(readDeal(record), freshDeal());
  assert.deepEqual(
    readDeal({ seed: 5, actions: [2, "log-0"], submitted: "yes" }),
    { seed: 5, actions: ["log-0"], submitted: false },
  );
  const mission = missions.find((m) => m.id === "collect-three");
  const actions = [...plays[mission.id].win, "log-0", "log-5"];
  actions.splice(1, 0, "log-0");
  const { state, moves } = replay(mission, { seed: 1, actions }, english);
  assert.deepEqual(
    moves.map((x) => x.id),
    plays[mission.id].win,
  );
  assert.equal(new Set(state.taken).size, 3);
  assert.equal(state.taken.length, 3);
});

test("a clean scorecard retains its target mix and measures held-out loss on four new cues", () => {
  const mission = missions.find((m) => m.id === "clean-scorecard");
  const state = playCards(mission, plays[mission.id].win);
  assert.equal(state.slots.length, 4);
  const held = state.slots.map((id) => state.records.find((r) => r.id === id));
  assert.deepEqual(held.map((r) => r.target).sort(), [
    "GO",
    "GO",
    "HOLD",
    "HOLD",
  ]);
  assert.ok(
    held.every((r) => state.training.every((seen) => r.cue[0] !== seen.cue[0])),
  );
  const result = mission.view(state, english);
  assert.equal(
    result.stats.find((s) => s.label === "Mean test loss").value,
    Math.log(3).toFixed(3),
  );
});

test("rereading consumes target slots without discovering new records", () => {
  const m = missions.find((m) => m.id === "reread-or-open");
  const repeated = playCards(m, ["A", "A", "A", "A"]);
  const mixed = playCards(m, plays[m.id].win);
  for (const s of [repeated, mixed]) {
    const read = s.turns
      .flatMap((turn) => turn.counts)
      .reduce((sum, n) => sum + n, 0);
    assert.equal(read, 8);
    assert.equal(m.view(s, english).stats[0].value, "8 / 8");
    assert.ok(m.view(s, english).cards.every((c) => c.disabled));
  }
  assert.equal(m.view(repeated, english).stats[1].value, "2");
  assert.equal(m.view(mixed, english).stats[1].value, "6");
});

test("accumulating microbatches changes weights only when the player applies an update", () => {
  const m = missions.find((m) => m.id === "schedule-updates");
  const start = m.start(1);
  const loaded = m.play(start, "read");
  assert.deepEqual(loaded.logits, start.logits);
  assert.ok(loaded.waiting.reduce((a, b) => a + b) > 0);
  const afterUpdate = m.play(loaded, "update");
  assert.notDeepEqual(afterUpdate.logits, start.logits);
  assert.deepEqual(afterUpdate.waiting, [0, 0, 0]);
  const done = playCards(m, plays[m.id].win);
  assert.deepEqual(done.updates, [6, 6, 6]);
  assert.deepEqual(done.waiting, [0, 0, 0]);
});
