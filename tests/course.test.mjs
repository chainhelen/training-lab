import { test } from "node:test";
import assert from "node:assert/strict";
import {
  courses,
  valuesFor,
  lessonKey,
  lessonUrl,
  resolveRoute,
} from "../src/course-data.mjs";

const languages = [(en) => en, (en, zh) => zh];

function settings(controls) {
  let cases = [{}];
  for (const c of controls) {
    const choices = c.options
      ? c.options.map((_, i) => i)
      : Array.from(
          { length: Math.round((c.max - c.min) / (c.step ?? 1)) + 1 },
          (_, i) => Number((c.min + i * (c.step ?? 1)).toFixed(8)),
        );
    cases = cases.flatMap((v) => choices.map((n) => ({ ...v, [c.key]: n })));
  }
  return cases;
}

for (const course of courses) {
  test(
    course.id + " exercises can be solved and replayed in both languages",
    () => {
      const ids = course.steps.map((s) => s.id);
      assert.equal(new Set(ids).size, ids.length);
      for (const step of course.steps) {
        assert.deepEqual(resolveRoute(lessonUrl(course, step).slice(1), {}), {
          course,
          step,
        });
        const initial = valuesFor(step);
        assert.equal(step.check(initial), false, step.id);
        let solvable = false;
        for (const v of settings(step.controls)) {
          assert.deepEqual(valuesFor(step, v), v, step.id);
          solvable ||= step.check(v);
          for (const t of languages) {
            const { rows, formula } = step.view(v, t);
            assert.ok(rows.length && formula.length, step.id);
            for (const { label, value, fraction } of rows) {
              assert.ok(label.length && String(value).length, step.id);
              assert.doesNotMatch(
                String(value),
                /NaN|undefined|Infinity/,
                step.id,
              );
              if (fraction !== undefined)
                assert.ok(fraction >= 0 && fraction <= 1, step.id);
            }
            assert.ok(step.feedback(v, t).length, step.id);
          }
        }
        assert.ok(solvable, step.id);
      }
    },
  );
}

test("chapter links resume known steps and invalid stored inputs return to defaults", () => {
  const course = courses[0];
  const step = course.steps[1];
  const progress = { positions: { [course.id]: step.id } };
  assert.equal(resolveRoute("learn/" + course.id, progress).step, step);
  assert.equal(
    resolveRoute("learn/" + course.id + "/missing", progress).step,
    step,
  );
  assert.equal(lessonKey(course, step), [course.id, step.id].join("/"));
  const c = step.controls[0];
  for (const invalid of [
    undefined,
    null,
    "1",
    Infinity,
    c.min - 1,
    c.max + 1,
    c.min + 0.5,
  ]) {
    assert.deepEqual(valuesFor(step, { [c.key]: invalid }), valuesFor(step));
  }
});
