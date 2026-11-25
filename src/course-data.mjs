import { pretrainSteps } from "./guide-pretrain.mjs";
import { sftSteps, dpoSteps } from "./guide-preference.mjs";
import { rlSteps, grpoSteps } from "./guide-reinforcement.mjs";

export const courses = [
  { id: "pretrain", name: ["Pretraining", "预训练"], steps: pretrainSteps },
  { id: "sft", name: ["SFT", "SFT"], steps: sftSteps },
  { id: "dpo", name: ["DPO", "DPO"], steps: dpoSteps },
  { id: "rl", name: ["RL · REINFORCE", "RL · REINFORCE"], steps: rlSteps },
  { id: "grpo", name: ["GRPO", "GRPO"], steps: grpoSteps },
];
export function valuesFor(step, stored = {}) {
  const values = {};
  for (const control of step.controls) {
    const min = control.options ? 0 : control.min,
      max = control.options ? control.options.length - 1 : control.max;
    const increment = control.step ?? 1;
    const raw = stored?.[control.key];
    const valid =
      Number.isFinite(raw) &&
      raw >= min &&
      raw <= max &&
      Math.abs((raw - min) / increment - Math.round((raw - min) / increment)) <
        1e-6;
    values[control.key] = valid ? raw : control.initial;
  }
  return values;
}
export const lessonKey = (course, step) => course.id + "/" + step.id;
export const lessonUrl = (course, step) => "#learn/" + lessonKey(course, step);
export function resolveRoute(route, save) {
  const [, courseId, stepId] = route.split("/");
  const course = courses.find((c) => c.id === courseId) ?? courses[0];
  const step =
    course.steps.find((s) => s.id === stepId) ??
    course.steps.find((s) => s.id === save?.positions?.[course.id]) ??
    course.steps[0];
  return { course, step };
}
