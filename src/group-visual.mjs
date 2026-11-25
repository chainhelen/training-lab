import {
  tokens,
  group,
  firstFour,
  exact,
  shortest,
  grpo,
  policyAt,
} from "./guide-reinforcement.mjs";

const signed = (n) => (n > 0 ? "+" : "") + n.toFixed(2);

function figure({ title, actions, rewards, detail, selected, legend }) {
  const cards = actions
    .map((action, i) => {
      const reward = rewards[i];
      const selectedAttribute = selected === i ? ' aria-current="true"' : "";
      const detailLine = detail
        ? `<span>${detail.symbol} ${detail.values[i]}</span>`
        : "";
      const marker = detail?.marks?.[i] ?? "";
      return `<li data-rollout="${i + 1}"${selectedAttribute} class="group-trace-card">
      <small>#${i + 1}</small><b>${tokens[action]}</b>
      <span class="group-trace-reward" data-points="${reward}">R ${reward}</span>
      ${detailLine}${marker}
    </li>`;
    })
    .join("");
  return `<figure class="lab-picture group-trace"><figcaption>${title}</figcaption>
    <ol class="group-trace-grid">${cards}</ol><p class="group-trace-legend">${legend}</p></figure>`;
}

// Each row of replies keeps the original sampling order, including repeated
// commands. All cards use the same zero and scale for the difference bars.
function differenceMark(value) {
  const width = Math.min(50, 50 * Math.abs(value));
  const side = value < 0 ? "right" : "left";
  return `<span class="group-trace-axis" aria-hidden="true"><i style="${side}:50%;width:${width}%" class="${side}"></i></span>`;
}

export function renderGroupVisual(step, v, t) {
  const rewardKey = t("R = reward", "R = 奖励");
  const groupTitle = t(
    "Eight replies to “Stop the rover”",
    "“让小车停车”的八份回答",
  );
  const fourTitle = t(
    "Four replies to “Stop the rover”",
    "“让小车停车”的四份回答",
  );
  switch (step.id) {
    case "score-rollout":
    case "reward-steers":
      return figure({
        title: fourTitle,
        actions: firstFour,
        rewards: firstFour.map(v.rule ? exact : shortest),
        legend:
          rewardKey +
          " · " +
          (v.rule
            ? t("HOLD earns 1", "HOLD 得 1 分")
            : t("The shortest command earns 1", "最短的命令得 1 分")),
      });
    case "score-group":
      return figure({
        title: groupTitle,
        actions: group,
        rewards: group.map(exact),
        selected: v.i - 1,
        legend:
          rewardKey +
          " · " +
          t(`Inspecting reply #${v.i}`, `正在查看第 ${v.i} 份回答`),
      });
    case "group-mean": {
      const rewards = group.map(exact);
      const sum = rewards.reduce((a, b) => a + b, 0);
      return figure({
        title: groupTitle,
        actions: group,
        rewards,
        legend:
          rewardKey +
          " · " +
          t("Your division", "你选的除法") +
          `  ${sum} ÷ ${v.divisor} = ${(sum / v.divisor).toFixed(2)}`,
      });
    }
    case "baseline":
    case "centre-rewards":
    case "scale-advantages":
    case "zero-reward-spread": {
      const useFour = step.id === "baseline";
      const actions = useFour ? firstFour : group;
      const flat = step.id === "zero-reward-spread" && v.rule === 1;
      const rewards = actions.map((action) => (flat ? 1 : exact(action)));
      const useFixed = useFour || step.id === "centre-rewards";
      const metrics = useFixed ? null : grpo({ rewards }).metrics;
      const baseline = useFixed ? v.b / 100 : metrics.meanReward;
      const normalize =
        step.id === "zero-reward-spread" ||
        (step.id === "scale-advantages" && v.scale === 1);
      const differences = normalize
        ? metrics.advantages
        : rewards.map((r) => r - baseline);
      const symbol = normalize || useFour ? "A" : "Δ";
      const comparison = normalize
        ? t("A = normalized advantage", "A = 标准化后的优势")
        : `${symbol} = R − ${baseline.toFixed(2)}`;
      return figure({
        title: useFour ? fourTitle : groupTitle,
        actions,
        rewards,
        detail: {
          symbol,
          values: differences.map(signed),
          marks: differences.map(differenceMark),
        },
        legend: `${rewardKey} · ${comparison}`,
      });
    }
    case "old-current-ratio":
    case "clip-one-term": {
      const m = grpo({ logits: policyAt(v.p) }).metrics;
      const showTerms = step.id === "clip-one-term";
      const detail = showTerms
        ? {
            symbol: "L",
            values: m.ratios.map((r, i) => {
              const ratioUsed = m.clipped[i]
                ? m.advantages[i] > 0
                  ? 1.2
                  : 0.8
                : r;
              return signed(ratioUsed * m.advantages[i]);
            }),
            marks: m.clipped.map((clipped) =>
              clipped
                ? `<small class="group-trace-cap">${t("Clipped", "已裁剪")}</small>`
                : "",
            ),
          }
        : { symbol: "r", values: m.ratios.map((r) => r.toFixed(2)) };
      return figure({
        title: groupTitle,
        actions: group,
        rewards: group.map(exact),
        detail,
        selected: 0,
        legend:
          rewardKey +
          " · " +
          (showTerms
            ? t(
                "L = this reply’s objective term",
                "L = 这份回答对目标函数的贡献",
              )
            : t(
                "r = current / old probability",
                "r = 当前概率 ÷ 采样时的旧概率",
              )),
      });
    }
    default:
      return "";
  }
}
