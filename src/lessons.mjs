// The explanations are original. The source links lead to the algorithm or to
// the training code where the reader can follow the corresponding operation.
const pair = (en, zh) => ({ en, zh });

const nanoGPT = {
  label: "nanoGPT · get_batch",
  url: "https://github.com/karpathy/nanoGPT/blob/master/train.py#L109-L124",
};
const instructGPT = {
  label: "InstructGPT · Supervised demonstrations",
  url: "https://arxiv.org/abs/2203.02155",
};
const dpo = {
  label: "Direct Preference Optimization",
  url: "https://arxiv.org/abs/2305.18290",
};
const reinforce = {
  label: "Spinning Up · Policy gradients",
  url: "https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html",
};
const grpo = {
  label: "verl · Group Relative Policy Optimization",
  url: "https://verl.readthedocs.io/en/latest/algo/grpo.html",
};
const ppo = {
  label: "verl · Proximal Policy Optimization",
  url: "https://verl.readthedocs.io/en/latest/algo/ppo.html",
};
const recipe = {
  label: "verl · Qwen3 GRPO recipe",
  url: "https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_8b_fsdp.sh#L119-L153",
};
const deepSeekMath = {
  label: "DeepSeekMath · Original GRPO paper",
  url: "https://arxiv.org/abs/2402.03300",
};
export const lessons = [
  {
    id: "pretrain",
    title: pair("Predict the next token", "猜下一个 token"),
    brief: pair(
      "Finish “The signal says …” using examples from the rover log. Build a batch with two HOLD tokens for each GO, then check which token the model favors.",
      "日志里的句子都从“The signal says …”开始。让后面接 HOLD 的样本数是 GO 的两倍，再看模型更倾向于接哪个词。",
    ),
    hint: pair(
      "Count the token immediately after “The signal says” in each line. HOLD, HOLD, GO gives the target a two-to-one advantage in the data.",
      "每行只数“The signal says”后面的那个词。HOLD、HOLD、GO 这三行里，HOLD 的出现次数就是 GO 的两倍。",
    ),
    learn: pair(
      "The training text supplies its own targets. For the sequence A B C, the prefix A has target B, and A B has target C. In nanoGPT you can see this one-position shift where get_batch builds x and y.",
      "文本自己就能给出答案。例如序列 A B C 可以出两道题：看到 A 要预测 B，看到 A B 要预测 C。nanoGPT 的 get_batch 就是把 y 的起点比 x 往后移一个位置。",
    ),
    source: nanoGPT,
  },
  {
    id: "sft",
    title: pair("Teach by example", "给助手一个示范"),
    brief: pair(
      "The rover hears “Stop the rover. Reply with one command.” Choose the command you want to teach it and check how the probability of that answer changes.",
      "这次指令是“停车，只回答一个命令”。你希望它回答 HOLD、GO 还是 CHECK？用你的选择作示范，看训练后这个回答的概率。",
    ),
    hint: pair(
      "HOLD applies the brakes. With HOLD as the demonstration, the supervised loss for this one answer token is −log P(HOLD | instruction).",
      "HOLD 表示停车。用它作示范，这个单 token 回答的监督损失就是 −log P(HOLD | 指令)。模型越确信应该答 HOLD，损失越小。",
    ),
    learn: pair(
      "Supervised fine-tuning (SFT) learns from instruction–answer examples. In a completion-only setup, the instruction provides context and the demonstration supplies the tokens used to score the prediction.",
      "监督微调（SFT）用指令和示范回答来训练。只对回答算损失时，模型读完整条指令，再按顺序预测示范回答里的 token。",
    ),
    source: instructGPT,
  },
  {
    id: "dpo",
    title: pair("Choose the better reply", "选出更好的回答"),
    brief: pair(
      "For the stop instruction you prefer HOLD over GO. Label the pair and track P(HOLD) / P(GO) through an update.",
      "对“停车”这条指令，你更希望听到 HOLD。把 HOLD 和 GO 标成偏好对，观察更新前后的 P(HOLD) / P(GO)。",
    ),
    hint: pair(
      "The numerator is the chosen answer. Compare its odds against the same ratio measured by the frozen reference.",
      "把选中的回答放在分子。先算当前模型的 P(HOLD) / P(GO)，再和固定参考模型算出的比值比一比。",
    ),
    learn: pair(
      "Direct Preference Optimization (DPO) compares two odds ratios. One comes from the model you are training and the other from a frozen reference. Training on a labeled pair favors the chosen reply relative to the rejected reply.",
      "直接偏好优化（DPO）会比较两个比值：当前模型的「好回答概率 ÷ 差回答概率」，以及固定参考模型给出的同一个比值。这对标注给出的更新方向，是让当前模型更偏向好回答。",
    ),
    source: dpo,
  },
  {
    id: "rl",
    title: pair("Design the reward", "给回答打分"),
    brief: pair(
      "The rover needs to stop. Compare a rule that rewards HOLD with a rule that rewards short commands. Which one gives the model an incentive to move?",
      "任务是停车。试试“答 HOLD 给高分”和“命令越短分越高”两种规则，看哪一种会鼓励车往前走。",
    ),
    hint: pair(
      "GO has two letters and HOLD has four. The short-answer rule can reward GO even on a stop instruction.",
      "GO 有两个字母，HOLD 有四个。“命令越短分越高”这条规则会给 GO 高分，哪怕题目让车停下。",
    ),
    learn: pair(
      "REINFORCE weights a sampled decision by its reward, optionally after subtracting a baseline. A reply with positive advantage contributes an update toward repeating that decision. The scoring rule determines which behavior earns credit.",
      "REINFORCE 用奖励来决定这次采样对更新的影响。减去基线后得到优势值。优势为正，这条样本就会推动模型下次做出同样的选择。规则奖励什么，模型就会努力多做什么。",
    ),
    source: reinforce,
  },
  {
    id: "grpo",
    title: pair("Compare a group of replies", "把同一题的回答放在一起比"),
    brief: pair(
      "Score a group once with a constant reward and once with a reward for HOLD. Inspect the advantage of each answer in a group containing different commands.",
      "让车对同一条指令答几次。先给所有回答同分，再改成只奖励 HOLD，看看一组含不同命令的回答各自得到什么优势值。",
    ),
    hint: pair(
      "A group with mixed scores distinguishes answers. If every answer earns 1, the mean is also 1. Subtract it and every answer has an advantage of 0.",
      "有高有低的分数能区分回答。假如全组都得 1 分，均值也是 1，每个分数减去均值都得到 0。",
    ),
    learn: pair(
      "Group Relative Policy Optimization (GRPO) uses the other answers to the same prompt as a baseline. Equal rewards give zero group advantage, so this group contributes no reward-driven update. A separately enabled KL term can still pull the policy toward its reference.",
      "GRPO 用同一道题这组回答的平均奖励作基线。全组同分时，组内优势都是 0，这组奖励就没有给出更新方向。如果另开了 KL 约束，它仍可能让当前策略向参考模型靠近。",
    ),
    source: grpo,
  },
];

export const glossary = [
  {
    term: "Pretraining / Post-training",
    text: pair(
      "Pretraining learns patterns from a text corpus. Post-training adapts a pretrained model to respond in ways people want, using demonstrations, preferences or rewards. REINFORCE, PPO and GRPO are examples of reinforcement learning (RL).",
      "预训练从语料学会预测后文。后训练接着调整已有模型，让它按期望的方式回答，可以用示范、偏好或奖励。强化学习（RL）是一类方法，REINFORCE、PPO、GRPO 都属于它。",
    ),
    url: instructGPT.url,
  },
  {
    term: "Actor / Policy",
    text: pair(
      "The actor is the model being updated. For a given prefix, its policy gives the probabilities used to sample the next token. In the game each bar stands for a possible choice.",
      "Actor 是要更新的模型。给定前文，它的策略给出下一个 token 的概率，采样就按这些概率进行。游戏里每根概率条对应一个选项。",
    ),
    url: ppo.url,
  },
  {
    term: "Rollout",
    text: pair(
      "A rollout is a sampled response to a prompt, together with the information needed for training. In the verl GRPO recipe, actor_rollout_ref.rollout.n controls how many responses are sampled per prompt.",
      "Rollout 是模型对一个问题采样生成的一次回答及其训练记录。verl 示例里的 actor_rollout_ref.rollout.n 决定每道题生成几份回答。",
    ),
    url: recipe.url,
  },
  {
    term: "Reward",
    text: pair(
      "A reward is the score assigned to a sampled answer. A math checker can give 1 for a correct result and 0 for an incorrect one. Other tasks can use a learned reward model.",
      "奖励是给一份采样回答打的分。比如验算器给算对的答案 1 分，给算错的 0 分。别的任务也可以用训练出的奖励模型打分。",
    ),
    url: grpo.url,
  },
  {
    term: "Reference policy / KL",
    text: pair(
      "A frozen reference keeps a set of probabilities for comparison. DPO uses it to compare preferences. In RL, a KL penalty can make an update pay a cost when the policy drifts from that reference.",
      "参考模型的参数被冻结，给出的概率可以当作对照。DPO 用它比较偏好。强化学习中还可以加 KL 这一项，让偏离参考策略的更新付出代价。",
    ),
    url: grpo.url,
  },
  {
    term: "Critic / PPO",
    text: pair(
      "In the usual actor–critic form of PPO, the critic predicts expected future reward from a prefix. PPO uses that value to estimate advantage and clips its policy objective. GRPO instead estimates advantage from rewards in the same prompt group.",
      "常见的 PPO 会另外训练一个 critic，输入当前前文，估计后面能拿多少奖励。PPO 据此估算优势，并对策略目标做裁剪。GRPO 的优势来自同一道题一组回答的奖励。",
    ),
    url: ppo.url,
  },
];

export const references = [
  nanoGPT,
  instructGPT,
  dpo,
  reinforce,
  ppo,
  grpo,
  deepSeekMath,
  recipe,
];
