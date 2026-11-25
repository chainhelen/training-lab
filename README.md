# Training Lab

**[Play Training Lab](https://hacking.pub/games/training-lab/)** · [简体中文](README.zh-CN.md) · [Neural Atelier](https://hacking.pub/games/neural-atelier/)

Train a small rover controller one step at a time. It can answer with three tokens: `HOLD`, `GO`, and `CHECK`. Choose the data or reward, apply a gradient update, and watch each command's probability change. The game runs entirely in the browser and saves checkpoints on the device. English and Chinese are available on desktop and mobile.

## Guided route

Start with the reading boundary in a four-word log. Move it one word and watch the prediction target change. The course builds from that example through 50 hands-on steps.

| Chapter                                                                           | Steps | What you work out                                                                                                                                                  |
| --------------------------------------------------------------------------------- | ----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Pretraining](https://hacking.pub/games/training-lab/#learn/pretrain/next-target) |    24 | Predict the next token, update three scores and test what happens when data gets scarce, duplicated or unevenly sampled.                                           |
| [SFT](https://hacking.pub/games/training-lab/#learn/sft/write-demo)               |     5 | Write the demonstration, select which tokens contribute to loss, score its answer, update the model, and save a checkpoint.                                        |
| [DPO](https://hacking.pub/games/training-lab/#learn/dpo/pair-replies)             |     5 | Label a pair, divide its probabilities, compare the ratio with a frozen reference, apply β, and update the two scores.                                             |
| [RL with REINFORCE](https://hacking.pub/games/training-lab/#learn/rl/draw-one)    |     6 | Draw a reply, score it, try a misleading reward, subtract a baseline, update the policy, and draw again.                                                           |
| [GRPO](https://hacking.pub/games/training-lab/#learn/grpo/one-prompt-group)       |    10 | Sample a group, score each reply, center and scale rewards, and try an all-tie group. Then explore policy ratios, clipping, KL and the group-size setting in verl. |

Move a control to see the numbers and formula change together. **Save for comparison** keeps the current result on screen while you try a second setting. **Check this step** gives feedback and opens the next step when you meet the goal. You can also explore the steps in any order from the chapter menu. Returning to the game restores your place, settings, and comparisons in that browser.

## What changes when you have less pretraining data?

The pretraining chapter starts with eight steps on prediction, loss and parameter updates. Use the remaining sixteen steps to investigate the training data.

- **Count the budget (3 steps).** Mark the target positions in a text window, count how many a batch supplies, and work out how many updates fit within 96 target reads. [Count targets](https://hacking.pub/games/training-lab/#learn/pretrain/count-targets)
- **Shrink or repeat the corpus (6 steps).** Keep three of twelve records and check which endings disappear. Then read three records four times or twelve records once. Each run trains on twelve target positions. The browser updates the model, freezes its scores and measures loss on the same six held-out records. [Shrink the corpus](https://hacking.pub/games/training-lab/#learn/pretrain/shrink-corpus) · [Compare equal budgets](https://hacking.pub/games/training-lab/#learn/pretrain/fixed-read-budget)
- **Inspect the data (6 steps).** Find exact copies, measure how copying a line changes its training weight, and remove overlap between training and evaluation. Change the mix of two domains and check each domain’s loss. [Find the copies](https://hacking.pub/games/training-lab/#learn/pretrain/exact-copies)
- **Try twelve draws (1 experiment).** Choose the number of distinct records in each run. Keep spending 48 target reads on 16 updates and see how often the draws miss CHECK. Compare the smallest, median and largest loss on the same 24 held-out records. [Compare twelve training sets](https://hacking.pub/games/training-lab/#learn/pretrain/compare-subsets)

## Full experiments

Each chapter ends with an open-ended training bench. Guided lessons use fixed examples so you can replay a calculation. The full benches let you change data, take multiple gradient steps, and carry your saved model into later experiments.

| Experiment                                | What you control                                                   | What you can observe                                                                                                                             |
| ----------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Pretraining                               | Add log lines to a next-token corpus                               | The tokens' frequencies shape the next-token distribution.                                                                                       |
| Supervised fine-tuning (SFT)              | Pick a demonstration for the instruction “Stop the rover”          | Cross-entropy increases the demonstrated answer's probability.                                                                                   |
| Direct preference optimization (DPO)      | Choose a preferred and a rejected response                         | The chosen/rejected odds move relative to their odds under a frozen reference.                                                                   |
| REINFORCE                                 | Sample eight replies and score correctness or brevity              | Rewarding brevity makes the short, incorrect command more likely.                                                                                |
| Group relative policy optimization (GRPO) | Score a group uniformly, switch reward rules, or reuse the rollout | Equal rewards have zero group advantage. A mixed group supplies a gradient. Reusing a group changes the policy ratios and can activate clipping. |

SFT can initialize its three trainable scores from a completed pretraining experiment. DPO and the two RL experiments independently start from the SFT checkpoint when available. Jumping ahead uses a built-in starting policy. Resetting an experiment loads its available starting checkpoint and keeps earned badges.

## The model

Each run trains three logits for one fixed prefix or instruction. A softmax converts them into a categorical distribution. The model has a one-token response and an explicit loss for each method. Transferring a checkpoint copies the three scores. This small action space lets the game calculate gradients and KL exactly. A production language model predicts many tokens with a shared neural network.

GRPO uses sample standard deviation, importance ratios against the **rollout policy**, and a clipped surrogate. An optional exact forward KL term penalizes divergence from a separate **frozen reference policy**. REINFORCE uses a zero baseline and draws a fresh on-policy rollout for the next update. Displayed losses are evaluated before each update and have different meanings across experiments.

## Run locally

From the repository root, start a static server with Python 3.

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open [the local game](http://127.0.0.1:8000). Run the numerical tests with Node.js 20 or newer.

```sh
node --test tests/*.test.mjs
```

The [course catalog](src/course-data.mjs) connects the bilingual exercises to the guided interface. The full experiments use the [gradient implementation](src/trainer.mjs), [mission rules](src/lab.mjs), and [teaching notes](src/lessons.mjs). Serve both HTML files, the root CSS files, `favicon.svg`, and the `src` directory for a static deployment.

## Read the original work

- Examine the fixed-compute experiments on repeated data and their released training material. [Scaling Data-Constrained Language Models](https://papers.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html) · [datablations](https://github.com/huggingface/datablations)
- Explore the studies behind the exercises on duplicates, train–validation overlap, and domain weighting. [Lee et al. on deduplication](https://aclanthology.org/2022.acl-long.577/) · [DoReMi](https://arxiv.org/abs/2305.10429)
- Follow a text batch from inputs to shifted target tokens in [nanoGPT](https://github.com/karpathy/nanoGPT/blob/master/train.py).
- Read about human-written demonstrations in [InstructGPT](https://arxiv.org/abs/2203.02155) and preference pairs in the [DPO paper](https://arxiv.org/abs/2305.18290).
- Compare [policy gradients](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html), the [original GRPO paper](https://arxiv.org/abs/2402.03300), and the [verl GRPO guide](https://verl.readthedocs.io/en/latest/algo/grpo.html).
- Find the group size, advantage estimator, and KL settings in [verl's Qwen3 GRPO recipe](https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_8b_fsdp.sh). The game implements its own browser-sized trainer.

MIT licensed.
