# 大模型训练场 · Training Lab

[开始游戏](https://hacking.pub/games/training-lab/) · [English](README.md) · [神经工坊](https://hacking.pub/games/neural-atelier/)

练习里有一辆小车，只会回答 HOLD（停车）、GO（前进）、CHECK（检查）。同样是让小车更愿意回答 HOLD，有时我们给它看日志，有时直接给它示范，也可以让它先回答、再给分。

## 34 个拆解练习

| 章节                                                                         | 步数 | 你动手做的事                                                                                                                           |
| ---------------------------------------------------------------------------- | ---: | -------------------------------------------------------------------------------------------------------------------------------------- |
| [预训练](https://hacking.pub/games/training-lab/#learn/pretrain/next-target) |    8 | 移动阅读边界找出下一个词，统计日志结尾，把分数换成概率。先算一条日志的损失，再算一批的平均损失，试着更新一次，最后用留出的日志选模型。 |
| [SFT](https://hacking.pub/games/training-lab/#learn/sft/write-demo)          |    5 | 写停车示范，选出要计算损失的部分，算回答的损失，更新一次，保存这轮的参数。                                                             |
| [DPO](https://hacking.pub/games/training-lab/#learn/dpo/pair-replies)        |    5 | 配出一好一差两个回答，计算二者的概率比，与冻结的参考模型对比，试 β，再更新一次。                                                       |
| [RL · REINFORCE](https://hacking.pub/games/training-lab/#learn/rl/draw-one)  |    6 | 抽一个回答、给分、故意试一次会教错的奖励，再看减去基线、更新参数和重新采样各起什么作用。                                               |
| [GRPO](https://hacking.pub/games/training-lab/#learn/grpo/one-prompt-group)  |   10 | 同一道题抽八份回答，逐个给分，算组内均值和优势。把全组改成同分看看，再试概率比、裁剪、KL，最后对应到 verl 的配置。                     |

进入第一步，只要移动阅读边界，看看哪个词成了目标。点「保存作对照」可以留住眼前的数，再移动滑杆看变化。「检查这一步」会根据你当前的选择给反馈。刷新或下次回来会恢复上次的位置、参数和对照值。

## 五个综合实验

每章最后都有一个可反复训练的小车模型。拆解练习用固定例子方便重算，综合实验可以换数据、连续更新，保存自己的模型。SFT 可以接着你的预训练通关点运行。DPO、REINFORCE 和 GRPO 各自从 SFT 分支出发。每个实验的「这一步具体怎么算」会展示本次更新的数值，训练手册附有 nanoGPT、verl 和论文入口。

## 在本机运行

在仓库目录执行 `python3 -m http.server 8000 --bind 127.0.0.1`，打开[本地游戏](http://127.0.0.1:8000)。运行测试使用 Node.js 20 或更高版本，命令为 `node --test tests/*.test.mjs`。本站默认英文，右上角可切中文。

MIT 许可证。
