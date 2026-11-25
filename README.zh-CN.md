# 大模型训练场 · Training Lab

[开始游戏](https://hacking.pub/games/training-lab/) · [English](README.md) · [神经工坊](https://hacking.pub/games/neural-atelier/)

练习里有一辆小车，只会回答 HOLD（停车）、GO（前进）、CHECK（检查）。同样是让小车更愿意回答 HOLD，有时我们给它看日志，有时直接给它示范，也可以让它先回答、再给分。

## 50 个拆解练习

| 章节                                                                         | 步数 | 你动手做的事                                                                                                       |
| ---------------------------------------------------------------------------- | ---: | ------------------------------------------------------------------------------------------------------------------ |
| [预训练](https://hacking.pub/games/training-lab/#learn/pretrain/next-target) |   24 | 从预测下一个词走到更新参数，再试数据量变少、重复读同一批日志、验证集混入原题，以及两类文本比例变化。               |
| [SFT](https://hacking.pub/games/training-lab/#learn/sft/write-demo)          |    5 | 写停车示范，选出要计算损失的部分，算回答的损失，更新一次，保存这轮的参数。                                         |
| [DPO](https://hacking.pub/games/training-lab/#learn/dpo/pair-replies)        |    5 | 配出一好一差两个回答，计算二者的概率比，与冻结的参考模型对比，试 β，再更新一次。                                   |
| [RL · REINFORCE](https://hacking.pub/games/training-lab/#learn/rl/draw-one)  |    6 | 抽一个回答、给分、故意试一次会教错的奖励，再看减去基线、更新参数和重新采样各起什么作用。                           |
| [GRPO](https://hacking.pub/games/training-lab/#learn/grpo/one-prompt-group)  |   10 | 同一道题抽八份回答，逐个给分，算组内均值和优势。把全组改成同分看看，再试概率比、裁剪、KL，最后对应到 verl 的配置。 |

进入第一步，只要移动阅读边界，看看哪个词成了目标。点「保存作对照」可以留住眼前的数，再移动滑杆看变化。「检查这一步」会根据你当前的选择给反馈。刷新或下次回来会恢复上次的位置、参数和对照值。

## 数据量变少会怎样？

预训练前 8 步讲怎么预测、算损失、更新参数。从这里往下，可以按自己关心的问题选练习。

- **先数清训练量（3 步）。** 一小段文本能出几道预测题？一批有几个目标位置？总共只读 96 个位置，能更新几次参数？[数一数目标位置](https://hacking.pub/games/training-lab/#learn/pretrain/count-targets)
- **少读几条，还是多读几遍（6 步）。** 从 12 条日志里留下 3 条，看看哪些词消失了。然后让 A 把这 3 条读四遍，B 把 12 条各读一遍。两边都训练 12 个目标词，浏览器逐步更新参数，再固定参数到同样六条留出的日志上算损失。[缩小训练集](https://hacking.pub/games/training-lab/#learn/pretrain/shrink-corpus) · [对齐预算后比较](https://hacking.pub/games/training-lab/#learn/pretrain/fixed-read-budget)
- **看看数据里混进了什么（6 步）。** 找出一模一样的行，看看多复制一份怎样改变一个词的训练占比，再从验证集里找出做过的原题。接下来调两类日志的配比，分开看两类各自的损失。[从识别重复行开始](https://hacking.pub/games/training-lab/#learn/pretrain/exact-copies)
- **换十二批资料再试一次（1 步）。** 自己选每次能拿到几条日志。每次训练都读 48 个目标词、更新 16 次，看十二次里有几次完全没抽到 CHECK。让这些模型做同样的 24 道留出题，再比较损失的最小值、中位数和最大值。[比较十二次训练](https://hacking.pub/games/training-lab/#learn/pretrain/compare-subsets)

上面的损失都由游戏里的小模型算出。想看真实语言模型上的实验，可以对照研究者怎样固定计算预算比较重复训练、检查训练和验证的重叠、调整语料来源的配比。[有限语料与重复训练](https://papers.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html) · [原作者的开源实验](https://github.com/huggingface/datablations) · [数据去重](https://aclanthology.org/2022.acl-long.577/) · [DoReMi 语料配比](https://arxiv.org/abs/2305.10429)

## 五个综合实验

每章最后都有一个可反复训练的小车模型。拆解练习用固定例子方便重算，综合实验可以换数据、连续更新，保存自己的模型。SFT 可以接着你的预训练通关点运行。DPO、REINFORCE 和 GRPO 各自从 SFT 分支出发。每个实验的「这一步具体怎么算」会展示本次更新的数值，训练手册附有 nanoGPT、verl 和论文入口。

## 在本机运行

在仓库目录执行 `python3 -m http.server 8000 --bind 127.0.0.1`，打开[本地游戏](http://127.0.0.1:8000)。运行测试使用 Node.js 20 或更高版本，命令为 `node --test tests/*.test.mjs`。本站默认英文，右上角可切中文。

MIT 许可证。
