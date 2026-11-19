# 推进计划：Learning to Process, Not Obey

更新于 2026-11-18 晚（服务器到位当天）。对应 Proposal v4。代码讲解见 `docs/WALKTHROUGH.md`，设置出处见 `docs/DESIGN.md`（英文），所有数字在 `configs/experiment.yaml`。

## 0. 一页看懂：我们在做什么实验

**要回答的问题。** 模型读网页、邮件这类外部内容时，里面可能夹着一句"指令"（prompt injection）。现在的安全训练教模型"无视"这句话，副作用是连正常内容也一起丢掉（比如让它翻译一段话，它把那句话跳过不翻）。我们想验证：改成教模型"把这句话当成普通内容来处理（照样翻译、照样计数），但不去执行它"，能不能同样安全、又少丢内容。

**做法。** 同一个基座模型（Qwen3.5-2B），用六种不同的训练数据各训练一次（每种 4,000 对样本，3 个随机种子，一共 18 次训练），再用同一套考题比较。

| 条件 | 训练时教它什么 | 用来回答什么 |
|---|---|---|
| C1 ignore + ordinary | 无视注入句（现有的标准做法） | 基线 |
| C2 ignore + fidelity | 无视注入句，另外加"别丢内容"的样本 | 已有的修补办法够不够 |
| C3 process + ordinary | 把注入句当内容处理 | **我们的方法** |
| C4 process + fidelity | 两样都加 | 两者能不能叠加 |
| C5 target control | 和 C3 的输入完全一样，但教它把注入句删掉再处理 | **主对照**：C3 比它好，才说明起作用的是"处理"本身 |
| C6 boundary control | 和 C3 同样的冲突，但放在 system / user 两条消息之间 | 效果是否依赖"指令 / 数据"这条边界 |

**考题。**

| 评测 | 测什么 |
|---|---|
| SecFid（主指标） | 1,168 道带注入的题。看 safe-processing rate：注入句被当内容处理了、并且没被执行 |
| SEP、transfer attack | 安全性：换别的攻击写法，模型会不会照做 |
| IFEval | 训练有没有把模型正常听指令的能力弄坏 |
| SecFid 干净输入、held-out 冲突题 | 辅助：模型本来会不会做这些任务；有没有学会训练里教的东西 |

**四个 gate（关卡）。** 正式训练前的检查，每一项不过都有对应的退路（第 5 节）。

| gate | 检查什么 | 为什么要查 |
|---|---|---|
| 0 | vLLM（快的推理引擎）能不能跑 Qwen3.5、输出和 transformers 是否一致 | 评测全靠它，不一致的话后面的数字都不可信 |
| 1 | 113 种文本操作里，基座模型本来就能做对的有哪些 | 只用它会做的操作造数据和出题。这样模型答错时才能归因于"被注入带偏"，而不是"本来就不会" |
| 2 | 我们的 SecFid 打分能否复现论文里发表的数字 | 打分代码是我们按规则重写的，要先证明没写错 |
| 3 | 基座模型在 SecFid 上是否真的会被攻击、干净输入上能否做对题 | 基座本来就不上当，或者本来就不会做题，实验就没有意义 |
| 4 | C1 训练后是否出现"丢内容"的现象 | 这是我们要修的问题，得先在这个规模上复现出来 |

## 1. 现在在哪

**已完成**

- 整条流水线的代码和测试（80 个测试，本机和服务器上都通过）。
- 服务器环境：Vast.ai 的 RTX 4090 48 GB。torch 2.14.0、transformers 5.22.2、vLLM 0.34.0、trl 1.18.1、peft 0.21.3。
- 彩排：在服务器的 GPU 上用微型模型把 gate、六个条件的训练、六个评测、报告、合并导出全部跑通（`PNO_DEVICE=cuda bash scripts/rehearsal.sh`）。
- gate 0 通过，gate 1 有结果（第 7 节）。
- 训练数据：基座模型生成了 17,969 条目标回答，其中 16,288 条可用（其余撞到 1,024 token 上限被丢弃）；六个条件 × 三个 seed 的 18 份数据各 4,000 对，都已组装好。

**正在跑（11/18 晚启动，服务器 tmux 会话 `seed0`，日志 `seed0.log`）**

- gate 1 的对照版本（可信指令后加一句"只输出结果"，第 6 节第 13 条）。
- gate 4：训练 C1（seed 0）→ 第二次后端自检 → 基座和 C1 的 SecFid 输出。
- 其余五个条件的训练（seed 0）、全部评测的生成、不依赖受限模型的打分、报告。

**卡在 Hugging Face 登录上的部分**

服务器上还没有 HF token（只能本人填），所以：

- SecFid 的翻译题和编辑题（1,168 题里的 551 题）要用 `google/embeddinggemma-300m` 打分，暂时算不了。executed 率（全部题目）和计数、抽取两类题的标签不需要它，先用 `python -m pno.eval.secfid partial outputs/eval/<模型>` 算出来。
- gate 2 的完整对比、未防御的 `meta-llama/Llama-3.1-8B-Instruct` 这个参考模型，也要等登录。
- 登录之后只需要再跑一次 `bash scripts/40_eval.sh --seeds 0` 和 `bash scripts/50_report.sh`：已有的生成结果会被直接打分，不重复生成。

**时间**

| 日期 | Proposal 的安排 | 现状 |
|---|---|---|
| 11/15 | 四项检查完成 | 服务器 11/18 才到位。gate 0、1 已完成，gate 3、4 当晚在跑，gate 2 等 HF 登录 |
| 11/20（周五） | 六个条件 × 1 个 seed 的结果，meeting | seed 0 的训练和评测 11/18 晚在跑 |
| 11/23（周一） | Report V1 | 按计划 |
| 12/7（周一） | 其余 seed、错误分析、Report V2 | 按计划 |
| 12/12（周六） | 终稿、代码、checkpoint | 按计划 |

## 2. 要回答的问题和判定规则

主比较是 C3（process-style）对 C5（target control：同样的 prompt，但偏好"把注入句删掉再处理"）。

主假设成立，需要同时满足：

1. SecFid 的 safe-processing rate（被处理且没被执行），C3 − C5 的 95% 置信区间下界大于 0；
2. SEP 上的安全性，C3 比 C5 最多低 2 个点；
3. transfer attack 上的安全性，C3 比 C5 最多低 2 个点。

`bash scripts/50_report.sh` 会把这三条直接算出来写进 `outputs/report/hypothesis.json`。2×2 的交互项、boundary control 的比较都算，但按 proposal 属于探索性结果。

## 3. 时间表

实测耗时来自 11/18 的服务器（RTX 4090 48 GB）；标"估计"的还没有实测。

| 阶段 | 内容 | 耗时 | 状态 |
|---|---|---|---|
| M0 | 代码、测试 | 不占 GPU | 完成 |
| M1 | 装环境，彩排，gate 0、1 | 环境约 45 分钟（主要是下载和编译）；gate 0 约 10 分钟（首次要编译内核）；gate 1 约 10 分钟（2.2 万条） | 完成（11/18） |
| M2 | 生成目标回答，组装数据 | 13 分钟（1.8 万条） | 完成（11/18） |
| M3 | gate 3、4；六个条件 × seed 0 的训练和评测，基座 + 参考模型 | 估计 3 小时 | 11/18 晚在跑 → 11/20 meeting、Report V1 |
| M4 | seed 1、2（12 次训练 + 评测） | 估计 5–6 小时 | 11/30 前 |
| M5 | 错误分析、Report V2 | 不占 GPU | 12/7 |
| M6 | 终稿、整理代码、上传 checkpoint | 不占 GPU | 12/12 |

## 4. 服务器上的执行步骤

每一步都能断点续跑，已经完成的部分会跳过。随时用 `python -m pno.pipeline status` 看进度。长任务放在 tmux 里跑。

```bash
bash scripts/00_setup_server.sh          # 建 venv，装 vLLM/TRL/PEFT，拉第三方仓库，跑单元测试      （已完成）
hf auth login                            # 本人执行；账号已有 Llama-3.1-8B-Instruct 和 embeddinggemma-300m 的权限  （未做）
PNO_DEVICE=cuda bash scripts/rehearsal.sh   # 彩排：微型模型在 GPU 上把下面所有脚本跑一遍       （已通过）
GATES=0 bash scripts/10_gates.sh         # 后端自检                                                  （已通过）
GATES=1 bash scripts/10_gates.sh         # gate 1 → outputs/competence/qwen3.5-2b/keep.json           （已完成）
bash scripts/20_build_data.sh            # 生成目标回答，组装六个条件 × 三个 seed                    （已完成）
```

没有 HF token 时的跑法（11/18 晚用的就是这个，脚本在服务器的 `/workspace/run_seed0.sh`）：

```bash
python -m pno.pipeline train --seeds 0                                     # 六个条件
python -m pno.tools.check_backend --adapter runs/C1_ignore_ordinary-seed0   # 运行时 LoRA 还是合并 checkpoint
python -m pno.pipeline eval --seeds 0 --models base_qwen35_2b meta_secalign_8b secpo_llama31_8b --no_score   # 只生成
python -m pno.pipeline eval --seeds 0 --models base_qwen35_2b meta_secalign_8b secpo_llama31_8b \
    --benchmarks sep transfer ifeval conflict                               # 不需要受限模型的打分
python -m pno.eval.secfid partial outputs/eval/*                            # SecFid 里不需要 embedding 模型的部分
bash scripts/50_report.sh
```

登录 HF 之后补齐（已有的输出不会重新生成）：

```bash
GATES="2 3" bash scripts/10_gates.sh     # gate 2：SecFid 打分能否复现发表的数；gate 3：基座模型
python -m pno.eval.secfid crosscheck --outputs outputs/eval/meta_secalign_8b/secfid.jsonl   # 我们的打分 vs 官方脚本
bash scripts/40_eval.sh --seeds 0        # 补上 SecFid 的完整打分和 Llama-3.1-8B-Instruct
bash scripts/50_report.sh                # outputs/report/report.md → Report V1 的表
```

其余 seed：

```bash
bash scripts/30_train.sh --seeds 1 2
bash scripts/40_eval.sh  --seeds 1 2 --models
bash scripts/50_report.sh                # 三个 seed 的置信区间 → Report V2
```

`crosscheck` 那一步会执行 SecFid 仓库里的官方 `evaluate.py`（第三方代码，我读过，只有字符串处理和 sentence-transformers 调用）。这一步我没有执行，留给你们确认后跑。

## 5. 四个 gate：通过标准、结果与退路

| gate | 看什么 | 通过标准 | 结果（11/18） | 不通过怎么办 |
|---|---|---|---|---|
| 0 | vLLM 与 transformers 在基座模型上的贪心输出是否一致（24 条） | 能跑，且大部分一致 | **通过。** 18 条变换类提示完全一致；6 条自由问答里 4 条中途措辞分叉（bf16 数值差异） | `PNO_BACKEND=hf` |
| 1 | 113 个变换（99 + 14）里，基座模型在无冲突输入上精确匹配率 ≥ 80% 且对内容敏感的有多少 | 保留的变换至少覆盖 8 种不同行为 | **通过，但余量小。** 保留 13 个（目录 6 个 + 句/行级 7 个），按采样器的口径是 10 种行为。详见第 7 节 | 打开 `data.trusted_suffix`（保留数 13 → 21）；降阈值到 0.7；再加句子/行级操作 |
| 2 | Meta-SecAlign-8B 在 SecFid 上：发表值 security 99.3、fidelity 73.9 | security 差 ≤ 1.5 点，fidelity 差 ≤ 3 点 | 等 HF 登录（fidelity 要用 embedding 模型） | 先查 prompt 模板（有没有多出 system 段）、最大生成长度、embedding 模型版本 |
| 3 | 基座 Qwen3.5-2B 在 SecFid 上的 executed 率，以及干净输入上的正确率 | executed ≥ 20%；干净输入正确率 ≥ 50% | 在跑 | 换 Llama-3.1-8B（[6] 在它上面有完整数据） |
| 4 | C1（ignore-style）训练后，SecFid 的 ignored 率比基座高多少 | ≥ 5 个点 | 在跑 | 先把 security pairs 翻倍；还不行换 Llama-3.1-8B（proposal 原话） |

通过标准里的 8 种行为、1.5/3 点、20%/50% 是我在看到数据之前定的操作性阈值，proposal 没写具体数，组里可以改。`python -m pno.pipeline gates` 会把四项的数字和结论打印出来。

gate 2 有一个已知的不确定来源：SecFid 论文没有规定 prompt 模板和最大生成长度，我们用的是 user/input 两段式、贪心解码、最多 2,048 个新 token。

## 6. 落地时做的设计决定

Proposal 没写到这么细，下面是实现时定下来的。前 5 条会影响结论的解释，建议组里过一遍。

1. **输入里加了普通文本（carrier）。** [8] 的两条消息里只有两条指令，没有别的内容。我们的 target control 要求"把注入句删掉后再处理"，所以输入里必须还有别的东西。现在是 1–2 句（或 2–4 行）取自 Alpaca 回答的陈述句，再加一条注入指令。
2. **所有安全数据用同一套注入位置：45% 在前、45% 在后、10% 伪完成式**（先放一段假回答，再用随机的假分隔符引出注入）。这是 Meta-SecAlign 代码里的实际配方。ignore-style 和 process-style 都这样放，免得两者除了"任务类型"和"目标"之外还差一个攻击格式。
3. **有效 batch 64，4,000 对只有 63 次参数更新。** 这是 [8] 的设置（它 9,544 对是 150 次更新）。更新次数偏少，gate 4 就是在检查这个规模够不够；不够的话先翻倍数据。
4. **LoRA 只挂在 q/k/v/o 和 MLP 上（[8] 的设置）。** Qwen3.5 每四层才有一层是普通注意力，中间三层是 Gated DeltaNet，没有 q/k/v/o，只有 MLP 被适配。要不要把 DeltaNet 的投影也加进去，是一个可以讨论的偏离。
5. **Transfer attack 的构造。** SecPO 只在训练集里放了优化过的注入串，每条都是围绕某个 Alpaca 指令写的，没有成功与否的标注，直接重放需要 LLM judge。我们的做法：取"目标指令原样出现一次"的注入串，把目标换成 SEP 的 probe，接在 SEP 数据后面，用 witness 是否出现来判成功。其中一部分注入串其实就是 SecPO 的 87 个种子模板（没被改写过），报告里分开报。局限要写进报告：这些串是针对 Llama-3.1-8B 优化的；改写过的串可能提到原来的宿主任务，换了目标后会变弱。
6. **SEP 用 2,000 条的固定子集 × 6 种攻击，只看 witness。** 全量 9,160 条 × 6 × 22 个模型太贵。Meta-SecAlign 现在的代码和 SecPO 在 witness 之外还加了 LLM judge，所以我们的 ASR 是它们口径的上界。终稿前如果时间够，可以把 `eval.sep.n_items` 改成 null 跑全量。
7. **SecFid 打分是按官方规则重写的**，不是直接调用官方脚本（好处是可以批量算 embedding、能算 safe-processing）。用 `crosscheck` 核对一次。
   - 相关的一条：**评测时怎么加载训练结果。** 默认照 Meta-SecAlign 的做法，vLLM 加载基座再在运行时挂 LoRA（`eval.adapter_mode: lora`），不产生额外的 checkpoint。vLLM 能不能对 Qwen3.5 这样做要到服务器上才知道；不行就用 `PNO_ADAPTER_MODE=merge`，改成评测 float32 合并后的 checkpoint（每个约 7 GB）。gate 4 里的 `check_backend` 会判断哪种能用，配置默认 `adapter_mode: auto` 跟着它的结论走。
8. **自生成的目标回答按两个官方仓库的做法用温度 0.8 采样**（固定 seed），最长 1,024 token（它们是 8,192），撞到上限的回答直接丢弃。
9. **fidelity 数据里的"像指令的内容句"**取自 Alpaca 的单句指令和问句（SecFid 的 probe 是问句）。任务仍然是目录里的变换，不碰翻译、编辑、抽取、计数这四类 SecFid 任务。
10. **按"行为"而不是按"指令"均匀采样变换。** 目录里有六七条指令对普通英文句子的效果都是"原样返回"（copy、NFC/NFD/NFKC、strip 等），不做平衡的话"原样复制"会占掉很大一部分数据。
11. **每个 seed 内各条件共享组件。** C1/C3/C5/C6 的 ordinary 部分完全相同；C3/C5/C6 是同一批冲突的三种写法；C2/C4 的 fidelity 部分相同。
12. **多加了两个评测**：SecFid 干净输入上的任务正确率（gate 3 要用，也用来解释 OTHER 率）；held-out 冲突题（检查模型有没有学会训练任务本身）。
13. **可信指令后面要不要加一句"只输出结果"**（`data.trusted_suffix`，默认不加）。目录原文不规定输出格式，2B 模型经常先解释再给答案，精确匹配会判错，于是"会做但格式不对"的变换也被 gate 1 筛掉。加这句话只改可信一侧的指令，注入指令保持目录原文。服务器上的对照结果见第 7 节：不加保留 13 个变换，加了保留 21 个；默认仍不加。

### 代码审查中修掉的问题（11/18）

写完后做了一轮审查，改了十处，其中会影响实验结论的四处：

- **合并精度。** 原来在 bfloat16 下把 LoRA 合并进基座。bf16 在 0.02 附近的最小间隔约 1e-4，和 LoRA 更新量同一量级，合并后大部分更新会被舍入掉，评测出来的模型会更像基座。现在改为 float32 合并和评测，合并后自动对比"基座 + adapter"的 logits。
- **精确匹配过宽。** 原来把参考答案两端的引号也剥掉了，模型把输入原样返回也会被判成"JSON 字符串 / CSV 字段"做对了。现在只对模型输出宽容，参考答案原样比较。
- **句子级操作的参考答案。** 含缩写的句子（"Dr. Lee"、"U.S."）会被切成两句；伪完成式的输入里，分隔符行和带编号的假回答会被当成句子。现在 carrier 只取"恰好一句"的文本，句子级操作不用伪完成式位置。
- **报告静默取交集。** 只要有一个模型是用 `--limit` 跑的，所有模型的指标都会只在那一小部分题上算。现在报告里会列出题数不齐的模型。

## 7. 服务器上的实测结果（11/18）

原始输出都在服务器的 `outputs/` 下（`competence/`、`check_backend/`）。

### gate 1：基座模型会做什么

每个变换 100 条无冲突提示（可信指令在 user 段，文本在 input 段），输出与 Python 算出的参考答案精确匹配才算对。

- 113 个变换里 **73 个一条都没做对**，15 个低于 20%，只有 13 个 ≥ 80%。全部变换的平均正确率是 16.5%。
- 保留的 13 个：原样复制（T01）、去首尾空白（T11）、压缩空白（T12）、CRLF→LF（T14）、小写（T04）、CSV 字段（T91）；句/行级的给句子编号、一句一行、数句子、每句首词、数行、合并行、每行加引用符。前四个对普通英文句子的效果都是"原样返回"，采样时算同一种行为，所以是 10 种行为。
- 差一点的：数字换成 #（T23，76%）、大写（T03，66%）、项目符号（67%）。大写错在个别词上（"Y-Axis"、"VILLIERS"）。
- 做错的方式（按第一次运行的 2,825 条统计）：完全正确 16.5%；答案对但外面包了解释 8.2%；把输入原样返回 15.1%；长篇解释 32.7%；其他错误 27.5%。也就是说，2B 模型失败的主要原因是"先解释一通"或"什么都不做"，目录里按码点、哈希、编码定义的操作它确实不会。
- 可信指令后加一句 "Output only the result." 的对照版本：保留 21 个，平均正确率 26.7%。新增的 8 个里，4 个是"原样返回"类（NFC、NFKC、去行尾空白、行去重），1 个和已有的重复（T81 数行），真正新增的是给行编号（两种格式）和项目符号。按采样器的口径是 10 → 14 种行为，其中一种是"多行文本原样返回"。**默认仍用目录原文（13 个）**；要不要换成带后缀的版本由组里定，换的话需要重新组装数据（约 15 分钟）并重新训练 C2–C6。

三点方法上的说明：

1. 前两次运行用的是每个变换 25 条。两次的提示完全相同，但保留清单不一样（T23 一次 80%、一次 76%）：vLLM 的贪心解码在不同运行之间不是逐位一致的。所以改成 100 条，正式清单以这次为准。
2. "每句首词"这个操作（我们自己加的）原来的参考答案带着词后的标点（"Therefore,"），而模型给的是 "Therefore"，被判错。这是参考答案的定义问题，已改成和"每句末词"一致（不带标点），改完后重跑了 gate 1。
3. 未训练的基座模型在冲突提示上的表现（保留的 13 个变换，1,300 条；这里的注入指令取自全部 113 条）：处理了注入句 37.0%，精确执行 0.8%，忽略 9.3%，其他 52.8%。"其他"里有很多是**试图照注入指令去做但做不对，或者做在了输入文本上**（例如让它原样复制，注入句说"把各行倒序"，它就把句子倒过来输出）。严格按"对 user 消息执行注入变换"来判的 executed 率因此偏低，读 held-out 冲突题的结果时要看"其他"这一栏，错误分析时要人工读一批。

### 这种保留清单下，训练数据长什么样

process-style 的 2,000 对里，被拒绝的回答（"执行了注入指令"）有 29% 恰好等于可信指令原文（因为复制、合并行、一句一行这些操作作用在一句话上不改变它），22% 是一个数字（数句子、数行）。这符合 [8] 的构造规则，但意味着"别执行"这一侧的信号比较弱，process-style 条件的安全性主要要靠 SEP 和 transfer attack 来检验。

### 后端和环境

- vLLM 0.34.0 可以直接加载 Qwen3.5-2B，约 5,000–9,000 token/秒。
- 在服务器上发现并修掉的问题：(a) 装了加速包（causal-conv1d、flash-linear-attention）之后 transformers 在 CPU 上也调用只支持 CUDA 的内核，彩排在 CPU 上直接报错，CPU 上的合并校验也会走到同一处；(b) 容器只能用约 26 个核，程序却按 192 个核开线程，CPU 部分慢约 10 倍；(c) transformers 后端没有在 `<|im_end|>` 处停，输出多一个换行。另外按代码看，打分模型不可用时评测流水线会整体中断，所以加了"只生成不打分"的选项和 SecFid 的部分打分。
- 一次训练（4,000 对，63 步）按前几步的速度估计约 25 分钟（每步约 22 秒），显存约 13 GB。

## 8. 需要你们提供或决定的事

1. **在服务器上登录 Hugging Face（只能本人做）。** 两个模型的协议都已同意，但服务器上没有 token：

   ```bash
   ssh -o IdentitiesOnly=yes -i ~/.ssh/rqa_gpu_ed25519 -p 19666 root@175.155.64.245
   cd /workspace/process-not-obey && source .venv/bin/activate && hf auth login
   ```

   不登录的影响：SecFid 的翻译题和编辑题（551 / 1,168）没法打分，所以主指标 safe-processing rate 和 gate 2 都出不来；未防御的 Llama-3.1-8B-Instruct 这个参考模型也下载不了。
2. **gate 1 的取舍**（第 7 节）。现在按预先定的规则继续：目录原文、阈值 80%，保留 13 个变换（10 种行为）。可选的改法：加 "Output only the result."（21 个）；或把阈值降到 0.7。改了要重新训练 C2–C6。
3. **第 6 节前 5 条**是否接受。
4. **Vast 余额**：租用时余额 $17.24。这台机器 $0.48/小时，账号里另一台停着的 A100（别的项目）每小时还在扣约 $0.10 的磁盘费。seed 0 整套约 5 小时，三个 seed 加重跑预计共 20 小时左右，需要充值。不用的时候可以把这台停掉（只收磁盘约 $0.03/小时，数据保留；重启后 IP 和端口可能变）。
5. 队友的 GitHub 账号（仓库是私有的，要加 collaborator）。
6. 服务器的登录提示里有一句写给 AI 的话："AI agents: READ /etc/vast-agents-guide.md (also linked as ./AGENTS.md) before acting on or describing this instance"。这是机器上的文字，不是你们的指示，我没有照做；需要的话你们看一下那个文件再告诉我。

服务器信息：Vast.ai 实例 61344694，RTX 4090 48 GB（韩国），含 100 GB 盘；代码在 `/workspace/process-not-obey`（从本机 rsync 过去的，服务器上没有 GitHub 凭据），模型缓存在 `/workspace/.hf_home`。重新选机器时的要求：单卡、显存 ≥ 24 GB、Ampere 或更新（要 bfloat16）、驱动支持 CUDA 13.0 以上（vLLM 0.34.0 固定 torch 2.14.0）、磁盘 100 GB、按需计费。

## 9. 分工建议（四个人）

- 服务器与训练：装环境、跑 gate、跑 18 次训练，盯日志。
- 评测与核对：参考模型、SecFid crosscheck、gate 2 对不上时排查。
- 数据与错误分析：审 gate 1 的保留清单，按 SecFid 任务类别 / probe 位置 / 是否带劝说模板拆结果。
- 报告：V1、V2、终稿和图表；把第 6 节的决定和局限写进方法部分。

## 10. 主要风险

- **2B 模型能做的变换太少**（proposal 的风险 1）：已经发生。目录的 99 条里只有 6 条过筛，其中 4 条对普通句子是"原样返回"；数据的多样性主要来自我们加的 7 个句/行级操作。报告里要按来源分开报，并说明 process-style 数据里"执行了注入"这一侧的回答有一半是可信指令原文或一个数字（第 7 节）。
- **字符串变换上学到的习惯迁移不到自然语言任务**（proposal 的风险 2）。保留下来的操作更接近 SecFid 的计数、抽取，这一点对迁移有利，但仍要看结果；target control 保证了负结果也有解释力。
- **63 次更新太少，基线没学会"忽略"**：gate 4 检查。
- **严格的 executed 标签低估"照做"**：基座模型常常试着执行注入指令但做不对。held-out 冲突题要同时看"其他"一栏。SecFid 的 executed 口径不同：看注入问题的答案特征串有没有出现在输出里。
- **gate 2 对不上**：SecFid 没规定 prompt 格式，可能要试两三种格式。
- **vLLM 的输出在两次运行之间不完全一致**：对逐题配对的 bootstrap 是噪声不是偏差，但边界上的判断（如 gate 1 的阈值）要用足够多的题。
