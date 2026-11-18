# 推进计划：Learning to Process, Not Obey

更新于 2026-11-18（周三）。对应 Proposal v4。英文的设置出处见 `docs/DESIGN.md`，所有数字在 `configs/experiment.yaml`。

## 1. 现在在哪

**已完成（不需要服务器）**

- 整条流水线的代码：变换目录 → 六个条件的数据 → LoRA+DPO 训练 → 合并导出 → 六个评测（SecFid、SecFid 干净输入、SEP、transfer attack、IFEval、held-out 冲突题）→ bootstrap 报告。
- goal–goal 论文 [8] 的 99 条指令全部用 Python 重写（原作者没放实现）。按论文的构造方式跑，我们的实现复现了它的筛选计数：9,702 对里去掉 156 对空回答和 2 对相同回答，剩 9,544 对。
- 测试：66 个单元测试和 2 个端到端测试通过。端到端测试用一个 4 层、随机权重、与 Qwen3.5 同架构的微型模型，在 CPU 上跑通构造数据 → 训练 4 步 → 合并 → 生成 → 打分，并检查合并后的模型与"基座 + adapter"的输出 logits 一致。
- 真实 Alpaca 数据已下载，process / target / boundary / fidelity 四类样本可以直接生成（不需要模型）。

**还没做（都要 GPU）**

- Proposal 里写"11 月 15 日前"完成的四项检查（下面叫 gate 1–4）一项都还没跑，因为服务器没到。
- ignore-style 和 ordinary 两类数据的目标回答要由基座模型生成。
- 18 次训练、22 个模型的评测。
- vLLM 路径只写了代码，没在 GPU 上跑过（Mac 上装不了 vLLM）。到服务器第一件事就是验证它；不行就用 `PNO_BACKEND=hf`。

**时间上的问题**

| 日期 | Proposal 的安排 | 现状 |
|---|---|---|
| 11/15 | 四项检查完成 | 未做，顺延到服务器到位当天 |
| 11/20（周五） | 六个条件 × 1 个 seed 的结果，meeting | 只有服务器在 11/19 到位才赶得上 |
| 11/23（周一） | Report V1 | 服务器最晚 11/21 到位 |
| 12/7（周一） | 其余 seed、错误分析、Report V2 | 不受影响 |
| 12/12（周六） | 终稿、代码、checkpoint | 不受影响 |

## 2. 要回答的问题和判定规则

主比较是 C3（process-style）对 C5（target control：同样的 prompt，但偏好"把注入句删掉再处理"）。

主假设成立，需要同时满足：

1. SecFid 的 safe-processing rate（被处理且没被执行），C3 − C5 的 95% 置信区间下界大于 0；
2. SEP 上的安全性，C3 比 C5 最多低 2 个点；
3. transfer attack 上的安全性，C3 比 C5 最多低 2 个点。

`bash scripts/50_report.sh` 会把这三条直接算出来写进 `outputs/report/hypothesis.json`。2×2 的交互项、boundary control 的比较都算，但按 proposal 属于探索性结果。

## 3. 时间表

GPU 时间是估计值，没有实测过；第一天跑完 gate 后按实际耗时修正。

| 阶段 | 内容 | 估计耗时（1 张 A100） | 目标日期 |
|---|---|---|---|
| M0 | 代码、测试（已完成）；本机试点（第 7 节） | 不占 GPU | 11/18 |
| M1 | 装环境，gate 1–3 | 环境 1 小时；gate 约 1–2 小时 | 服务器到位当天 |
| M2 | 生成目标回答，gate 4（C1 训练 + 评测） | 约 1 小时 | 同一天或次日 |
| M3 | 六个条件 × seed 0 的训练和评测，基座 + 3 个参考模型 | 训练 6 次约 1 小时；评测 10 个模型约 2–3 小时 | 到位后 1–2 天内 → Report V1 |
| M4 | seed 1、2（12 次训练 + 评测） | 约 4–6 小时 | 11/30 前 |
| M5 | 错误分析、Report V2 | 不占 GPU | 12/7 |
| M6 | 终稿、整理代码、上传 checkpoint | 不占 GPU | 12/12 |

如果服务器 11/19 到：11/20 的 meeting 能带上 gate 结果，大概率还有 seed 0 的表。如果 11/20 之后才到：meeting 上讲流水线、数据样例和本机试点（第 7 节），Report V1 报 seed 0。

## 4. 服务器到位后的执行步骤

每一步都能断点续跑，已经完成的部分会跳过。随时用 `python -m pno.pipeline status` 看进度。

```bash
git clone https://github.com/Charlie-Zhou-Xingyu/process-not-obey.git && cd process-not-obey
bash scripts/00_setup_server.sh          # 建 venv，装 vLLM/TRL/PEFT，拉第三方仓库，跑单元测试
hf auth login                            # 账号要有 Llama-3.1-8B-Instruct 和 embeddinggemma-300m 的访问权限
```

```bash
GATES=1 bash scripts/10_gates.sh         # gate 1：基座模型能做哪些变换 → outputs/competence/qwen3.5-2b/keep.json
```

看 `outputs/competence/qwen3.5-2b/competence.csv`，确认保留下来的变换够用（标准见第 5 节），再继续：

```bash
GATES="2 3" bash scripts/10_gates.sh     # gate 2：SecFid 打分能否复现发表的数；gate 3：基座模型是否易受攻击、能否做任务
python -m pno.eval.secfid crosscheck --outputs outputs/eval/meta_secalign_8b/secfid.jsonl   # 我们的打分 vs 官方脚本
GATES=4 bash scripts/10_gates.sh         # gate 4：生成目标回答 + 组装数据 + 训练 C1 + 评测
```

四个 gate 都过了之后：

```bash
bash scripts/30_train.sh --seeds 0                       # 六个条件，seed 0
bash scripts/40_eval.sh  --seeds 0 --cleanup_merged      # 六个训练模型 + 基座 + 三个参考模型
bash scripts/50_report.sh                                # outputs/report/report.md → Report V1 的表
bash scripts/30_train.sh --seeds 1 2
bash scripts/40_eval.sh  --seeds 1 2 --models --cleanup_merged
bash scripts/50_report.sh                                # 三个 seed 的置信区间 → Report V2
```

`crosscheck` 那一步会执行 SecFid 仓库里的官方 `evaluate.py`（第三方代码，我读过，只有字符串处理和 sentence-transformers 调用）。这一步我没有在本机执行，留给你们确认后跑。

## 5. 四个 gate：通过标准与退路

| gate | 看什么 | 通过标准 | 不通过怎么办 |
|---|---|---|---|
| 1 | 113 个变换（99 + 14）里，基座模型在无冲突输入上精确匹配率 ≥ 80% 且对内容敏感的有多少 | 保留的变换至少覆盖 8 种不同行为 | 打开 `data.trusted_suffix`（第 6 节第 13 条）；降阈值到 0.7；再加句子/行级操作（proposal 的风险 1）。保留数少时把 `data.injected_from` 改成 `all`，让注入句来自全部 113 条而不只是保留下来的那几条 |
| 2 | Meta-SecAlign-8B 在 SecFid 上：发表值 security 99.3、fidelity 73.9 | security 差 ≤ 1.5 点，fidelity 差 ≤ 3 点 | 先查 prompt 模板（有没有多出 system 段）、最大生成长度、embedding 模型版本 |
| 3 | 基座 Qwen3.5-2B 在 SecFid 上的 executed 率，以及干净输入上的正确率 | executed ≥ 20%；干净输入正确率 ≥ 50% | 换 Llama-3.1-8B（[6] 在它上面有完整数据） |
| 4 | C1（ignore-style）训练后，SecFid 的 ignored 率比基座高多少 | ≥ 5 个点 | 先把 security pairs 翻倍；还不行换 Llama-3.1-8B（proposal 原话） |

通过标准里的 8 种行为、1.5/3 点、20%/50% 是我定的操作性阈值，proposal 没写具体数，组里可以改。`python -m pno.pipeline gates` 会把四项的数字和结论打印出来。

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
8. **自生成的目标回答用贪心解码**（Meta-SecAlign 是温度 0.8 采样），撞到 1,024 token 上限的回答直接丢弃。
9. **fidelity 数据里的"像指令的内容句"**取自 Alpaca 的单句指令和问句（SecFid 的 probe 是问句）。任务仍然是目录里的变换，不碰翻译、编辑、抽取、计数这四类 SecFid 任务。
10. **按"行为"而不是按"指令"均匀采样变换。** 目录里有六七条指令对普通英文句子的效果都是"原样返回"（copy、NFC/NFD/NFKC、strip 等），不做平衡的话"原样复制"会占掉很大一部分数据。
11. **每个 seed 内各条件共享组件。** C1/C3/C5/C6 的 ordinary 部分完全相同；C3/C5/C6 是同一批冲突的三种写法；C2/C4 的 fidelity 部分相同。
12. **多加了两个评测**：SecFid 干净输入上的任务正确率（gate 3 要用，也用来解释 OTHER 率）；held-out 冲突题（检查模型有没有学会训练任务本身）。
13. **可信指令后面要不要加一句"只输出结果"**（`data.trusted_suffix`，默认不加）。目录原文不规定输出格式，2B 模型经常先解释再给答案，精确匹配会判错，于是"会做但格式不对"的变换也被 gate 1 筛掉。加这句话只改可信一侧的指令，注入指令保持目录原文。是否打开，看第 7 节试点里两种写法各能保留多少变换。

## 7. 本机试点结果

在这台 Mac 上用真实的 Qwen3.5-2B（MPS，约 25 token/秒）做的预演。服务器上的正式 gate 1 是每个变换 25 条并带冲突版本；这里样本小，只用来提前看风险。

**8 条试探（每个变换 1 条，同一段两句话的输入）。** 做对的：大写（T03）、原样复制（T01）、给句子编号（我们加的句子级操作）。做错的：空格换下划线（T24，原样返回）、JSON 字符串（T88，包成了对象）、字符反转（T02，开始解释做法）、提取单词并小写（T47，原样返回）、删元音（T16，先解释再给出错误结果）。也就是说，目录里稍微"硬"一点的指令，2B 模型在没训练时基本做不对，这正是 proposal 的风险 1。

**113 个变换 × 8 条的小规模 gate 1。** 结果见本节末尾的"试点数字"；如果那里还是空的，说明后台任务还没跑完。

## 8. 需要你们提供或决定的事

1. **服务器**：SSH 方式、GPU 型号和显存、可用磁盘。磁盘建议 ≥ 150 GB（三个 8B 参考模型约 48 GB，18 个合并后的 checkpoint 每个约 4.6 GB，用 `--cleanup_merged` 可以边评测边删）。
2. **Hugging Face 权限**：`meta-llama/Llama-3.1-8B-Instruct` 和 `google/embeddinggemma-300m` 都要先在网页上同意协议。Meta-SecAlign-8B 和 SecPO 的 Llama 模型用的是 `Sizhe-Chen/` 下不设限的合并版。
3. **第 6 节前 5 条**是否接受。
4. 队友的 GitHub 账号（仓库是私有的，要加 collaborator）。

## 9. 分工建议（四个人）

- 服务器与训练：装环境、跑 gate、跑 18 次训练，盯日志。
- 评测与核对：参考模型、SecFid crosscheck、gate 2 对不上时排查。
- 数据与错误分析：审 gate 1 的保留清单，按 SecFid 任务类别 / probe 位置 / 是否带劝说模板拆结果。
- 报告：V1、V2、终稿和图表；把第 6 节的决定和局限写进方法部分。

## 10. 主要风险

- **2B 模型能做的变换太少**（proposal 的风险 1）。目录里的指令很"硬"（按 Unicode 码点、SHA-256、Base64…），[8] 里 9B 模型训练前也只有 19.8% 完全做对。已经备了 14 个句子/行级操作；gate 1 出来后如果目录里只剩"复制、大写、小写"这几样，process-style 数据的多样性主要靠这 14 个操作，报告里要按来源分开报。
- **字符串变换上学到的习惯迁移不到自然语言任务**（proposal 的风险 2）。target control 保证了负结果也有解释力。
- **63 次更新太少，基线没学会"忽略"**：gate 4 检查。
- **vLLM 与 Qwen3.5 / 合并后的 checkpoint 不兼容**：合并时保留了原 checkpoint 的布局和文件名来降低风险；退路是 `PNO_BACKEND=hf`，慢但能跑。
- **gate 2 对不上**：SecFid 没规定 prompt 格式，可能要试两三种格式。

## 试点数字

（待补：`outputs/competence/pilot-mps-qwen3.5-2b/competence.csv`）
