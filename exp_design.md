# 实验设计

## 实验 ： PIM Attention 中保护 KV Cache 的必要性

### 1. 前期实验目标

前期不要铺开太多实验，核心目标只做一件事：

> 用可复现的 vLLM fault injection 证明：当 LLM 的 attention 由 PIM/近存单元读取 `KV cache` 时，`KV cache` 中少量 BF16 bit flip 就可能造成输出偏移、答案错误或生成崩溃，因此 PIM 内需要对 `KV cache` 做快速检测和触发式纠错。

本阶段不研究：

- 真实物理 soft error rate 建模。
- 多模型泛化。
- 大规模 layer/head/dim 全扫描。
- prefix caching / shared prefix block。
- 完整硬件 ECC 实现。

本阶段只保留 2 个必做实验和 1 个可选实验：

```text
实验 1A：KV Cache 单 bit flip 是否会破坏输出
实验 1B：KV Cache 错误是否具有 decoder 传播放大效应
实验 1C：Fast Detection + Triggered Recovery 的收益上界模拟
```

其中 `1A` 和 `1B` 用于证明“为什么要保护 KV cache”，`1C` 用于连接后续 PIM/ECC 设计。

### 2. 固定实验平台

```text
Framework: vLLM
Model: Qwen/Qwen3-8B
dtype: bfloat16
enable_prefix_caching: False
temperature: 0
top_p: 1
max_tokens: 128
seed: fixed
```

配置含义：

- `Qwen/Qwen3-8B`：前期只固定一个主模型，减少变量。
- `bfloat16`：贴近 LLM 推理和 PIM 存储中的常见 KV cache 数据格式。
- `enable_prefix_caching=False`：前期只研究单请求 decoder，不引入跨请求共享。
- `temperature=0`、`top_p=1`：使用确定性 greedy decoding，便于 clean run 和 fault run 对比。
- `max_tokens=128`：足够观察错误传播，同时控制实验成本。
- `seed=fixed`：保证同一 prompt 的 clean baseline 稳定。

建议记录环境：

```text
GPU 型号
CUDA / driver 版本
vLLM 版本
PyTorch 版本
Qwen3-8B checkpoint
KV cache block_size
num_layers
num_kv_heads
head_dim
```

### 3. vLLM Fault Injection 参考实现

#### 3.1 KV cache 目标形态

vLLM 的实际 KV cache layout 会随版本和 backend 变化，但逻辑上可以按下面维度理解：

```text
KV[layer][K_or_V][block_id][token_offset][kv_head][head_dim]
```

或等价理解为单层 cache block：

```text
[2, num_blocks, block_size, num_kv_heads, head_dim]
```

其中：

- `2`：`0 = K`，`1 = V`。
- `block_id`：PagedAttention 分配的 KV block。
- `token_offset`：目标 token 在 block 内的位置。
- `kv_head`：目标 KV head。
- `head_dim`：目标 head 内的维度。

正式注入前先打印：

```text
kv_cache object type
kv_cache tensor shape
kv_cache dtype
block_size
num_layers
num_kv_heads
head_dim
request -> block table mapping
```

#### 3.2 注入位置

参考 `Bit-Flip Vulnerability of Shared KV-Cache Blocks` 的思路，动态 patch vLLM model runner。建议入口：

```text
GPUModelRunner.execute_model()
```

基本策略：

```text
1. 保存原始 execute_model()。
2. 每次调用 execute_model() 后，KV cache 已经被当前 prefill/decode step 写入。
3. 判断当前 request_id 和 step 是否命中 fault config。
4. 命中后定位目标 layer / K_or_V / block / offset / head / dim。
5. 将目标 BF16 元素 reinterpret 为 16-bit integer。
6. 用 XOR mask 翻转指定 bit。
7. 记录 old_bits / new_bits / old_value / new_value。
8. 继续执行后续 decode。
```

#### 3.3 BF16 bit flip

概念代码：

```python
def flip_bf16_bit_(kv_tensor, flat_index, bit_pos):
    raw = kv_tensor.view(torch.uint16)
    old_bits = int(raw[flat_index].item())
    raw[flat_index] ^= (1 << bit_pos)
    new_bits = int(raw[flat_index].item())
    return old_bits, new_bits
```

如果当前 PyTorch / CUDA backend 不支持直接 `torch.uint16` view，可用：

```text
方案 A：torch.int16 view，然后用 signed integer 记录 bits。
方案 B：写一个很小的 CUDA kernel，对目标地址执行 uint16 XOR。
方案 C：先在 Python 层定位 flat_index，再调用自定义 extension。
```

前期优先选择最容易跑通的方案，不追求注入框架优雅。

### 4. Fault Injection 粒度与注入强度

#### 4.1 当前默认粒度

前期实验默认采用 `single-bit injection per trial`。也就是说，每一次 fault run 只翻转一个 BF16 bit，目标粒度是：

```text
一次 decoder run
  -> 一个 layer
  -> 一个 K 或 V cache
  -> 一个 token row
  -> 一个 kv_head
  -> 一个 head_dim 元素
  -> 一个 BF16 bit
```

可以写成：

```text
KV[layer][K_or_V][block_id][token_offset][kv_head][head_dim][bit_position] ^= 1
```

因此，当前实验中的“注入率”不是硬件意义上的 `bit error rate`，而是人为控制的 `fault injection intensity`。它用于回答：

```text
如果 PIM 可访问区域中的某个 KV cache bit 已经发生错误，
这个错误是否足以改变后续 LLM 输出？
```

#### 4.2 为什么前期不用真实 bit error rate

真实硬件 `bit error rate` 需要依赖工艺、电压、温度、容量、刷新策略、ECC 状态、运行时长等因素。前期如果直接引入真实错误率，会把问题变复杂：

```text
模型敏感性
硬件错误发生概率
错误空间分布
错误时间分布
ECC 检测覆盖率
```

这些因素会混在一起，不利于先证明核心动机。因此前期采用 controlled fault injection：

```text
先固定错误已经发生，
再观察错误位置、K/V 类型、bit position 对输出的影响。
```

#### 4.3 建议的注入强度层级

前期主实验只做第一层：

| 层级 | 每个 trial 的 bit flips 数量 | 目的 | 是否前期必做 |
|---|---:|---|---|
| Single-bit | 1 | 建立 KV cache vulnerability 证据 | 必做 |
| Sparse multi-bit | 2 | 检查稀疏多点错误是否放大损害 | 可选 |
| Medium multi-bit | 4 或 8 | 观察高强度错误下是否 collapse | 后续扩展 |

论文表述上建议明确：

```text
We use controlled single-bit fault injection rather than modeling a concrete hardware BER.
The goal is to characterize the vulnerability of BF16 KV cache under sparse faults.
```

#### 4.4 后续如何扩展为注入率实验

如果单 bit 实验已经证明 KV cache 有明显脆弱性，后续可以再补一个小规模 intensity sweep：

```text
fault_count per request = 1 / 2 / 4 / 8
target space = valid KV cache elements
sampling = uniform random over selected layer/K_or_V/token/head/dim
bit_position = 14 或 {0, 14, 15}
```

但这不是前期主线。前期最关键的是先证明：

```text
即使只有一个 KV bit flip，也可能造成可观测输出损害。
```

### 5. 数据集和 Prompt

前期只准备两类输入，避免实验过重。

#### 5.1 长文本生成 Prompt

用于观察 token divergence、silent divergence、collapse。

```text
数量：30 个 prompt
长度：每个 prompt 尽量 128 tokens 以上
来源：ShareGPT / Alpaca / 自建中文或英文 instruction
输出长度：max_tokens = 128
```

示例 prompt 类型：

```text
总结一段长文本
根据背景材料回答问题
多步推理问题
代码解释问题
长 system instruction + user query
```

#### 5.2 选择题 / 短答案任务

用于得到更清晰的 accuracy drop。

```text
数量：50 - 100 条
候选：ARC-Easy / BoolQ / MMLU 子集
输出格式：要求模型只输出 A/B/C/D 或 yes/no
```

前期如果时间紧，只做长文本生成；选择题作为补充。

### 6. 评价指标

必须记录的核心指标：

```text
TCR:
  Token Change Rate，fault run 与 clean run 的 token sequence 是否不同。

TDR:
  Token Diff Ratio，单个请求中不同 token 位置的比例。

First Divergence Step:
  第一次 token mismatch 的生成位置。

Collapse Rate:
  是否出现重复、乱码、异常短输出、NaN/Inf 或明显无效输出。

ROUGE-L / BERTScore:
  文本语义相似度。

Accuracy:
  选择题或短答案任务上的正确率。
```

建议前期重点报告：

```text
TCR
mean TDR
First Divergence Step 分布
collapse rate
accuracy drop
```

### 7. 实验 1A：KV Cache 单 Bit Flip 是否会破坏输出

#### 7.1 实验目的

这是最重要的基础实验。它要回答：

> 在 Qwen3-8B 的 BF16 KV cache 中，只翻转一个 bit，是否足以改变后续生成？

如果单 bit flip 已经能造成可测输出变化，就可以直接支撑：

```text
PIM 读取 KV cache 时不能只关注计算正确性，
还必须保护 PIM 访问区域中的 KV cache 存储数据。
```

#### 7.2 变量设置

为了前期可执行，只取少量代表点：

```text
Fault target:
  prefill KV cache

Token position:
  middle prompt token

Layer:
  middle layer

K_or_V:
  K
  V

Bit position:
  bit 0   mantissa LSB，低影响参考
  bit 7   exponent LSB，中高影响
  bit 14  exponent MSB，高影响
  bit 15  sign bit，高影响

Head/dim:
  每个 prompt 随机采样 3 个 (kv_head, dim)
```

推荐 trial 数：

```text
30 prompts
2 K/V choices
4 bit positions
3 random head-dim samples
= 720 fault runs
```

如果 720 次太多，最小版本：

```text
20 prompts
2 K/V choices
3 bit positions: 0, 14, 15
2 random head-dim samples
= 240 fault runs
```

#### 7.3 具体流程

对每个 prompt：

```text
1. Clean run:
   使用固定 decoding 参数生成 y_clean，保存 token ids 和文本。

2. Fault run:
   重新运行同一个 prompt。

3. Prefill 完成后：
   定位 middle prompt token 对应的 KV cache slot。

4. 注入 bit flip：
   layer = middle layer
   K_or_V = K 或 V
   bit_position = 0 / 7 / 14 / 15
   head/dim = 随机采样

5. Continue decode:
   继续生成 128 tokens。

6. Compare:
   比较 y_fault 和 y_clean。
```

#### 7.4 输出表格

建议最终汇总成：

| Target | Bit | TCR | Mean TDR | Collapse Rate | Mean ROUGE-L | Accuracy Drop |
|---|---:|---:|---:|---:|---:|---:|
| K | 0 |  |  |  |  |  |
| K | 7 |  |  |  |  |  |
| K | 14 |  |  |  |  |  |
| K | 15 |  |  |  |  |  |
| V | 0 |  |  |  |  |  |
| V | 7 |  |  |  |  |  |
| V | 14 |  |  |  |  |  |
| V | 15 |  |  |  |  |  |

#### 7.5 预期结论

如果观察到：

- `bit 14/15` 的 TCR、TDR 或 collapse rate 明显高于 `bit 0`；
- `K` 或 `V` 任一目标出现稳定输出偏移；
- 部分输出没有崩溃但语义改变；

则可得出前期结论：

```text
BF16 KV cache bit significance 明显不均匀；
KV cache 错误可造成 silent divergence；
PIM attention 的 KV 读取路径需要 integrity protection。
```

### 8. 实验 1B：KV Cache 错误是否具有 Decoder 传播放大效应

#### 8.1 实验目的

这个实验要证明 KV cache 和普通一次性 activation 不同：

```text
KV cache 写入后会被后续多个 decode steps 反复读取。
因此一个早期 KV 错误可能影响多个未来 token。
```

这正是 PIM KV cache 保护的关键理由：PIM 不是只算一次 attention，而是在 decode 中反复从内存读取历史 K/V。

#### 8.2 变量设置

前期只测试 generated KV，不做复杂全扫描。

```text
Fault target:
  generated KV cache

K_or_V:
  V-only

Layer:
  middle layer

Bit position:
  bit 14

Injection step:
  16
  64
  96

Target generated row:
  注入当前已经生成的最后一个 token 的 V row

Head/dim:
  每个 prompt 随机采样 3 个 (kv_head, dim)
```

为什么先用 `V-only + bit 14`：

- `V` 错误直接进入 context vector，容易观察。
- `bit 14` 是 BF16 exponent 高位，扰动强。
- 固定变量后，更容易观察传播窗口差异。

推荐 trial 数：

```text
30 prompts
3 injection steps
3 random head-dim samples
= 270 fault runs
```

最小版本：

```text
20 prompts
3 injection steps
2 random head-dim samples
= 120 fault runs
```

#### 8.3 具体流程

对每个 prompt：

```text
1. Clean run:
   生成 128 tokens，保存 y_clean。

2. Fault run:
   重新运行同一个 prompt。

3. 正常完成 prefill。

4. Decode 到 injection_step:
   injection_step = 16 / 64 / 96。

5. 注入 generated KV:
   选择刚刚写入的 generated token V row。
   layer = middle layer。
   bit_position = 14。
   head/dim = 随机采样。

6. Continue decode:
   继续生成直到 128 tokens。

7. 只比较 injection_step 之后的 suffix。
```

#### 8.4 输出表格

| Injection Step | Remaining Tokens | TCR | Post-injection TDR | First Divergence After Injection | Collapse Rate |
|---:|---:|---:|---:|---:|---:|
| 16 | 112 |  |  |  |  |
| 64 | 64 |  |  |  |  |
| 96 | 32 |  |  |  |  |

#### 8.5 预期结论

如果 `injection_step=16` 比 `96` 造成更高的 post-injection TDR 或 affected token count，则可以说明：

```text
KV cache 错误不是局部一次性错误；
它会随 decoder 后续读取持续传播；
越早进入 KV cache 的错误，潜在影响窗口越大。
```

这可以直接服务后续 PIM/ECC 设计：

```text
PIM bank 中更早、后续复用次数更多的 KV tile 应具有更高检测优先级。
```


### 9. 每个 Trial 必须记录的 Metadata

```text
prompt_id
model
dtype
vLLM version
max_tokens
seed
target_type: prefill / generated
injection_step
layer_id
K_or_V
block_id
token_offset
kv_head
head_dim
bit_position
old_bits
new_bits
old_value
new_value
clean_output_token_ids
fault_output_token_ids
TCR
TDR
First Divergence Step
Collapse or not
ROUGE-L
BERTScore
Accuracy if applicable
```

### 10. 前期最小执行计划

建议按下面顺序执行：

| 阶段 | 内容 | 规模 | 目标 |
|---|---|---:|---|
| Step 1 | 跑通 clean run 和 deterministic decoding | 10 prompts | 确认 baseline 可复现 |
| Step 2 | 跑通单点 BF16 bit flip | 1 prompt, 1 fault | 确认能修改 GPU KV cache |
| Step 3 | 实验 1A 最小版 | 240 fault runs | 证明 KV bit flip 会影响输出 |
| Step 4 | 实验 1B 最小版 | 120 fault runs | 证明 decoder 传播放大效应 |

