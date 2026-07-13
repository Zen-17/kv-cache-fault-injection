# vLLM 与 Qwen3-8B 使用说明

本文档记录当前服务器上的 vLLM 环境、相关目录，以及使用 RTX 4090 运行 Qwen3-8B 的完整步骤。

## 1. 当前硬件与软件环境

| 项目 | 当前配置 |
| --- | --- |
| 操作系统 | Linux x86_64 |
| GPU | NVIDIA GeForce RTX 4090，24 GB 显存 |
| NVIDIA 驱动 | 535.146.02 |
| 驱动报告的 CUDA 版本 | CUDA 12.2 |
| 本地 CUDA Toolkit | CUDA 12.1 |
| Conda 环境 | `vllm0.8.5` |
| Python | 3.10 |
| vLLM | `0.8.5.post1` 源码 editable 安装，复用 cu121 预编译扩展 |
| PyTorch | `2.6.0+cu124` |
| Transformers | `4.51.3` |
| 模型 | Qwen3-8B，BF16 |

该组合已在当前服务器完成真实 GPU 推理验证。虽然 PyTorch 自带的运行时是 CUDA 12.4，但当前 535 驱动可通过 CUDA 12.x 次版本兼容机制正常运行。

## 2. 相关目录

| 内容 | 路径 |
| --- | --- |
| Anaconda 安装目录 | `/opt/data/data/anaconda3` |
| vLLM 0.8.5 Conda 环境 | `/opt/data/data/anaconda3/envs/vllm0.8.5` |
| Qwen3-8B 模型 | `/opt/data/data/models/Qwen3-8B` |
| vLLM 0.8.5 源码与实验目录 | `/opt/data/data/workspace-vllm` |
| vLLM 0.8.5 CUDA 12.1 wheel | `/opt/data/data/workspace-vllm/wheels/vllm-0.8.5.post1+cu121-cp38-abi3-manylinux1_x86_64.whl` |
| Qwen3-8B 源码运行 demo | `/opt/data/data/workspace-vllm/examples/offline_inference/qwen3_8b_source_demo.py` |
| 基线实验结果目录 | `/opt/data/data/workspace-vllm/experiments/results` |
| 本说明文档 | `/opt/data/data/workspace-vllm/VLLM_QWEN3_8B_GUIDE.md` |

`/opt/data/data/workspace-vllm` 当前检出 `v0.8.5.post1`，实验分支为 `experiments-v0.8.5`。Conda 环境已通过 editable 模式连接该源码目录，因此修改 Python 源码后不需要重新安装，重新启动推理进程即可生效。CUDA 扩展来自上表中的预编译 wheel。

## 3. 激活环境

```bash
source /opt/data/data/anaconda3/bin/activate
conda activate vllm0.8.5
cd /opt/data/data/workspace-vllm
```

检查版本和 GPU：

```bash
python -c "import vllm, torch, transformers; \
print('vLLM:', vllm.__version__); \
print('PyTorch:', torch.__version__); \
print('Transformers:', transformers.__version__); \
print('CUDA available:', torch.cuda.is_available()); \
print('GPU:', torch.cuda.get_device_name(0))"
```

预期可看到：

```text
vLLM: 0.8.5.post1
PyTorch: 2.6.0+cu124
Transformers: 4.51.3
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

## 4. 使用 Python 运行 Qwen3-8B

### 推荐：运行可交付 demo

该 demo 会验证 vLLM 是否从当前源码目录导入，加载本地 Qwen3-8B，输出生成文本和 token IDs，并可将完整运行信息保存为 JSON。

```bash
source /opt/data/data/anaconda3/bin/activate
conda activate vllm0.8.5
cd /opt/data/data/workspace-vllm

PYTHONUNBUFFERED=1 python \
  examples/offline_inference/qwen3_8b_source_demo.py \
  --prompt "请用简洁的语言介绍一下人工智能。" \
  --max-tokens 64 \
  --output-json experiments/results/qwen3_baseline.json
```

demo 默认采用适合错误注入基线的配置：

- `seed=42`
- `temperature=0`
- `enforce_eager=True`
- `enable_prefix_caching=False`
- `max_num_seqs=1`
- 关闭 Qwen3 thinking 模式
- 关闭 V1 Engine 额外子进程，避免源码被重复导入，并方便观察 Python 注错 hook
- 默认忽略 EOS，严格生成 `--max-tokens` 指定的 token 数

常用参数：

```bash
python examples/offline_inference/qwen3_8b_source_demo.py --help

# 允许模型遇到 EOS 后提前停止
python examples/offline_inference/qwen3_8b_source_demo.py \
  --max-tokens 64 \
  --allow-early-stop

# 开启 Qwen3 thinking 模式
python examples/offline_inference/qwen3_8b_source_demo.py \
  --max-tokens 256 \
  --enable-thinking
```

已完成的源码实测结果：

```text
vLLM source: /opt/data/data/workspace-vllm/vllm/__init__.py
模型权重显存占用: 15.2683 GiB
Prompt tokens: 19
Generated tokens: 32
Generation time: 1.227 s
Throughput: 26.07 token/s
QWEN3_8B_SOURCE_DEMO=OK
```

结构化结果保存在：

```text
/opt/data/data/workspace-vllm/experiments/results/qwen3_baseline.json
```

该 JSON 包含环境版本、源码路径、prompt、生成配置、输入/输出 token IDs、耗时和吞吐量，后续可直接作为 KV Cache 错误注入实验的 golden baseline。

### 最小内联示例

下面的示例输入一个中文 prompt，并严格生成 64 个新 token：

```bash
python - <<'PY'
from vllm import LLM, SamplingParams

MODEL_PATH = "/opt/data/data/models/Qwen3-8B"
PROMPT = "请用简洁的语言介绍一下人工智能。"
OUTPUT_TOKEN_COUNT = 64

llm = LLM(
    model=MODEL_PATH,
    dtype="bfloat16",
    max_model_len=2048,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
)

tokenizer = llm.get_tokenizer()
messages = [{"role": "user", "content": PROMPT}]
formatted_prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

result = llm.generate(
    [formatted_prompt],
    SamplingParams(
        max_tokens=OUTPUT_TOKEN_COUNT,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        ignore_eos=True,
    ),
)[0].outputs[0]

print("输入：", PROMPT)
print("输出：", result.text)
print("实际生成 token 数：", len(result.token_ids))
print("Token IDs：", result.token_ids)
PY
```

参数说明：

- `max_tokens=64`：最多生成 64 个新 token。
- `ignore_eos=True`：忽略模型的结束符，从而严格生成 64 个 token。
- 如果希望模型自然结束，将其改为 `ignore_eos=False`；此时生成数量可能少于 64。
- `enable_thinking=False`：关闭 Qwen3 的思考模式，只输出普通回答。
- `max_model_len=2048`：限制最大上下文，减少 KV Cache 显存占用。
- `enforce_eager=True`：使用 eager 模式，减少 CUDA Graph 带来的额外复杂性，适合首次验证。

当前机器上的实测结果：

```text
模型权重显存占用：约 15.27 GiB
生成速度：约 25 token/s
实际输出：64 token
```

## 5. 启动 OpenAI 兼容 API

启动服务：

```bash
conda activate vllm0.8.5
cd /tmp

vllm serve /opt/data/data/models/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 \
  --port 8000
```

服务启动后，在另一个终端发送请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-8B",
    "messages": [
      {
        "role": "user",
        "content": "请用简洁的语言介绍一下人工智能。"
      }
    ],
    "max_tokens": 64,
    "temperature": 0.6,
    "top_p": 0.95
  }'
```

查看服务中的模型：

```bash
curl http://127.0.0.1:8000/v1/models
```

使用 `Ctrl+C` 停止服务。

## 6. 启用 Qwen3 思考模式

Python 调用中，将聊天模板参数改为：

```python
enable_thinking=True
```

启动 API 服务时可以启用推理内容解析：

```bash
vllm serve /opt/data/data/models/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --dtype bfloat16 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.90 \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --port 8000
```

思考模式会消耗更多输出 token，应相应提高 `max_tokens`。

## 7. 环境复现命令

如需重新创建环境：

```bash
source /opt/data/data/anaconda3/bin/activate
conda create -n vllm0.8.5 python=3.10 -y
conda activate vllm0.8.5

pip install \
  "/opt/data/data/workspace-vllm/wheels/vllm-0.8.5.post1+cu121-cp38-abi3-manylinux1_x86_64.whl"

pip install \
  "transformers==4.51.3" \
  "tokenizers==0.21.4" \
  "huggingface-hub==0.36.0" \
  "setuptools-scm>=8.0"

cd /opt/data/data/workspace-vllm
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION="/opt/data/data/workspace-vllm/wheels/vllm-0.8.5.post1+cu121-cp38-abi3-manylinux1_x86_64.whl" \
pip install --no-build-isolation -e .
```

检查依赖：

```bash
pip check
python -c "import vllm, vllm._C; print(vllm.__version__); print(vllm.__file__)"
```

第二条命令输出的源码路径应为：

```text
/opt/data/data/workspace-vllm/vllm/__init__.py
```

### 修改源码时的生效方式

- 修改 `.py` 文件：editable 安装会直接读取工作目录中的源码，重启 Python 或 vLLM 服务即可生效。
- 修改 C++/CUDA 文件：预编译扩展不会包含修改，需要重新编译。
- 当前本地 CUDA Toolkit 是 12.1，而 PyTorch 2.6.0 使用 CUDA 12.4；如需编译 C++/CUDA 扩展，应先安装 CUDA 12.4 Toolkit，并让 `CUDA_HOME` 指向它。
- 开始实验前可用 `git status` 查看改动；当前实验分支为 `experiments-v0.8.5`。

## 8. 常见问题

### CUDA 显存不足

优先降低上下文长度：

```text
--max-model-len 1024
```

同时确认 GPU 上没有其他进程：

```bash
nvidia-smi
```

### 首次启动较慢

模型包含约 16 GB 权重，当前实测五个权重分片加载约 20–35 秒。除此之外，源码和 Conda 环境位于 `/opt/data/data` 共享存储，大量 Python 小文件的冷启动读取可能需要 10–20 分钟。日志长时间没有更新不一定代表死锁，可通过以下命令判断：

```bash
# 查看进程是否存在
pgrep -af qwen3_8b_source_demo.py

# 查看是否已经开始占用 GPU
nvidia-smi
```

只要 Python 进程仍有 CPU/I/O 活动，就应继续等待。demo 已关闭 V1 的额外引擎子进程，避免第二次导入源码。模型成功启动后，实际生成 32 token 只需约 1.2 秒。

### FlashInfer 未安装

日志可能出现：

```text
FlashInfer is not available. Falling back to the PyTorch-native implementation
```

这只是性能提示，不影响正常推理。当前环境已经使用 Flash Attention 后端。

### 指定 token 数量

- 自然生成：设置 `max_tokens=N` 和 `ignore_eos=False`，最多生成 N 个 token。
- 严格数量：设置 `max_tokens=N` 和 `ignore_eos=True`，生成恰好 N 个 token。
- `max_tokens` 只计算新生成的 token，不包含输入 prompt。

## 9. KV Cache Bit Error 实验基础

### 9.1 当前环境是否支持

当前环境支持在 Python 层对 GPU KV Cache 进行单 bit 翻转：

- vLLM 源码通过 editable 模式安装，修改 `.py` 文件后重启进程即可生效。
- Qwen3-8B 在 RTX 4090 上使用 V1 Engine 和 FlashAttention 2。
- demo 使用 `enforce_eager=True`，Python hook 不会被完整 CUDA Graph 绕过。
- CUDA 扩展来自预编译 wheel，因此可以在 CUDA 写入完成后修改 KV tensor，但不能直接修改预编译 CUDA kernel 的内部行为。

### 9.2 Qwen3-8B KV Cache 结构

Qwen3-8B 共有 36 层，使用 GQA：

```text
Attention heads: 32
KV heads: 8
Head dimension: 128
KV dtype: BF16
Block size: 16
```

每层 FlashAttention KV Cache 的布局为：

```text
[2, num_blocks, 16, 8, 128]
```

第一维 `0` 表示 K，`1` 表示 V。其余维度依次是物理 block、block 内 token、KV head 和 head dimension。

BF16 共 16 bit：

```text
bit 0–6:  尾数
bit 7–14: 指数
bit 15:   符号
```

建议先从尾数 bit 0–6 开始实验。指数位更容易产生极端数值、Inf 或 NaN。

### 9.3 推荐注错位置

主要修改文件：

```text
vllm/v1/attention/backends/flash_attn.py
```

目标函数：

```text
FlashAttentionImpl.forward
```

推荐在以下两个调用之间注错：

```python
torch.ops._C_cache_ops.reshape_and_cache_flash(...)

# 在这里对 key_cache 或 value_cache 进行 bit flip。

flash_attn_varlen_func(...)
```

此时 K/V 已写入显存、即将被 Attention 读取，最接近“KV Cache 存储发生软错误”的语义，而且不需要重新编译 CUDA。

当前 token 对应的物理位置可通过 `slot_mapping` 得到：

```python
slot = int(attn_metadata.slot_mapping[token_index].item())
block_index = slot // 16
block_offset = slot % 16

target = key_cache[block_index, block_offset, kv_head, dim]
```

BF16 单 bit 翻转的核心逻辑：

```python
integer_view = target.view(torch.int16)
mask = 1 << bit
if mask >= 32768:
    mask -= 65536
integer_view.bitwise_xor_(mask)
```

必须增加 one-shot 状态保护。否则每个 decode step 对相同位置重复 XOR，可能把错误 bit 翻回原值。

### 9.4 建议的注错配置

建议把实验参数放在独立 JSON 中，而不是硬编码：

```json
{
  "enabled": true,
  "seed": 42,
  "mode": "single",
  "phase": "decode",
  "decode_step": 1,
  "layer": 17,
  "kv": "K",
  "kv_head": 3,
  "dim": 64,
  "bit": 6
}
```

建议通过环境变量指定配置：

```bash
export VLLM_KV_FAULT_CONFIG=/path/to/kv_fault_config.json
```

推荐新增但尚未实现的文件结构：

```text
vllm/kv_fault_injection.py
experiments/configs/kv_fault_config.json
experiments/results/qwen3_baseline.json
experiments/results/qwen3_fault_layer17_k_bit6.json
```

### 9.5 标准实验流程

1. 确认 GPU 空闲：

   ```bash
   nvidia-smi
   ```

2. 固定源码版本和记录改动：

   ```bash
   cd /opt/data/data/workspace-vllm
   git branch --show-current
   git describe --tags --exact-match HEAD
   git status --short
   ```

3. 禁用注错，生成 baseline：

   ```bash
   unset VLLM_KV_FAULT_CONFIG
   PYTHONUNBUFFERED=1 python \
     examples/offline_inference/qwen3_8b_source_demo.py \
     --prompt "请用简洁的语言介绍一下人工智能。" \
     --max-tokens 64 \
     --output-json experiments/results/qwen3_baseline.json
   ```

4. 启用单次注错，保持 prompt、seed、token 数和所有推理参数不变：

   ```bash
   export VLLM_KV_FAULT_CONFIG="$PWD/experiments/configs/kv_fault_config.json"
   PYTHONUNBUFFERED=1 python \
     examples/offline_inference/qwen3_8b_source_demo.py \
     --prompt "请用简洁的语言介绍一下人工智能。" \
     --max-tokens 64 \
     --output-json experiments/results/qwen3_fault_layer17_k_bit6.json
   ```

5. 比较 baseline 与 fault run：

   - 首个不同 token 的位置。
   - Token edit distance。
   - 最终文本是否一致。
   - KV 翻转前后的十六进制值。
   - 是否出现 NaN/Inf。
   - 不同 layer、K/V、head 和 bit 的敏感度。

### 9.6 实验日志至少记录

```text
run_id
源码 commit
prompt 与 prompt token IDs
sampling seed
layer
prefill/decode 阶段和 decode step
K/V
block index 和 block offset
KV head 和 dimension
bit
翻转前后的 BF16 十六进制值
输出 token IDs
首个分叉 token 位置
是否出现 NaN/Inf
```

### 9.7 Python 注错与 CUDA 内核注错的边界

当前环境无需重新编译即可完成：

- KV 写入后、Attention 读取前的 bit flip。
- 指定 layer、K/V、KV head、dimension 和 bit。
- 单次、持续或按概率注错。
- prefill 与 decode 阶段区分。

以下修改需要重新编译 CUDA：

```text
csrc/cache_kernels.cu
reshape_and_cache_flash_kernel
FlashAttention FA2 paged KV 读取内核
```

当前本地 CUDA Toolkit 为 12.1，而 PyTorch 2.6.0 使用 CUDA 12.4。若要修改并编译 CUDA 内核，应先安装 CUDA 12.4 Toolkit，并设置正确的 `CUDA_HOME`。在此之前，建议先完成 Python 写后读前注错，其实验语义清晰且实现成本最低。

## 10. 同事交付检查清单

- [ ] `conda activate vllm0.8.5` 成功。
- [ ] `git describe --tags --exact-match HEAD` 输出 `v0.8.5.post1`。
- [ ] 当前分支是 `experiments-v0.8.5` 或基于它创建的实验分支。
- [ ] `import vllm` 指向 `/opt/data/data/workspace-vllm/vllm/__init__.py`。
- [ ] `import vllm._C` 成功。
- [ ] `pip check` 输出 `No broken requirements found`。
- [ ] 本地模型目录 `/opt/data/data/models/Qwen3-8B` 完整。
- [ ] demo 输出 `QWEN3_8B_SOURCE_DEMO=OK`。
- [ ] baseline JSON 已保存并纳入实验记录。
- [ ] 注错实验固定 prompt、seed、模型参数和输出 token 数。
- [ ] 每次运行记录 Git commit、注错配置和输出 token IDs。

