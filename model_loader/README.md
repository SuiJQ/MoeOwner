# model_loader 模块文档

---

## 概述

纯 Python 实现的 GGUF v3 文件解析器 + PyTorch 原生量化适配层。

**无需** `llama-cpp-python`、`bitsandbytes` 或任何 C 扩展，完全依赖：
- `struct` — Little-endian 二进制解析
- `mmap` — 零拷贝文件映射
- `torch` — 张量映射与反量化运算

---

## 架构

```
model_loader/
  __init__.py       — 公共 API（load_model, GGUFFile）
  gguf_reader.py    — 底层 GGUF 文件解析 + 反量化 kernel
  model_adapter.py  — 高层模型适配器（与引擎 UnifiedScheduler 对接）
```

### 4 项零成本优化（已默认启用）

| # | 优化 | 启用位置 |
|---|------|----------|
| 1 | Flash Attention SDPA | `OptimisationContext` / `torch.backends.cuda.enable_flash_sdp` |
| 2 | torch.compile | `GGUFFlashAttention.forward` 加上 `@torch.compile` |
| 3 | TF32 精度 | `torch.set_float32_matmul_precision("high")` + cuDNN |
| 4 | 动态 Block Size | `GGUFModelAdapter.suggest_block_size()` |

---

## 快速开始

```python
from model_loader import load_model

# 加载 GGUF 模型（自动反量化权重 → FP16 CUDA）
model = load_model("deepseek-r1-distill-qwen-7b.Q4_0.gguf", device="cuda")

# 推理
import torch
input_ids = torch.tensor([[1, 2, 3]], device="cuda")
logits = model.forward(input_ids)
next_token = logits[:, -1, :].argmax(dim=-1)
```

---

## 支持的 GGML 类型

| 类型 | 常量 | 状态 | 反量化方式 |
|------|------|------|-----------|
| F32 | 0 | ✅ 零拷贝 | `torch.frombuffer(float32)` |
| F16 | 1 | ✅ 零拷贝 | `torch.frombuffer(float16)` |
| Q4_0 | 2 | ✅ 纯 PyTorch | 位移解包 → FP16 |
| Q8_0 | 8 | ✅ 纯 PyTorch | INT8 缩放 → FP16 |
| Q4_K_M | 18 | 🔧 待扩展 | 需 K-quant 解包 |
| 其他 | — | 🔧 待扩展 | — |

---

## 低级别用法

```python
from model_loader.gguf_reader import open_gguf, load_tensor

# 打开 GGUF 文件
gguf = open_gguf("model.gguf")
print(f"Version: {gguf.version}")
print(f"Metadata: {gguf.metadata}")
print(f"Tensors: {list(gguf.tensors.keys())[:5]}...")

# 加载单个张量
w = load_tensor(gguf, "blk.0.attn_q.weight", device="cuda")
print(w.shape, w.dtype)
```

---

## 与引擎集成

```bash
# 使用 GGUF 模型启动引擎
python main.py --gguf /path/to/model.Q4_0.gguf

# 或通过 --model 自动检测 .gguf 后缀
python main.py --model /path/to/model.Q4_0.gguf

# 手动指定 block-size（推荐 32 对于 >7B 模型）
python main.py --gguf model.gguf --block-size 32
```

---

## 精度

| 加载方式 | 精度 | 显存 vs FP16 |
|----------|------|-------------|
| F16/F32 原生 | 无损 | 100% |
| Q8_0 → FP16 | <0.1% 损失 | ~50% |
| Q4_0 → FP16 | <1% 损失 | ~25% |

反量化后的权重以 FP16 存储于 GPU，推理速度与原生 FP16 模型一致。
