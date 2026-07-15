# Pure Python Inference Engine

> **Hybrid PagedAttention + RadixAttention KV Cache Engine**  
> 纯 Python 实现的大模型推理引擎，融合 PagedAttention 的物理块管理与 RadixAttention 的哈希前缀匹配。

---

## 项目概述

本项目是一个**教学性推理引擎参考实现**，展示了如何将两个主流 KV Cache 管理策略融入一个统一的调度框架：

| 组件 | 文件 | 功能 |
|------|------|------|
| 🔥 **FlashAttention 核** | `attention_kernel.py` | 基于 PyTorch SDPA + `torch.compile` 的优化注意力 |
| 📦 **混合缓存** | `cache_manager.py` | PagedAttention 物理块 + RadixAttention 哈希索引 |
| ⏱ **统一调度器** | `scheduler.py` | Chunked Prefill + Decode 双 CUDA 流调度 |
| 🚀 **入口** | `main.py` | 全局配置、模型注入、事件循环 |

### 核心创新点

- **增量 SHA-256 哈希链**：严格 `SHA256(SHA256(prev).digest() + token_bytes)`，非 `hashlib.update()`，保证 Radix 树可匹配任意前缀
- **显存感知容量计算**：`total_blocks = int(free_mem * 0.85 / (block_size * hidden_size * 4))`
- **复合键守卫 GC**：防止哈希重用导致的误删
- **双 CUDA 流管线**：Prefill 流 + Decode 流，主线程统一同步（防死锁）
- **引用计数驱逐**：每个匹配的自增引用，ref_count=0 时回收至空闲队列

---

## 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.12+ | 推荐 3.12 |
| PyTorch | 2.6.0+cu124 | CUDA 12.4 |
| Transformers | 4.51.3 | HuggingFace 模型加载 |

```bash
# 安装
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.51.3
```

> **注**：`cache_manager.py` 无需 GPU 即可运行单元测试（无 CUDA 时回退至 10000 块）

---

## 快速开始

### 1. 运行单元测试

```bash
cd pure-python-engine
python3 -c "
from cache_manager import HybridCache

cache = HybridCache(block_size=16, hidden_size=4096, total_blocks=512)

# 基本分配
b1 = cache.allocate([101, 102, 103])
print(f'Allocated block {b1.block_id}')

# 前缀匹配
matched_id, remaining = cache.match_prefix([101, 102])
print(f'Matched block {matched_id}, remaining: {remaining}')

# 引用计数与回收
cache.free_block(b1.block_id)
print(f'Cache stats: {cache.stats()}')
"
```

### 2. 启动推理服务

```bash
python main.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --block-size 16 \
  --hidden-size 4096 \
  --verbose
```

---

## 架构设计

### 阶段流程

```
阶段 1: 环境锁定
  └─ Python 3.12 + torch 2.6 + transformers 4.51

阶段 2: 全局 Torch 配置
  ├─ Flash SDP 强制启用
  ├─ TF32 matmul 允许
  └─ float32 精度 = 'high'

阶段 3: 模块组装
  ├─ attention_kernel.py ─── FlashAttentionKernel (torch.compile)
  ├─ cache_manager.py ────── HybridCache (Paged + Radix)
  └─ scheduler.py ────────── UnifiedScheduler (双流管线)

阶段 4: 模型注入
  ├─ 加载 HF 模型 (fp16)
  ├─ 替换每层 self_attn → FlashAttentionKernel
  └─ 预热编译

阶段 5: 事件循环
  └─ 无限调度: step() → 预填充/解码/同步/GC
```

### 缓存结构

```
┌─────────────────────────────────────────────────────────────┐
│                    HybridCache                              │
│                                                             │
│  ┌─────────────────────┐    ┌───────────────────────────┐   │
│  │  free_block_queue   │    │     radix_index            │   │
│  │  (LIFO 空闲池)      │    │  hash(t1)        → block  │   │
│  │                     │    │  hash(t1+t2)     → block  │   │
│  │  [0, 1, 2, ...]     │    │  hash(t1+t2+t3)  → block  │   │
│  └─────────────────────┘    └───────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  allocated_blocks: { block_id → Block }              │   │
│  │  Block { phys_addr, ref_count, hash, next_block }   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 双流管线

```
 时间线 →
┌─────────┐    ┌─────────┐    ┌─────────┐
│ Prefill │    │ Prefill │    │ Prefill │  ← prefill_stream
│ Chunk 1 │    │ Chunk 2 │    │ Chunk 3 │
└─────────┘    └────┬────┘    └─────────┘
                     │
               ┌─────▼──────┐
               │  Decode 1  │                  ← decode_stream
               └────────────┘

  同步点: torch.cuda.synchronize()  ← 主线程 (仅此处)
```

---

## 测试

```bash
python3 -c "exec(open('cache_manager.py').read().split('if __name__')[0])"  # 语法验证

# 运行完整测试（见上方代码清单）
```

当前 `cache_manager.py` 通过 **18 项**自动化测试：
- ✅ 块分配与空闲队列管理
- ✅ 增量哈希链一致性
- ✅ Radix 前缀匹配（精确/部分/无匹配）
- ✅ 引用计数自增与递减
- ✅ 块驱逐与空闲回收
- ✅ GC 过期条目清理
- ✅ OOM 异常处理
- ✅ 复合键守卫防误删

---

## 注意事项

⚠️ **生产部署关键路径**：

1. **`past_key_values` 集成**：当前调度器的 `model.forward()` 调用中的 `past_key_values` 是占位实现。需实现自定义 `DynamicCache` 子类，从 `HybridCache` 的物理块池读写 KV 张量
2. **解码路径**：`scheduler.step()` 中的解码路径目前是一个生命周期钩子（`decode_req.step()`），实际的 `model.forward()` 调用需补充
3. **CUDA 图捕获**：`torch.compile(mode="reduce-overhead")` 在首次运行时会有编译开销

---

## 许可证

**CC BY-NC-SA 4.0**（署名-非商业性使用-相同方式共享 4.0 国际）

- ✅ **学习研究** — 欢迎
- ✅ **修改分发** — 允许，但须以相同协议共享
- ❌ **商业使用** — 禁止
- ✅ **贡献代码** — 提交者自动授权项目使用

**完整许可文本见 [LICENSE](./LICENSE)**
