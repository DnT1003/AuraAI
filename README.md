# 🌌 Aura AI — Custom Decoder LLM with SOTA Inference Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.2%2B-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Training-Kaggle%20T4-blueviolet?logo=kaggle" />
</p>

> **Aura AI** là một mô hình ngôn ngữ Decoder-Only được xây dựng từ đầu **(from scratch)** với kiến trúc lai tiên tiến, tích hợp các kỹ thuật SOTA về tăng tốc inference và tiết kiệm VRAM: **BitNet 1.58-bit · MLA Attention · DeepSeek MoE · QLoRA · EAGLE Speculative Decoding · TurboQuant KV Cache · PagedAttention**.

---

## ✨ Kiến Trúc Nổi Bật

| Thành phần | Kỹ thuật | Ý nghĩa |
|---|---|---|
| **BitLinear** | BitNet 1.58-bit (STE) | Trọng số {−1, 0, 1}; giảm cực mạnh dung lượng inference |
| **LoRALinear** | QLoRA / PEFT | Đóng băng base, chỉ train adapter nhỏ — tiết kiệm >99% VRAM |
| **MultiHeadLatentAttention** | MLA (DeepSeek V2) | Nén KV qua Low-Rank latent space, giảm KV footprint 4–10× |
| **DeepSeekMoE** | Sparse MoE + Shared Expert | 64 routed experts, mỗi token chọn top-6; Load Balancing Loss |
| **RMSNorm + RoPE** | Llama-style normalization | Ổn định gradient + positional encoding không tham số |
| **EagleHead** | Speculative Decoding (EAGLE) | Draft token nhanh O(1) từ hidden state, verify song song bằng base model |
| **TurboQuant** | PolarQuant (WHT + 4-bit) | Nén KV Cache xuống 4-bit với Walsh-Hadamard Transform, chống outlier |
| **PagedAttentionEngine** | vLLM-style Paged KV | Quản lý bộ nhớ dạng block vật lý, Sparse V, Radix Tree Prefix Caching |
| **SubAgentDispatcher** | OS-Level Swarm | Multi-agent song song qua IPC Mailbox trên ổ cứng (Windows) |

---

## 📁 Cấu Trúc Dự Án

```
AuraAI/
├── src/
│   ├── config.py           # ModelConfig dataclass — toàn bộ siêu tham số
│   ├── model.py            # SotaDecoderCausalLM — kiến trúc chính (BitNet, MLA, MoE)
│   ├── generate.py         # EagleHead + EagleSpeculativePipeline
│   ├── engine.py           # PagedAttentionEngine (TurboQuant V2+, Prefix Caching)
│   ├── turboquant.py       # TurboQuant: PolarQuant KV Cache Compression
│   ├── cache.py            # Radix Tree (Prefix Caching) implementation
│   ├── dataset.py          # DataLoader cho văn bản thô (.txt / .jsonl)
│   ├── train.py            # Script train cơ bản (local)
│   ├── swarm.py            # SubAgentDispatcher — OS-Level Multi-Agent Swarm
│   ├── mailbox.py          # IPC Mailbox (giao tiếp liên process qua file)
│   └── agent_worker.py     # Worker process chạy trong cửa sổ terminal độc lập
├── train_kaggle.py         # 🚀 Entry-point train trên Kaggle (QLoRA + EAGLE, ~1.2B)
├── prepare_dataset.py      # Tải dataset từ HuggingFace Hub
├── test_all.py             # Test tích hợp toàn bộ module
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🚀 Huấn Luyện Trên Kaggle (GPU T4/P100)

### Bước 1 — Clone repo trong Notebook

```python
!git clone https://github.com/DnT1003/AuraAI.git
%cd AuraAI
!pip install -r requirements.txt -q
```

### Bước 2 — Chuẩn bị dữ liệu

```python
!python prepare_dataset.py
```

Hoặc dùng file `.txt` / `.jsonl` của bạn và truyền vào `--data_path`.

### Bước 3 — Chạy training

```python
!python train_kaggle.py \
    --data_path "/kaggle/input/your-dataset/data.txt" \
    --epochs 3 \
    --batch_size 2 \
    --lr 2e-4 \
    --save_dir "/kaggle/working/weights"
```

> **Cấu hình mặc định** (`train_kaggle.py`) đã được tối ưu cho **12 GB VRAM** (RTX 3060 / Kaggle T4):
> - Base Model: **~1.2B tham số** (đóng băng hoàn toàn)
> - Trainable: **~160M tham số** (LoRA adapters + EAGLE Head)
> - Ước tính VRAM: **7–9 GB**

**Checkpoints được lưu tại:**
```
/kaggle/working/weights/base_model.pt   # LoRA-merged base weights
/kaggle/working/weights/eagle_head.pt   # EAGLE speculative head
```

---

## ⚙️ Cấu Hình (ModelConfig)

File `src/config.py` chứa toàn bộ siêu tham số. Một số giá trị quan trọng:

```python
from config import ModelConfig

config = ModelConfig(
    vocab_size=100277,          # tiktoken cl100k_base
    hidden_size=5120,           # Chiều embed (full-size model)
    num_hidden_layers=48,       # Số Transformer blocks
    num_attention_heads=40,     # Số head MLA
    q_lora_rank=1536,           # Rank nén Query latent
    kv_lora_rank=512,           # Rank nén KV latent (giảm 4–10× footprint)
    n_routed_experts=64,        # Số lượng routed experts
    num_experts_per_tok=6,      # Top-K routing mỗi token
    use_bitnet=False,           # Bật BitNet 1.58-bit (train from scratch)
    use_qlora=True,             # Bật QLoRA (fine-tuning)
    peft_lora_rank=16,          # Rank của LoRA adapter
)
```

---

## 🧠 Chi Tiết Kỹ Thuật

### BitNet 1.58-bit
Trọng số tất cả các linear layer được lượng tử hóa về `{−1, 0, 1}` trong quá trình forward, sử dụng **Straight-Through Estimator (STE)** để truyền gradient. Activation được lượng tử hóa xuống Int8 absmax trước khi nhân ma trận.

### Multi-Head Latent Attention (MLA)
Thay vì lưu toàn bộ KV states, MLA nén K/V xuống không gian latent low-rank (`kv_lora_rank = 512`), giảm KV cache footprint **4–10 lần** so với GQA/MHA thông thường. Query cũng được nén tương tự với `q_lora_rank`.

### DeepSeek MoE
- **64 Routed Experts** + **2 Shared Experts** (luôn bật cho mọi token)
- Mỗi token được định tuyến tới **top-6 experts**
- **Load Balancing Loss** ngăn model tập trung vào 1 expert

### EAGLE Speculative Decoding
`EagleHead` học dự đoán **hidden state tương lai** từ `(current_hidden, token_embedding)` với chi phí O(1). Base model verify song song toàn bộ draft tokens trong **1 forward pass**, tăng throughput **2–5×** mà không mất chất lượng.

### TurboQuant (PolarQuant)
Áp dụng **Walsh-Hadamard Transform (WHT)** + Random Sign Flips để phân phối hóa Gaussian, sau đó **Scalar Quantization 4-bit** (Lloyd-Max). KV Cache được nén không đối xứng: `K → fp16`, `V → turbo4`. **Sparse V** bỏ qua dequantize với tokens có attention weight < 1e-6.

### PagedAttention + Prefix Caching
Quản lý KV Cache bằng **block vật lý** (tương tự vLLM). **Radix Tree** chia sẻ block prefix giữa các sequence, tránh recompute prompt dùng lại.

---

## 🤖 Sub-Agent Swarm

`SubAgentDispatcher` spawn các process Windows độc lập (mỗi process là 1 agent chuyên trách):

| Agent | Vai trò |
|---|---|
| `SearchExpert` | Tìm kiếm và thu thập thông tin |
| `MathExpert` | Giải quyết bài toán số học / logic |
| `CodeReviewer` | Review và phân tích code |
| `Synthesizer` | Tổng hợp kết quả từ các agent khác |

Các agent giao tiếp qua **IPC Mailbox** (file-based, thư mục `.aura_teams/`).

```python
from src.swarm import SubAgentDispatcher
swarm = SubAgentDispatcher()
result = swarm.run_swarm("Hãy phân tích và tối ưu hàm fibonacci sau đây...")
print(result)
```

---

## 📦 Cài Đặt (Local)

```bash
# Clone repo
git clone https://github.com/DnT1003/AuraAI.git
cd AuraAI

# Tạo venv và cài dependencies
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

**Requirements:**
```
torch>=2.2.0
transformers>=4.38.0
tiktoken>=0.6.0
accelerate>=0.27.0
datasets>=2.17.0
wandb>=0.16.3
numpy>=1.26.4
```

---

## 🧪 Tests

```bash
python test_all.py
```

Kiểm tra tích hợp toàn bộ: model forward, MoE routing, EAGLE pipeline, TurboQuant compress/decompress, PagedAttention.

---

## 📊 Thông Số Model Mặc Định (Kaggle Config)

| Tham số | Giá trị |
|---|---|
| Số layers | 12 |
| Hidden size | 1536 |
| Attention heads | 12 |
| Routed experts | 8 (top-2) |
| Shared experts | 2 |
| Vocab size | 100,277 |
| Max sequence length | 512 |
| Tổng tham số base | ~1.2B |
| Trainable (LoRA + Eagle) | ~160M |

---

## 📄 License

MIT License © 2026 DnT1003

---

<p align="center">
  <i>Built with ❤️ from scratch — No HuggingFace architectures, pure PyTorch.</i>
</p>
