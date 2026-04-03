from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # --- Cấu hình Mặc định (Base Config) ---
    vocab_size: int = 100277
    hidden_size: int = 5120
    num_hidden_layers: int = 48
    
    # --- Cơ chế lai Lượng tử hóa (Hybrid Quantization) ---
    # Nếu True: Model tự động dựng mạng bằng BitLinear (-1, 0, 1) để train từ đầu.
    # Nếu False: Sử dụng Linear tiêu chuẩn (chuẩn bị cho NF4, AWQ khi load inference).
    use_bitnet: bool = False 
    
    # --- QLoRA (Quantized LoRA Tuning) ---
    # Bật cờ này để đóng băng Model Base, chỉ trượt ma trận LoRA A và B (Tiết kiệm 99% VRAM khi train)
    use_qlora: bool = False
    peft_lora_rank: int = 8
    peft_lora_alpha: float = 16.0
    peft_lora_dropout: float = 0.05
    
    
    # --- V2+ Cấu hình Mở rộng Inference (TurboQuant & Caching) ---
    kv_cache_type_k: str = "fp16"   # Theo Asymmetric config: K giữ nguyên gốc
    kv_cache_type_v: str = "turbo4" # V nén PolarQuant 4-bit cực hạn
    enable_sparse_v: bool = True    # Bỏ qua giải mã V dựa trên Attention Mask
    enable_prefix_caching: bool = True # Bật Radix Tree cho PagedAttention
    
    # --- Multi-Head Latent Attention (MLA) ---
    # Thay vì dùng num_key_value_heads của GQA/MQA, MLA nén KV thành biến trung gian
    num_attention_heads: int = 40
    q_lora_rank: int = 1536     # Không gian nén của Query
    kv_lora_rank: int = 512     # Không gian nén của Key/Value (Giảm ít nhất 4->10 lần footprint)
    qk_nope_head_dim: int = 128 # Kích thước vector Query/Key không dùng RoPE
    qk_rope_head_dim: int = 64  # Kích thước vector Query/Key dùng riêng cho RoPE
    v_head_dim: int = 128       # Kích thước vector Value
    
    # --- Cấu hình RoPE ---
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    
    # --- DeepSeek MoE (Sinh thái đa luồng chuyên gia) ---
    # Mặc định sử dụng SwiGLUMLP nếu n_routed_experts = 0
    moe_intermediate_size: int = 1536  # Hệ số thu nhỏ của mỗi Expert (dày đặc nhưng gầy) so với Dense gốc (14336)
    n_routed_experts: int = 64         # Số lượng Experts định tuyến độc lập
    num_experts_per_tok: int = 6       # Mỗi Token được phân bổ tới bao nhiêu Experts
    n_shared_experts: int = 2          # Sinh ra mạng luôn luôn bật cho mọi token
    
    # Mức độ Dropout
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    
    pad_token_id: Optional[int] = 0
    bos_token_id: Optional[int] = 1
    eos_token_id: Optional[int] = 2

    def __post_init__(self):
        # Kiểm tra tính đồng bộ
        assert self.qk_rope_head_dim % 2 == 0, "Kích thước RoPE (qk_rope_head_dim) phải là số chẵn."
        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size phải chia hết cho num_attention_heads."
        if self.n_routed_experts > 0:
            assert self.num_experts_per_tok <= self.n_routed_experts, "Số chuyên gia cho mỗi token không được cao hơn tổng chuyên gia."
