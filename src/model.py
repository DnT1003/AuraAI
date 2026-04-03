import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Tuple

from config import ModelConfig

# ==========================================
# 1. Cơ Chế BitNet 1.58-bit Chính Thức (Toán học Thực tế)
# Quantize CẢ Đầu Vào (Activation) lẫn Trọng Số (Weight) 
# Định mức logic The Era of 1-bit LLMs
# ==========================================
def activation_quant(x: torch.Tensor) -> torch.Tensor:
    """Ép Activation xuống định dạng Int8 absmax"""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True)[0].clamp_(min=1e-5)
    # y = round(x * scale) -> Giữ nguyên mạch Gradient qua STE
    y = torch.round(x * scale)
    return (y - x * scale).detach() + x * scale

def weight_quant(w: torch.Tensor) -> torch.Tensor:
    """Ép Weight xuống {-1, 0, 1} dựa trên mean absolute"""
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    y = torch.round(w * scale).clamp_(-1, 1)
    return (y - w * scale).detach() + w * scale

class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Trong hệ thống BitNet thực, input phải được Normalize trước khi Quantize
        x_norm = self.norm(x)
        
        # 1. Quantize Input xuống Không gian 8-bit
        x_quant = activation_quant(x_norm)
        
        # 2. Quantize Weight xuống Không gian -1, 0, 1
        w_quant = weight_quant(self.weight)
        
        # 3. Phép nhân ma trận tốc độ cao (Thực tế hàm c++ Cắt phép toán MUL, ở mô phỏng ta xài lại F.linear matrix)
        out = F.linear(x_quant, w_quant, self.bias)
        
        # 4. Trả lại không gian tỷ lệ (Scale Inverse)
        w_scale = self.weight.abs().mean().clamp_(min=1e-5)
        x_scale = x.abs().max(dim=-1, keepdim=True)[0].clamp_(min=1e-5) / 127.0
        
        return out * (w_scale * x_scale)

class LoRALinear(nn.Module):
    """
    Adapter Mở rộng cho Parameter-Efficient Fine-Tuning.
    Sẽ cuộn vòng ngoài layer `BitLinear` hoặc `Linear` thông thường.
    """
    def __init__(self, base_layer: nn.Module, in_features: int, out_features: int, config: ModelConfig):
        super().__init__()
        self.base_layer = base_layer
        
        # Đóng băng (Freeze) lớp core
        for p in self.base_layer.parameters():
            p.requires_grad = False
            
        r = config.peft_lora_rank
        # Mở rộng 2 ma trận tinh chỉnh nhẹ
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        self.scaling = config.peft_lora_alpha / r
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B) # B bắt đầu bằng 0 để đảm bảo initial network == Identity
        
        self.dropout = nn.Dropout(p=config.peft_lora_dropout) if config.peft_lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        # Bóc tác qua ngõ LoRA bypass
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

def build_linear(config: ModelConfig, in_features: int, out_features: int, bias: bool = False):
    if getattr(config, "use_bitnet", False):
        layer = BitLinear(in_features, out_features, bias=bias)
    else:
        layer = nn.Linear(in_features, out_features, bias=bias)
        
    if getattr(config, "use_qlora", False):
        layer = LoRALinear(layer, in_features, out_features, config)
        
    return layer


# ==========================================
# CÁC HÀM ROPE VÀ NORMALIZATION CORES
# ==========================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm_x.type_as(x)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    shape = [d if i == 1 or i == xq_.ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ==========================================
# 2. Multi-Head Latent Attention (Toán học Đặc Tả DeepSeek V2)
# ==========================================
class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        self.kv_a_proj_with_norm = nn.Sequential(
            build_linear(config, config.hidden_size, config.kv_lora_rank, bias=False),
            RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        )
        self.kv_b_proj = build_linear(config, config.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.k_pe_proj = build_linear(config, config.hidden_size, self.qk_rope_head_dim, bias=False)
        
        self.q_a_proj_with_norm = nn.Sequential(
            build_linear(config, config.hidden_size, config.q_lora_rank, bias=False),
            RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        )
        self.q_b_proj = build_linear(config, config.q_lora_rank, self.num_heads * self.qk_nope_head_dim, bias=False)
        self.q_pe_proj = build_linear(config, config.q_lora_rank, self.qk_rope_head_dim, bias=False)
        
        self.o_proj = build_linear(config, self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # Hệ số scaling cho Dot-Product chuẩn hóa lại theo kích thước Head vật lý
        self.software_scale = (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.size()
        c_kv = self.kv_a_proj_with_norm(x)
        kv_b = self.kv_b_proj(c_kv).view(bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_b, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = self.k_pe_proj(x).view(bsz, seq_len, 1, self.qk_rope_head_dim)
        
        c_q = self.q_a_proj_with_norm(x)
        q_nope = self.q_b_proj(c_q).view(bsz, seq_len, self.num_heads, self.qk_nope_head_dim)
        q_pe = self.q_pe_proj(c_q).view(bsz, seq_len, 1, self.qk_rope_head_dim)
        
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, freqs_cis=freqs_cis)
        q_pe = q_pe.expand(-1, -1, self.num_heads, -1)
        k_pe = k_pe.expand(-1, -1, self.num_heads, -1)
        
        q = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)
        k = torch.cat([k_nope, k_pe], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Xử lý Hợp nhất Mask (Causal + Padding)
        if attention_mask is not None:
            device, dtype = q.device, q.dtype
            if attention_mask.dtype == torch.bool:
                # Merge kiểu Boolean (AND logic)
                causal_bool = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
                attention_mask = attention_mask & causal_bool
            else:
                # Merge kiểu Additive (-inf float logic)
                causal_inf = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), diagonal=1)
                attention_mask = attention_mask.to(dtype) + causal_inf
            is_causal_flag = False
        else:
            is_causal_flag = True

        output = F.scaled_dot_product_attention(
            q * self.software_scale, k, v, 
            attn_mask=attention_mask, 
            dropout_p=self.dropout.p if self.training else 0.0, 
            is_causal=is_causal_flag,
            scale=1.0 # Đã tự scale phía trên
        )
        
        return self.o_proj(output.transpose(1, 2).contiguous().view(bsz, seq_len, -1))

# ==========================================
# 3. Dense & Sparse Expert Blocks với Hệ số Cân Bằng Tải
# ==========================================
class ExpertBlock(nn.Module):
    def __init__(self, config: ModelConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = build_linear(config, config.hidden_size, intermediate_size, bias=False)
        self.up_proj = build_linear(config, config.hidden_size, intermediate_size, bias=False)
        self.down_proj = build_linear(config, intermediate_size, config.hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class DeepSeekMoE(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_routed = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        
        self.shared_experts = ExpertBlock(config, config.moe_intermediate_size * config.n_shared_experts)
        self.routed_experts = nn.ModuleList([ExpertBlock(config, config.moe_intermediate_size) for _ in range(self.n_routed)])
        self.gate = nn.Linear(config.hidden_size, self.n_routed, bias=False)

    def forward(self, x: torch.Tensor):
        bsz, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        router_logits = self.gate(x_flat)
        routing_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Lấy top_k chuyên gia
        routing_weights, selected_experts = torch.topk(routing_probs, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        
        # --- Hàm Tính Toán Load Balancing Loss (Auxiliary Loss) ---
        # Ngăn chặt hiện tượng chênh lệch độ tải giữa các Expert
        expert_mask = F.one_hot(selected_experts, num_classes=self.n_routed).float() # (N, top_k, num_experts)
        tokens_per_expert = expert_mask.sum(dim=(0, 1)) # (num_experts,)
        router_probs_mean = routing_probs.mean(dim=0)  # (num_experts,)
        
        # Loss tỷ lệ thuận với: Số phần trăm token chảy vào expert X * Điểm xác suất đi vào X
        # Nếu model rọi hết dữ liệu vào 1 expert -> hàm N.alpha sẽ trừng phạt gradient để khuếch tán sang expert khác
        alpha = self.n_routed 
        balance_loss = alpha * torch.sum(tokens_per_expert * router_probs_mean) / (x_flat.size(0) * self.top_k)
        
        # Điều hướng phân bổ qua các module
        final_hidden_states = torch.zeros_like(x_flat)
        for expert_idx in range(self.n_routed):
            idx, nth_expert = torch.where(selected_experts == expert_idx)
            if idx.numel() == 0: continue
            
            out = self.routed_experts[expert_idx](x_flat[idx])
            out = out * routing_weights[idx, nth_expert, None].type_as(out)
            final_hidden_states.index_add_(0, idx, out)

        shared_out = self.shared_experts(x_flat)
        final_hidden_states = final_hidden_states + shared_out
        
        # Retun kèm loss để train.py có thể update đạo hàm
        return final_hidden_states.view(bsz, seq_len, hidden_size), balance_loss


class ParallelTransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = MultiHeadLatentAttention(config)
        self.mlp = DeepSeekMoE(config) if getattr(config, "n_routed_experts", 0) > 0 else ExpertBlock(config, config.moe_intermediate_size * 2)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        norm_x = self.input_layernorm(x)
        attn_out = self.attention(norm_x, freqs_cis, attention_mask)
        
        # Lưu loss cân bằng nếu đó là MoE
        mlp_out, aux_loss = self.mlp(norm_x) if isinstance(self.mlp, DeepSeekMoE) else (self.mlp(norm_x), 0.0)
        
        return x + attn_out + mlp_out, aux_loss

class SotaDecoderCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([ParallelTransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = build_linear(config, config.hidden_size, config.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(config.qk_rope_head_dim, config.max_position_embeddings * 2, config.rope_theta)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None):
        bsz, seq_len = input_ids.shape
        h = self.embed_tokens(input_ids)
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)
        
        total_aux_loss = 0.0
        for layer in self.layers:
            h, aux_loss = layer(h, freqs_cis)
            total_aux_loss += aux_loss
            
        final_hidden = self.norm(h)
        logits = self.lm_head(final_hidden)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            # Cộng thêm hệ số làm mượt cân bằng tải
            loss = loss + 0.01 * total_aux_loss
            
        return {"loss": loss, "logits": logits, "aux_loss": total_aux_loss, "hidden_states": final_hidden}

if __name__ == "__main__":
    print("[Nâng Cấp Kế Cấu Toán Cấp Thấp] model.py đã được nạp chuẩn Loss MoE và STE Quantization.")
