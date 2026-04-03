import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from cache import PrefixTree
from turboquant import TurboQuant

class PagedAttentionEngine:
    """
    Toán học lõi (Physics) của hệ thống vLLM PagedAttention.
    Đã nâng cấp (V2+): Hỗ trợ TurboQuant (KV Cache Compression) và Radix Tree Prefix Caching.
    """
    def __init__(self, 
                 num_blocks: int, 
                 block_size: int, 
                 num_kv_heads: int, 
                 head_dim: int, 
                 device: torch.device = torch.device("cpu"),
                 cache_type_k: str = "fp16",
                 cache_type_v: str = "turbo4",
                 enable_sparse_v: bool = True,
                 enable_prefix_caching: bool = True):
        
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        
        # --- CẤU HÌNH V2+ TURBOQUANT & Caching ---
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.enable_sparse_v = enable_sparse_v
        
        # Tích hợp Radix Tree
        self.enable_prefix_caching = enable_prefix_caching
        self.prefix_tree = PrefixTree(block_size) if enable_prefix_caching else None
        
        # Khởi tạo Lõi nén WHT
        self.tq = TurboQuant(head_dim, device)
        
        # --- TENSOR BỘ NHỚ VẬT LÝ ---
        # 1. Khai báo Cache cho K
        if self.cache_type_k == "fp16":
            self.k_cache = torch.zeros((num_blocks, num_kv_heads, head_dim, block_size), dtype=torch.float16, device=device)
        else:
            # 4-bit (chúng ta dùng int8 array mỏng để chứa index 4-bit)
            self.k_cache_indices = torch.zeros((num_blocks, num_kv_heads, head_dim, block_size), dtype=torch.int8, device=device)
            self.k_cache_gamma = torch.zeros((num_blocks, num_kv_heads, 1, block_size), dtype=torch.float16, device=device)

        # 2. Khai báo Cache cho V (Asymmetric thiết kế: nén V mạnh tay hơn K)
        if self.cache_type_v == "fp16":
            self.v_cache = torch.zeros((num_blocks, num_kv_heads, head_dim, block_size), dtype=torch.float16, device=device)
        else:
            self.v_cache_indices = torch.zeros((num_blocks, num_kv_heads, head_dim, block_size), dtype=torch.int8, device=device)
            self.v_cache_gamma = torch.zeros((num_blocks, num_kv_heads, 1, block_size), dtype=torch.float16, device=device)
        
        self.free_blocks: List[int] = list(range(num_blocks))[::-1]
        self.block_tables: Dict[int, List[int]] = {}

    def allocate(self, seq_id: int, num_tokens: int, token_ids: Optional[List[int]] = None) -> bool:
        """Cấp phát Block tọa độ. Có Prefix Caching nếu đã bật."""
        self.block_tables[seq_id] = []
        
        # -- 1. Kiểm tra Prefix Caching --
        if self.enable_prefix_caching and token_ids is not None:
            matched_blocks = self.prefix_tree.match_prefix(token_ids)
            if matched_blocks:
                self.block_tables[seq_id].extend(matched_blocks)
                # Trừ đi số token đã có trong cache
                num_tokens -= len(matched_blocks) * self.block_size
                if num_tokens <= 0: return True
        
        # -- 2. Cấp phát phần còn thiếu --
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < blocks_needed:
            return False
            
        allocated = []
        for _ in range(blocks_needed):
            b_idx = self.free_blocks.pop()
            allocated.append(b_idx)
            
        self.block_tables[seq_id].extend(allocated)
        return True

    def free_sequence(self, seq_id: int, token_ids: Optional[List[int]] = None):
        """
        Thu hồi toàn bộ tài nguyên của một Sequence ID.
        Nếu dùng Prefix Caching, gọi decrease_ref trước để xem block nào thực sự an toàn (ref_count = 0) để xóa.
        """
        if seq_id not in self.block_tables:
            return
            
        blocks_to_free = self.block_tables[seq_id]
        
        if self.enable_prefix_caching and token_ids is not None:
            visited_blocks, freed_from_tree = self.prefix_tree.decrease_ref(token_ids)
            # Khối an toàn để xóa: 
            # 1. Block không được quản lý bởi Radix Tree (uncached)
            # 2. Block được quản lý nhưng ref_count đã chạm 0 (freed_from_tree)
            safe_to_free = [b for b in blocks_to_free if b not in visited_blocks or b in freed_from_tree]
            self.free_blocks.extend(safe_to_free)
        else:
            # Thu hồi bình thường (thẳng tay)
            self.free_blocks.extend(blocks_to_free)
            
        del self.block_tables[seq_id]

    def write_to_cache(self, seq_id: int, start_idx: int, k_states: torch.Tensor, v_states: torch.Tensor, token_ids: Optional[List[int]] = None):
        """
        Ghi vào bộ nhớ. Nếu Cache_type != fp16, dùng TurboQuant nén (PolarQuant) trước khi ghi.
        k_states/v_states: [num_tokens, num_heads, head_dim]
        """
        table = self.block_tables.get(seq_id, [])
        if not table: return
        
        num_tokens = k_states.shape[0]
        
        # Tiết kiệm vòng lặp Python bằng cách xác định các khối vật lý
        for i in range(num_tokens):
            global_pos = start_idx + i
            logical_b = global_pos // self.block_size
            offset = global_pos % self.block_size
            
            if logical_b >= len(table): break
            phys_b = table[logical_b]
            
            # --- Write K ---
            k_t = k_states[i] # [num_heads, head_dim]
            if self.cache_type_k == "fp16":
                self.k_cache[phys_b, :, :, offset] = k_t
            else:
                idx, gamma = self.tq.compress(k_t, bits=4)
                self.k_cache_indices[phys_b, :, :, offset] = idx
                self.k_cache_gamma[phys_b, :, :, offset] = gamma
                
            # --- Write V --- 
            v_t = v_states[i] # [num_heads, head_dim]
            if self.cache_type_v == "fp16":
                self.v_cache[phys_b, :, :, offset] = v_t
            else:
                idx, gamma = self.tq.compress(v_t, bits=4)
                self.v_cache_indices[phys_b, :, :, offset] = idx
                self.v_cache_gamma[phys_b, :, :, offset] = gamma

        # Cầm token ids đẩy vào Radix Tree để train cache
        if self.enable_prefix_caching and token_ids is not None:
             self.prefix_tree.insert(token_ids, table)

    def paged_attention_forward(self, seq_id: int, q_states: torch.Tensor, logical_length: int) -> torch.Tensor:
        """
        Nhân ma trận Dot-product. Giải nén nhanh on-the-fly. Hỗ trợ Sparse V đánh chặn giải nén.
        """
        table = self.block_tables[seq_id]
        blocks_to_fetch = len(table)
        
        # --- 1. Fetch Key (Và giải mã nếu cần) ---
        if self.cache_type_k == "fp16":
            phys_k = self.k_cache[table]
        else:
            k_idx = self.k_cache_indices[table]
            k_gam = self.k_cache_gamma[table]
            # Giải mã bằng tensor ops
            # Shape sau khi transpose về đúng khuôn mẫu compress: [..., head_dim]
            k_idx_flat = k_idx.permute(0, 3, 1, 2).reshape(-1, self.num_kv_heads, self.head_dim)
            k_gam_flat = k_gam.permute(0, 3, 1, 2).reshape(-1, self.num_kv_heads, 1)
            
            phys_k = self.tq.decompress(k_idx_flat, k_gam_flat)
            # Khôi phục shape [blocks, num_heads, head_dim, block_size]
            phys_k = phys_k.view(blocks_to_fetch, self.block_size, self.num_kv_heads, self.head_dim).permute(0, 2, 3, 1)
        
        # Gom mảnh đứt gãy
        final_k = phys_k.permute(1, 2, 0, 3).reshape(self.num_kv_heads, self.head_dim, -1)
        final_k = final_k[..., :logical_length].transpose(1, 2)
        
        # --- 2. Tính Attention Mask (Dot-Product Q x K) ---
        q_swapped = q_states.transpose(0, 1) # [num_heads, 1, head_dim]
        scale = q_swapped.shape[-1] ** -0.5
        scores = torch.matmul(q_swapped * scale, final_k.transpose(-2, -1))
        attn_weights = torch.softmax(scores, dim=-1) # [num_heads, 1, seq_len]
        
        # --- 3. Fetch Value (Giải mã + Sparse V Optimization) ---
        if self.cache_type_v == "fp16":
            phys_v = self.v_cache[table]
            final_v = phys_v.permute(1, 2, 0, 3).reshape(self.num_kv_heads, self.head_dim, -1)
            final_v = final_v[..., :logical_length].transpose(1, 2)
        else:
            v_idx = self.v_cache_indices[table]
            v_gam = self.v_cache_gamma[table]
            v_idx_flat = v_idx.permute(0, 3, 1, 2).reshape(-1, self.num_kv_heads, self.head_dim)
            v_gam_flat = v_gam.permute(0, 3, 1, 2).reshape(-1, self.num_kv_heads, 1)
            
            # Tính năng Sparse V: Truyền mask chứa attention weights xuống
            # Mở rộng attn_weights về đúng số tokens cần dequantize, điền 0 cho dummy
            sparse_mask = None
            if self.enable_sparse_v:
                pad_len = (blocks_to_fetch * self.block_size) - logical_length
                padded_mask = F.pad(attn_weights.transpose(1, 2), (0, 0, 0, pad_len)) # [num_heads, seq_len_padded, 1]
                # Đưa mask về mặt phẳng flat giống v_idx_flat: [-1, num_heads, 1]
                sparse_mask = padded_mask.transpose(0, 1).reshape(-1, self.num_kv_heads, 1)
            
            phys_v = self.tq.decompress(v_idx_flat, v_gam_flat, attention_mask=sparse_mask)
            phys_v = phys_v.view(blocks_to_fetch, self.block_size, self.num_kv_heads, self.head_dim).permute(0, 2, 3, 1)
            final_v = phys_v.permute(1, 2, 0, 3).reshape(self.num_kv_heads, self.head_dim, -1)
            final_v = final_v[..., :logical_length].transpose(1, 2)

        # --- 4. Tích lũy Result ---
        out = torch.matmul(attn_weights, final_v)
        return out.transpose(0, 1) # [1, num_heads, head_dim]

if __name__ == "__main__":
    print("[Aura V2+ PagedAttention] Kích hoạt TurboQuant (q8_0 K + turbo4 V) + Prefix Caching.")
    engine = PagedAttentionEngine(num_blocks=10, block_size=4, num_kv_heads=1, head_dim=128, 
                                  cache_type_k="fp16", cache_type_v="turbo4", enable_prefix_caching=True)
    engine.allocate(seq_id=1, num_tokens=5, token_ids=[10, 20, 30, 40, 50])
    
    dummy_k = torch.randn(5, 1, 128, dtype=torch.float16)
    dummy_v = torch.randn(5, 1, 128, dtype=torch.float16)
    
    # 1. Pipeline Ghi (Nén trực tiếp V)
    engine.write_to_cache(1, 0, dummy_k, dummy_v, token_ids=[10, 20, 30, 40, 50])
    print("- Trọn vẹn quá trình Ghi, Nén PolarQuant (V) và Sinh Radix Tree Edge.")
    
    # 2. Query tiếp với prompt giống hệ -> Test Caching
    engine.allocate(seq_id=2, num_tokens=2, token_ids=[10, 20])
    print("- Số Block hiện có của Seq_id 2 sau khi match Radix Tree (Block xài chung):", engine.block_tables[2])
    
    # 3. Test Generate có Sparse V dequantization
    q = torch.randn(1, 1, 128, dtype=torch.float16)
    result = engine.paged_attention_forward(1, q, logical_length=5)
    print("- Kết quả tính Attention Paged thành công, giải nén (Dequant) dựa trên mức Sparse.")
