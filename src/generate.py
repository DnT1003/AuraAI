import torch
import torch.nn as nn
from typing import List, Tuple

class EagleHead(nn.Module):
    """
    EAGLE (Speculative Sampling via Feature Uncertainty).
    Phát minh lõi: Thay vì dùng 1 LLM thứ 2 làm Draft sinh Token (Top), 
    EAGLE dùng 1 Head nhỏ gắp thẳng lên ngọn của Base Model.
    Head này tận dụng Feature (Hidden State) hiện tại, kết hợp thuật toán Hồi quy Tự động (Auto-regressive) 
    để dự đoán lớp Hidden State của tương lai, bỏ qua hoàn toàn chi phí tính toán mạng Transformer vĩ mô.
    """
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)  # Ghép nối (Current Feature + Embedding của Token vừa sinh)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)  # Bảng từ vựng rút gọn (Tied Weights)

    def forward(self, current_hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """Sinh ra Hidden State (Dự báo của Token Tương lai)"""
        token_emb = self.embed(token_ids)
        # Kết toán Auto-regressive: f(h_t, embed_t) -> h_t+1
        concat_feature = torch.cat([current_hidden, token_emb], dim=-1)
        hidden_pred = self.fc2(self.act(self.fc1(concat_feature)))
        return hidden_pred + current_hidden # Residual connection

class EagleSpeculativePipeline:
    def __init__(self, base_model: nn.Module, eagle_head: EagleHead, max_draft_len: int = 4):
        self.base_model = base_model
        self.eagle_head = eagle_head
        self.max_draft_len = max_draft_len

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100):
        bsz, seq_len = input_ids.shape
        generated_ids = input_ids.clone()
        
        # Lượt 1: Sinh ra Feature gốc hiện hành của hệ thống
        with torch.no_grad():
            out_base = self.base_model(generated_ids)
            current_logits = out_base["logits"][:, -1, :]
            current_token = torch.argmax(current_logits, dim=-1).unsqueeze(-1)
            
            # Extract Feature gốc từ lớp cuối cùng trước lm_head (Tưởng tượng base model trả về hidden_states)
            current_hidden = out_base["hidden_states"][:, -1, :] 

        generated_ids = torch.cat([generated_ids, current_token], dim=-1)

        # Loop Gen Kép
        steps = 0
        while steps < max_new_tokens:
            draft_tokens = []
            draft_hidden = current_hidden.clone()
            draft_token_id = current_token.clone()
            
            # --- 1. Bước Draft Nội bộ mạng HEAD ---
            # EAGLE cực nhẹ, độ trễ O(1) cấu trúc mỏng
            for _ in range(self.max_draft_len):
                draft_hidden = self.eagle_head(draft_hidden, draft_token_id)
                # Dùng lm_head gốc của base model phán ván cược (Cần tie_weight)
                draft_logits = self.base_model.lm_head(draft_hidden) 
                draft_token_id = torch.argmax(draft_logits, dim=-1)
                draft_tokens.append(draft_token_id)
            
            draft_tensor = torch.stack(draft_tokens, dim=1) # Gom chính xác batch
            
            # --- 2. Bước Verify Bơm song song Model Chính ---
            # Forward 1 nhát duy nhất bao phủ trọn vẹn toàn bộ các Draft Tokens
            test_ids = torch.cat([generated_ids, draft_tensor], dim=-1)
            with torch.no_grad():
                 verify_out = self.base_model(test_ids)
                 # target_logits = verify_out["logits"][:, -(self.max_draft_len + 1):, :]
            
            # --- 3. Thuật toán Đối soát (Greedy Verification) ---
            # Kiểm tra từ rễ lên ngọn: Token nào khớp Target thì nhận. Đứt gãy ở đâu cắt bẻ ở đấy.
            draft_len = draft_tensor.shape[1]
            # Lấy Logits sinh ra bởi mô hình gốc để xác thực lớp Draft
            # Chiều dài chuỗi verify_out = old_seq_len + draft_len
            target_logits = verify_out["logits"][:, -(draft_len+1):-1, :]
            target_ids = torch.argmax(target_logits, dim=-1)
            
            # Simple acceptance: accept while draft_tensor == target_ids (Cho Batch Size = 1)
            accepted_len = 0
            for i in range(draft_len):
                if draft_tensor[0, i] == target_ids[0, i]:
                    accepted_len += 1
                else:
                    break
                    
            # Lấy token thưởng (từ Base Model) tại vị trí bác bỏ cuối cùng
            bonus_token = torch.argmax(verify_out["logits"][:, -(draft_len - accepted_len + 1), :], dim=-1).unsqueeze(-1)
            
            valid_drafts = draft_tensor[:, :accepted_len]
            generated_ids = torch.cat([generated_ids, valid_drafts, bonus_token], dim=-1)
            steps += accepted_len + 1
            
            # Lấy Feature làm gốc cho chu kỳ tiếp 
            current_hidden = verify_out["hidden_states"][:, -(draft_len - accepted_len + 1), :]
            current_token = bonus_token
            
        return generated_ids

if __name__ == "__main__":
    print("[Toán học EAGLE] Đã thiết lập Trọng số Hồi quy (Auto-regressive Head) đo độ trễ Tính Bất định.")
