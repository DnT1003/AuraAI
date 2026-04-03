# -*- coding: utf-8 -*-
"""
Kaggle Training Script cho Aura AI V2
Copy lệnh này vào cell của Kaggle sau khi clone repo:
!python train_kaggle.py --data_path "/kaggle/input/your-dataset/data.txt" --epochs 3 --batch_size 4
"""
import os
# Cứu tinh chống phân mảnh RAM cho Kaggle GPU T4
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

# Ép đưa thư mục src vào Python Path để giải quyết trượt ModuleNotFoundError từ model.py
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import ModelConfig
from model import SotaDecoderCausalLM
from generate import EagleHead
from dataset import create_dataloader

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dummy.txt", help="Đường dẫn file txt/jsonl dataset trên Kaggle")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1) # Rút batch size về 1 để cứu VRAM
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_dir", type=str, default="/kaggle/working/weights")
    args = parser.add_argument_group()
    args = parser.parse_args()

    # Tạo thư mục lưu weights
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Đang khởi động Training trên thiết bị: {device}")

    # 1. Khởi tạo Cấu hình & Model
    # Cấu hình "Kịch khung an toàn" cho dòng Card 12GB VRAM (như RTX 3060, RTX 4070)
    # Tổng Base Model: ~1.2 B tham số (đóng băng).
    # Tổng Tham số Trainable (LoRA + Eagle): ~160 Triệu tham số.
    # Ước lượng VRAM tiêu thụ lúc Train: 7GB - 9GB (Dư sức để nâng batch_size=2 hoặc seq_len=1024).
    config = ModelConfig(
        num_hidden_layers=12, hidden_size=1536, moe_intermediate_size=768,
        n_routed_experts=8, num_experts_per_tok=2, n_shared_experts=2,
        num_attention_heads=12, q_lora_rank=128, kv_lora_rank=64,
        qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=64,
        vocab_size=100277, use_bitnet=False, use_qlora=True, peft_lora_rank=16
    )
    
    # Kỹ thuật xịn: Không ép cứng dtype để tránh NaN. Dùng chuẩn Float32 và để AMP lo phần thu nhỏ Activation.
    
    model = SotaDecoderCausalLM(config).to(device)
    eagle = EagleHead(hidden_size=1536, vocab_size=100277).to(device)
    
    # Ép kiểu LoRA layer về Float32 nếu cần (nhưng ta train thẳng trên fp16 với T4 cho nhẹ)
    
    # 2. Chuẩn bị Dữ liệu
    # (Nếu chạy test không có file, tạo 1 file dummy)
    if not os.path.exists(args.data_path):
        print(f"[!] Không tìm thấy {args.data_path}. Đang tạo file giả lập để chạy thử...")
        with open(args.data_path, "w", encoding="utf-8") as f:
            f.write("Aura AI là hệ thống trí tuệ nhân tạo thế hệ mới. " * 500)

    # Max sequence length giảm xuống 512 nếu Kaggle T4 bị đầy RAM
    dataloader = create_dataloader([args.data_path], batch_size=args.batch_size, max_seq_length=512)
    
    # 3. Optimizer cực hạn: Chỉ Train gradient mở (LoRA weights & Eagle Head)
    # Rà soát đóng băng TOÀN BỘ Base Model (bao gồm cả Embedding và LM_Head tốn kém VRAM)
    for name, param in model.named_parameters():
        param.requires_grad = False
        # Chỉ bật lại Gradient cho các cục Adapter siêu nhỏ
        if "lora_" in name:
            param.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad] + \
                       [p for p in eagle.parameters() if p.requires_grad]
                       
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))

    print(f"[*] Sẵn sàng huấn luyện {sum(p.numel() for p in trainable_params):,} tham số!")

    # Công cụ Tự động cân bằng và thu gọn ma trận tính toán chống NaNs
    scaler = GradScaler("cuda")

    # 4. Vòng lặp Train
    model.train()
    eagle.train()
    for epoch in range(args.epochs):
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Kích hoạt vùng bộ nhớ Hỗn hợp (Mixed Precision)
            with autocast("cuda", dtype=torch.float16):
                # Forward Base Model
                out = model(x, labels=y)
                loss_base = out["loss"]
                
                # Forward EAGLE Head (Học dự đoán token tiếp theo từ current_hidden)
                current_hidden = out["hidden_states"][:, :-1, :] # Lấy tới token kế cuối
                eagle_pred = eagle(current_hidden, x[:, 1:]) # Khớp với shift
                
                # Cho điểm phụ: Eagle prediction phải khớp với Hidden đích
                target_hidden = out["hidden_states"][:, 1:, :].detach()
                loss_eagle = nn.functional.mse_loss(eagle_pred, target_hidden)
                
                # Tổng hợp
                loss = loss_base + 0.5 * loss_eagle
                
            # Backward qua Scaler chuẩn mực
            scaler.scale(loss).backward()
            
            # Gỡ scale trước khi phạt clip_grad_norm_
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            if step % 10 == 0:
                print(f"[Epoch {epoch+1}/{args.epochs} | Step {step}] Loss Base: {loss_base.item():.4f} - Loss Eagle: {loss_eagle.item():.4f}")
                
    # 5. Lưu kết quả
    print("[*] Đang lưu Checkpoints...")
    torch.save(model.state_dict(), os.path.join(args.save_dir, "base_model.pt"))
    torch.save(eagle.state_dict(), os.path.join(args.save_dir, "eagle_head.pt"))
    print(f"[*] Thành công! Dữ liệu nằm tại: {args.save_dir}")

if __name__ == "__main__":
    train()
