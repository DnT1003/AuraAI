import argparse
import time
import sys
import os

# Cho phép chạy độc lập từ thư mục src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mailbox import SwarmMailbox
from config import ModelConfig
from model import SotaDecoderCausalLM
from generate import EagleHead, EagleSpeculativePipeline

try:
    import tiktoken
except ImportError:
    pass

def run_agent(role: str, session_id: str):
    """
    Vòng lặp sự kiện của Agent. Khởi chạy trong cửa sổ nền đen của Windows.
    Liên tục lắng nghe hộp thư (Mailbox) và xuất kết quả.
    """
    # Đổi màu chữ theo Role để tạo cảm giác Matrix-hacker nếu được
    colors = {
        "SearchExpert": "\033[96m", # Cyan
        "MathExpert": "\033[92m",   # Green
        "CodeReviewer": "\033[93m", # Yellow
        "Synthesizer": "\033[95m"   # Magenta
    }
    color = colors.get(role, "\033[0m")
    reset = "\033[0m"
    
    print(f"{color}===============================================")
    print(f"|  HỆ THỐNG SWARM AURA V2+ - AGENT ACTIVE     |")
    print(f"|  Role: {role:<36} |")
    print(f"|  Session: {session_id:<33} |")
    print(f"==============================================={reset}\n")
    print(f"[*] Đang tải trọng lượng mô hình (Tiny Scale VRAM OOM Prevented)....")
    
    # 1. Khởi tạo LLM Engine siêu nhỏ (Lưu ý: Nếu dev train 14B thì tăng thông số lên)
    # Kích thước này chỉ tốn vài trăm MB RAM/VRAM để chạy ổn định 3 process song song trên máy cá nhân
    config = ModelConfig(
        num_hidden_layers=1, hidden_size=64, moe_intermediate_size=32,
        n_routed_experts=2, num_experts_per_tok=1, n_shared_experts=1,
        num_attention_heads=2, q_lora_rank=16, kv_lora_rank=8,
        qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16,
        vocab_size=100277, use_bitnet=False, use_qlora=False
    )
    # Khởi tạo mô hình trên CPU/GPU
    device = torch.device("cpu") # Giả lập Swarm Dev Mode ép chạy CPU tránh nổ GPU memory
    base_model = SotaDecoderCausalLM(config).to(device)
    base_model.eval()
    
    eagle = EagleHead(hidden_size=64, vocab_size=100277).to(device)
    eagle.eval()
    
    # Kịch bản tải Checkpoint.pt thực tế nếu User đã Train
    ckpt_base = os.path.join(os.path.dirname(__file__), "weights", "base_model.pt")
    ckpt_eagle = os.path.join(os.path.dirname(__file__), "weights", "eagle_head.pt")
    if os.path.exists(ckpt_base):
        print("[+] Tìm thấy Checkpoint, đang Load Weights thật...")
        base_model.load_state_dict(torch.load(ckpt_base, map_location=device, weights_only=True))
    if os.path.exists(ckpt_eagle):
        eagle.load_state_dict(torch.load(ckpt_eagle, map_location=device, weights_only=True))
        
    pipeline = EagleSpeculativePipeline(base_model, eagle, max_draft_len=2)
    
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except NameError:
        tokenizer = None
        print("[!] Không tìm thấy tiktoken. Chạy fallback mock tokens.")
        
    print(f"[*] Sẵn sàng lắng nghe nhiệm vụ từ Dispatcher.\n")
    
    mbox = SwarmMailbox(session_id)
    
    while True:
        # Polling mailbox
        unread = mbox.get_unread_messages(role)
        for msg in unread:
            task_content = msg["content"]
            sender = msg["sender"]
            
            if task_content.strip() == "EXIT_SWARM":
                print(f"\n[!] Nhận lệnh kết thúc. Đóng process {role}...")
                time.sleep(1)
                return
                
            print(f"{color}>>> [Nhận lệnh từ {sender}]:{reset} {task_content}")
            print(f"    {color}[Suy nghĩ]: Phân tích qua SotaDecoderCausalLM...{reset}")
            
            if tokenizer:
                # Flow thực tế: Băm Prompt -> Tensor -> Generate -> Giải mã
                sys_prompt = f"Tôi là {role}. Yêu cầu: {task_content}. Trả lời:"
                input_ids = torch.tensor([tokenizer.encode(sys_prompt)], device=device)
                with torch.no_grad():
                    out_ids = pipeline.generate(input_ids, max_new_tokens=15)
                # Decode
                generated_text = tokenizer.decode(out_ids[0].tolist())
                # Dọn chữ rác do Tensor khởi tạo random sinh ra
                response = f"({role} Replying): [Tensor Sinh Chữ]: {generated_text}"
            else:
                # Nếu User chưa cài tiktoken
                time.sleep(2.0)
                response = f"({role} Replying): Đã xử lý (Không có Tiktoken) -> '{task_content}'"
                
            print(f"    {color}[Response]: Gửi kết quả về Dispatcher...{reset}\n")
            
            mbox.post_message(sender=role, target=sender, content=response)
            
        time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, help="Tên nhận dạng của Agent")
    parser.add_argument("--session", required=True, help="Session ID chung")
    args = parser.add_argument_group()
    args = parser.parse_args()
    
    # Bật kích hoạt mã màu ANSI trên Command Prompt Windows
    os.system('color')
    
    try:
        run_agent(args.role, args.session)
    except KeyboardInterrupt:
        print("\nBị ngắt bởi phím cứng.")
