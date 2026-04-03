import subprocess
import time
import uuid
from pathlib import Path
from mailbox import SwarmMailbox

class SubAgentDispatcher:
    """
    Sub-agent OS-Level Framework (Kiến trúc tương tự Claude Code / Hare).
    Thay vì chạy chung 1 tiến trình và chia sẻ biến bộ nhớ, 
    Dispatcher này sẽ bật các Cửa sổ Terminal (Process hệ thống) hoàn toàn biệt lập 
    và giao tiếp qua Mailbox trên ổ cứng.
    """
    def __init__(self, generator=None):
        # Generator được truyền vào có thể không dùng trong OS-Level (vì dễ Over-VRAM)
        # Tiến trình con sẽ tự kích hoạt Mock LLM hoặc nạp mô hình riêng.
        self.session_id = f"session_{uuid.uuid4().hex[:6]}"
        self.mailbox = SwarmMailbox(self.session_id)
        self.roles = ["SearchExpert", "MathExpert", "CodeReviewer", "Synthesizer"]
        self.agent_worker_path = Path(__file__).parent / "agent_worker.py"

    def _boot_agents(self):
        print(f"\n[OS-Level Dispatcher] Thiết lập kênh IPC tại .aura_teams/{self.session_id}")
        """Kéo Agent từ lòng đất lên (Spawn windows terminals)"""
        import sys
        for role in self.roles:
            # Dùng cờ CREATE_NEW_CONSOLE (giá trị 0x00000010) của Windows để bung cửa sổ Terminal độc lập
            cmd = [sys.executable, str(self.agent_worker_path), "--role", role, "--session", self.session_id]
            subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            time.sleep(0.5) # Time-staggering để cửa sổ popup lần lượt giống hacker
            
        print("[OS-Level Dispatcher] Tất cả các đặc vụ đã thức tỉnh trên OS và đang nghe ngóng.")

    def run_swarm(self, user_complex_request: str) -> str:
        self._boot_agents()
        print("[OS-Level Dispatcher] Đợi hông (Warming up) 3s để các process con gõ lệnh...\n")
        time.sleep(3) 

        print(f"[OS-Level Dispatcher] Phân công nhiệm vụ: {user_complex_request}")
        # Bắn thư rác cho 3 chuyên gia song song
        for role in ["SearchExpert", "MathExpert", "CodeReviewer"]:
            self.mailbox.post_message("Dispatcher", role, f"Rà quét và test mảnh nội dung: {user_complex_request}")

        # Polling vòng lặp đợi thư trả lời
        print("[OS-Level Dispatcher] Đang lắng nghe báo cáo từ các Agent...\n")
        results = []
        timeout = 20
        start_t = time.time()
        
        while len(results) < 3 and time.time() - start_t < timeout:
            msgs = self.mailbox.get_unread_messages("Dispatcher")
            for m in msgs:
                print(f" <== [Bưu tá mang thư từ {m['sender']}]: {m['content']}")
                results.append(m['content'])
            time.sleep(0.5) # Nhịp tim Mailbox

        if len(results) < 3:
            print("[CẢNH BÁO] Có Agent chết hoặc Timeout!")

        print("\n[OS-Level Dispatcher] Gom đủ dữ liệu từ chuyên gia, ném cho Synthesizer...")
        synth_payload = " | ".join(results)
        self.mailbox.post_message("Dispatcher", "Synthesizer", f"Hãy diễn dịch và tổng hợp: {synth_payload}")

        final_answer = "(Trống)"
        while final_answer == "(Trống)" and time.time() - start_t < timeout + 10:
            msgs = self.mailbox.get_unread_messages("Dispatcher")
            for m in msgs:
                if m['sender'] == "Synthesizer":
                    final_answer = m['content']
            time.sleep(0.5)

        print(f"\n[OS-Level Dispatcher] Thu được Báo cáo Tổng thể!")
        
        # Shutdown an toàn (Graceful Exit)
        print("[OS-Level Dispatcher] Gửi lệnh EXIT_SWARM (Tự thiêu) cho tất cả Process.")
        for role in self.roles:
            self.mailbox.post_message("Dispatcher", role, "EXIT_SWARM")

        print("[OS-Level Dispatcher] Đợi tiến trình Windows tự thoát và Cleanup ổ cứng...")
        time.sleep(2)
        self.mailbox.cleanup() # Xóa .aura_teams/session_xxx
        
        return final_answer

if __name__ == "__main__":
    print("[MÔI TRƯỜNG TEST] Kiểm thử OS-Level Swarm (Giao diện Windows Popup)")
    swarm = SubAgentDispatcher()
    print("KẾT QUẢ CUỐI CÙNG:\n> ", swarm.run_swarm("Lập trình cho tôi hàm sigmoid."))
