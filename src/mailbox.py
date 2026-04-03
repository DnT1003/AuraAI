import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

MAILBOX_DIR = Path(__file__).resolve().parent.parent / ".aura_teams"

class SwarmMailbox:
    """
    Hệ thống Giao tiếp Liên Tiến Trình (IPC) cho Swarm.
    Mô phỏng cơ chế TeammateMailbox của Claude Code.
    Dữ liệu được đẩy vào ổ cứng để các Process terminal cửa sổ khác nhau có thể trò chuyện.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.team_dir = MAILBOX_DIR / session_id
        self.messages_file = self.team_dir / "messages.json"
        
        # Đảm bảo thư mục tồn tại
        self.team_dir.mkdir(parents=True, exist_ok=True)
        if not self.messages_file.exists():
            self._write_raw([])

    def _wait_lock(self) -> bool:
        lockfile = self.team_dir / "mbox.lock"
        for _ in range(50): # Đợi tối đa 5 giây
            try:
                # O_EXCL đảm bảo chỉ 1 process tạo được file, chống race-condition tuyệt đối
                fd = os.open(lockfile, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
                return True
            except FileExistsError:
                time.sleep(0.1)
        return False

    def _release_lock(self):
        lockfile = self.team_dir / "mbox.lock"
        if lockfile.exists():
            try:
                lockfile.unlink()
            except OSError:
                pass

    def _read_raw(self) -> List[Dict[str, Any]]:
        if not self.messages_file.exists(): return []
        try:
            with open(self.messages_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip(): return []
                return json.loads(content)
        except (json.JSONDecodeError, IOError):
            return []

    def _write_raw(self, messages: List[Dict[str, Any]]):
        try:
            with open(self.messages_file, 'w', encoding='utf-8') as f:
                json.dump(messages, f, ensure_ascii=False, indent=2)
        except IOError:
            pass

    def post_message(self, sender: str, target: str, content: str):
        """Gửi thư cho một Agent cụ thể, hoặc 'Dispatcher'"""
        if not self._wait_lock(): return
        
        try:
            messages = self._read_raw()
            messages.append({
                "id": len(messages) + 1,
                "sender": sender,
                "target": target,
                "content": content,
                "timestamp": time.time(),
                "read": False
            })
            self._write_raw(messages)
        finally:
            self._release_lock()

    def get_unread_messages(self, target: str) -> List[Dict[str, Any]]:
        """Lấy thư chưa đọc của Agent"""
        if not self._wait_lock(): return []
        
        try:
            messages = self._read_raw()
            unread = []
            changed = False
            
            for msg in messages:
                if msg["target"] == target and not msg["read"]:
                    unread.append(msg)
                    msg["read"] = True
                    changed = True
                    
            if changed:
                self._write_raw(messages)
            return unread
        finally:
            self._release_lock()

    def cleanup(self):
        """Dọn dẹp hòm thư và file sau khi session kết thúc"""
        import shutil
        if self.team_dir.exists():
            shutil.rmtree(self.team_dir, ignore_errors=True)

if __name__ == "__main__":
    print("[Mailbox] Kiểm thử ghi/đọc IPC")
    mbox = SwarmMailbox("test_session")
    mbox.post_message("Dispatcher", "MathExpert", "Tính 1+1")
    mbox.post_message("MathExpert", "Dispatcher", "Kết quả: 2")
    
    print("Mail cho MathExpert:", mbox.get_unread_messages("MathExpert"))
    print("Mail cho Dispatcher:", mbox.get_unread_messages("Dispatcher"))
    mbox.cleanup()
