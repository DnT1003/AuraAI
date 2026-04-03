from typing import List, Dict, Optional

class RadixNode:
    """Nút trên cây Tiền tố chứa đoạn Token Hash và Block Index."""
    def __init__(self):
        self.children: Dict[int, 'RadixNode'] = {}
        
        # Địa chỉ Block Vật lý gắn với Token id tại Node này
        # Node gốc trống, các Node con sẽ trỏ tới Block Index
        self.physical_block_index: Optional[int] = None
        
        # Ref-count đếm số lượng Sequence đang cùng trỏ vào (Để giải phóng bộ nhớ khi cần)
        self.ref_count: int = 0

class PrefixTree:
    """
    Cấu trúc cây Radix chặn Context.
    Cho phép PagedAttentionEngine tìm xem chuỗi Input_ids (Prompt) 
    đã được phân bổ Block hay nén bằng TurboQuant trước đó chưa.
    """
    def __init__(self, block_size: int):
        self.root = RadixNode()
        self.block_size = block_size
        
        # LRU list để giải phóng memory (chưa implement sâu cho prototype)

    def match_prefix(self, token_ids: List[int]) -> List[int]:
        """
        Duyệt chuỗi Token qua cây để nhặt lại các Physical Blocks đã Caching.
        Trả về: Danh sách Physical Block Indices.
        """
        matched_blocks = []
        current_node = self.root
        
        # Ta nhóm token theo block_size vì mỗi Block chứa 'block_size' tokens
        # Tìm match nguyên một Block
        logical_blocks_count = len(token_ids) // self.block_size
        
        for i in range(logical_blocks_count):
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            block_tokens = tuple(token_ids[start_idx:end_idx])
            
            # Cấu trúc Radix Tree thực tế sẽ ghép nhiều tokens thành nhãn (label) của Edge.
            # Ở bản Prototype này, ta dùng hash của cụm Tuple block_tokens làm key chẻ nhánh.
            hashed_key = hash(block_tokens)
            
            if hashed_key in current_node.children:
                current_node = current_node.children[hashed_key]
                matched_blocks.append(current_node.physical_block_index)
                current_node.ref_count += 1
            else:
                # Đứt mạch (Tiền tố kết thúc ở đây)
                break
                
        return matched_blocks

    def insert(self, token_ids: List[int], physical_blocks: List[int]):
        """
        Đưa System Prompt / Context vào Tree sau khi cấp phát và gen.
        Để lần tới có Sequence giống y xì thì tái sử dụng.
        """
        current_node = self.root
        
        logical_blocks_count = len(physical_blocks)
        for i in range(logical_blocks_count):
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            block_tokens = tuple(token_ids[start_idx:end_idx])
            hashed_key = hash(block_tokens)
            
            if hashed_key not in current_node.children:
                new_node = RadixNode()
                new_node.physical_block_index = physical_blocks[i]
                current_node.children[hashed_key] = new_node
                
            current_node = current_node.children[hashed_key]
            # Tăng Ref-count (Một chuỗi mới vừa học được context này)
            current_node.ref_count += 1

    def decrease_ref(self, token_ids: List[int]) -> tuple[List[int], List[int]]:
        """
        Duyệt chuỗi Token để giảm ref_count cho các block.
        Trả về tuple: (visited_blocks, freed_blocks)
        """
        current_node = self.root
        logical_blocks_count = len(token_ids) // self.block_size
        
        visited_blocks = []
        freed_blocks = []
        path = [(None, None, current_node)]
        
        for i in range(logical_blocks_count):
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            block_tokens = tuple(token_ids[start_idx:end_idx])
            hashed_key = hash(block_tokens)
            
            if hashed_key in current_node.children:
                parent = current_node
                current_node = current_node.children[hashed_key]
                path.append((parent, hashed_key, current_node))
                visited_blocks.append(current_node.physical_block_index)
            else:
                break
                
        # Duyệt ngược từ dưới lên để gỡ node (tỉa cành) nếu ref_count == 0
        for parent, key, node in reversed(path[1:]):
            if node.ref_count > 0:
                node.ref_count -= 1
            if node.ref_count == 0:
                freed_blocks.append(node.physical_block_index)
                del parent.children[key]
                
        return visited_blocks, freed_blocks

if __name__ == "__main__":
    print("[Radix Tree] Cấu trúc bộ đệm Prefix Caching.")
    
    tree = PrefixTree(block_size=4)
    # Ví dụ prompt: "System prompt hello world", chia mảng [10, 20, 30, 40]
    prompt = [10, 20, 30, 40, 50, 60, 70, 80]
    blocks_allocated = [900, 901] # Cấp block 900 cho 4 token đầu, 901 cho 4 token sau
    
    # 1. Ghi lại Prompt dài này vào Cache
    tree.insert(prompt, blocks_allocated)
    
    # 2. Xảy ra một Query mới có cùng tiền tố
    new_query = [10, 20, 30, 40, 99, 88, 77] # Chỉ giống khúc đầu (Block 1)
    
    matched = tree.match_prefix(new_query)
    print(f" - Physical Blocks có thể tái sử dụng cho nhánh mới: {matched} (Dự kiến: [900])")
