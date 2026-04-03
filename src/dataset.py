import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import os
import random

class TiktokenDataset(Dataset):
    def __init__(self, data_sources, max_seq_length=8192):
        """
        Dataloader cho model dạng causal language modeling có sử dụng Tiktoken (Chuẩn Claude/Gemini-ish ~100k tokens)
        
        data_sources: danh sách file txt hoặc jsonl
        max_seq_length: chiều dài chập tối đa (Sequence Packing)
        """
        super().__init__()
        self.max_seq_len = max_seq_length
        # Dùng tiktoken để giả lập bộ tokenizer lớn (ngôn ngữ tự nhiên & code tốt hơn BPE cũ)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        self.tokenized_chunks = []
        self._load_and_pack(data_sources)
        
    def _load_and_pack(self, sources):
        buffer = []
        for src in sources:
            # Fake logic: đọc text file hoặc db lớn. 
            # Dưới đây giả định đọc file text. Trong thực tế cần dùng datasets của huggingface mapping.
            if not os.path.exists(src):
                print(f"Warning: Data source {src} not found during init.")
                continue
                
            with open(src, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = self.tokenizer.encode(text, allowed_special="all")
                buffer.extend(tokens)
                
                # Cắt thành các pack size chuẩn xác max_seq_length để tận dụng 100% compute (Packing technique)
                while len(buffer) >= self.max_seq_len + 1: # +1 để có input_id và label (shift 1)
                    segment = buffer[:self.max_seq_len + 1]
                    self.tokenized_chunks.append(segment)
                    buffer = buffer[self.max_seq_len:]
                    
    def __len__(self):
        # Không dùng dummy tĩnh. Trả về đúng chiều dài thực tế để tránh lỗi dữ liệu khi train thật.
        return len(self.tokenized_chunks)
        
    def __getitem__(self, idx):
        if len(self.tokenized_chunks) == 0:
            raise IndexError("Dataset rỗng! Chưa có data source hợp lệ.")
            
        chunk = self.tokenized_chunks[idx]
        chunk_tensor = torch.tensor(chunk, dtype=torch.long)
        
        x = chunk_tensor[:-1]
        y = chunk_tensor[1:]
        return x, y

def create_dataloader(data_paths, batch_size=4, max_seq_length=4096, num_workers=4):
    ds = TiktokenDataset(data_paths, max_seq_length=max_seq_length)
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True # Chuyển tensor lên GPU tốc độ cao
    )
