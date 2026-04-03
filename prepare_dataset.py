# -*- coding: utf-8 -*-
"""
Script kéo dữ liệu huấn luyện xịn từ thư viện HuggingFace Datasets
Chạy lệnh này trước khi train:
!pip install datasets
!python prepare_dataset.py

Nó sẽ tự động tải 1 phần của bộ dữ liệu nổi tiếng (những mẩu truyện ngôn ngữ tự nhiên) 
và xuất ra file text chuẩn hóa cho Aura AI.
"""
import os

try:
    from datasets import load_dataset
except ImportError:
    print("Vui lòng cài đặt thư viện: pip install datasets")
    exit(1)

def main():
    print("[*] Đang kết nối tới kho HuggingFace HuggingFace/datasets...")
    
    # 1. Chọn Dataset: Ở đây chọn TinyStories (Bộ truyện ngắn chuẩn hóa để train AI nhí)
    # Lấy thử độ 1% đầu tiên (Tương đương vài vạn câu chữ) để test trên Kaggle cho nhanh.
    # Muốn train xịn bạn bỏ chữ "[:1%]" đi là nó kéo hàng triệu truyện!
    dataset_name = "roneneldan/TinyStories"
    print(f"[*] Đang tải dữ liệu: {dataset_name}")
    
    dataset = load_dataset(dataset_name, split="train[:1%]")
    
    # 2. Xử lý & Ghi file
    output_path = "kaggle_data.txt"
    print(f"[*] Đang ghi dữ liệu chuẩn hóa ra tệp: {output_path}")
    
    total_written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        # Lặp qua từng mẩu truyện trong bộ dataset
        for item in dataset:
            text = item.get("text", "")
            if text.strip():
                # Dấu ngắt token chuẩn để mô hình biết khi nào kết thúc một văn bản
                f.write(text + "\n<|end_of_text|>\n")
                total_written += 1
                
    print(f"[*] Tuyệt vời! Đã xuất thành công {total_written:,} mẫu hội thoại/truyện ngắn.")
    print(f"[*] File sẵn sàng tại: {os.path.abspath(output_path)}")
    print("[*] Lệnh Train tiếp theo của bạn: !python train_kaggle.py --data_path kaggle_data.txt")

if __name__ == "__main__":
    main()
