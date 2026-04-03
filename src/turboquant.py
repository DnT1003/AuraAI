import torch
import torch.nn.functional as F

class TurboQuant:
    """
    Toán học lõi (Physics) của hệ thống TurboQuant (ICLR 2026).
    Áp dụng PolarQuant (Walsh-Hadamard Transform + Scalar Quantization) 
    để nén KV Cache xuống 2/3/4 bit chống Outliers.
    """
    def __init__(self, head_dim: int, device: torch.device):
        self.head_dim = head_dim
        self.device = device
        
        # 1. Khởi tạo Ma trận Walsh-Hadamard (WHT) bằng Sylvester Construction
        # Bắt buộc phải là luỹ thừa của 2 để đệ quy
        assert (head_dim & (head_dim - 1) == 0) and head_dim > 0, "head_dim phải là lũy thừa của 2"
        self.wht_matrix = self._build_wht_matrix(head_dim).to(device)
        self.scale_factor = 1.0 / (head_dim ** 0.5)
        
        # 2. Sinh Random Sign Flips (Rademacher distribution) cố định cho channel
        # Sử dụng Generator riêng để tránh ảnh hưởng manual_seed hệ thống
        gen = torch.Generator(device=device)
        gen.manual_seed(42)
        self.sign_flips = (torch.randint(0, 2, (1, 1, head_dim), generator=gen, device=device) * 2 - 1).float()
        
        # 3. Simulate Lloyd-Max Centroids Cache
        self.centroids_cache = {}
        
    def _get_centroids(self, bits: int) -> torch.Tensor:
        if bits not in self.centroids_cache:
            self.centroids_cache[bits] = torch.linspace(-3.0, 3.0, steps=(2 ** bits), device=self.device)
        return self.centroids_cache[bits]
        
    def _build_wht_matrix(self, n: int) -> torch.Tensor:
        """Sinh ma trận Hadamard kích cỡ NxN (N phải là lũy thừa của 2)"""
        if n == 1:
            return torch.tensor([[1.]])
        h_half = self._build_wht_matrix(n // 2)
        top = torch.cat([h_half, h_half], dim=1)
        bottom = torch.cat([h_half, -h_half], dim=1)
        return torch.cat([top, bottom], dim=0)

    def compress(self, x: torch.Tensor, bits: int = 4):
        """
        Input: x tensor shape [..., head_dim]
        Output: indices (int8), gamma (fp16)
        """
        # 1. Trích xuất Chuẩn (Norm)
        # x_hat = x / gamma
        gamma = torch.norm(x, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
        x_normed = x / gamma
        
        # 2. Xoay bằng WHT (kéo theo Sign flip) -> Biến phân phối thành Gaussian
        # y = WHT * (x_hat * sign) * (1/sqrt(d))
        x_flipped = x_normed * self.sign_flips
        y = F.linear(x_flipped, self.wht_matrix) * self.scale_factor
        
        # 3. Lượng tử hóa Vô hướng (Scalar Quantization - mô phỏng Lloyd-Max)
        y_expanded = y.unsqueeze(-1)
        centroids_expanded = self._get_centroids(bits).view(1, 1, 1, -1)
        
        # Lấy index của Centroid gần nhất
        distances = torch.abs(y_expanded - centroids_expanded)
        indices = torch.argmin(distances, dim=-1).to(torch.int8)
        
        return indices, gamma.half()

    def decompress(self, indices: torch.Tensor, gamma: torch.Tensor, attention_mask: torch.Tensor = None, bits: int = 4):
        """
        Giải mã. Nếu attention_mask được truyền vào, kích hoạt Sparse V:
        Chỉ giải mã các token có trọng số Attention > 1e-6.
        """
        # Lấy lại giá trị lượng tử
        quantized_vals = self._get_centroids(bits)[indices.long()]
        
        if attention_mask is not None:
            # SPARSE V: Giải thuật attention-gated
            # attention_mask là softmax weight. Bỏ qua dequant nếu < 1e-6
            # Trong Python ta tính ngầm qua mask nhân số 0, 
            # (Ở C++ là bỏ hẳn for loop để skip cycle)
            valid_mask = (attention_mask > 1e-6).float()
            quantized_vals = quantized_vals * valid_mask
            
        # Nghịch đảo WHT = chính nó vì WHT đối xứng và trực giao
        x_flipped_hat = F.linear(quantized_vals, self.wht_matrix) * self.scale_factor
        
        # Trả lại dấu và nhân với chuẩn hình học
        x_hat = x_flipped_hat * self.sign_flips
        x_out = x_hat * gamma
        
        return x_out.half()

if __name__ == "__main__":
    print("[TurboQuant] Kiểm thử Toán học K-V Cache Compression")
    device = torch.device("cpu")
    tq = TurboQuant(head_dim=128, device=device)
    
    dummy_x = torch.randn(2, 5, 128) # batch=2, seq=5, head_dim=128
    
    indices, gamma = tq.compress(dummy_x, bits=4)
    print(f" - Compressed Shape: {indices.shape} | type: {indices.dtype}")
    
    recovered_x = tq.decompress(indices, gamma)
    print(f" - Recovered Shape: {recovered_x.shape} | type: {recovered_x.dtype}")
    
    # Tính sai số
    mse = F.mse_loss(dummy_x, recovered_x.float())
    print(f" - Sai số tái cấu trúc (MSE loss): {mse.item():.4f}")
