# -*- coding: utf-8 -*-
import sys
import io

# Ép hệ thống dùng mảng kí tự Unicode để tránh mojibake trên Terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, 'src')
import torch

print("=" * 60)
print("AUDIT REPORT - AURA V2 CODEBASE")
print("=" * 60)

errors = []

# TEST 1
print("\n[TEST 1] Init SotaDecoderCausalLM (small)...")
from config import ModelConfig
from model import SotaDecoderCausalLM
config = ModelConfig(
    num_hidden_layers=2, hidden_size=256, moe_intermediate_size=128,
    n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
    num_attention_heads=4, q_lora_rank=128, kv_lora_rank=64,
    qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
    vocab_size=500, use_bitnet=False
)
try:
    model = SotaDecoderCausalLM(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] Total params: {total_params:,}")
except Exception as e:
    errors.append(("TEST1-init", str(e)))
    print(f"  [FAIL] {e}")

# TEST 2
print("\n[TEST 2] Forward pass (batch=1, seq=8)...")
try:
    x = torch.randint(0, 500, (1, 8))
    out = model(x, labels=x)
    print(f"  [OK] Loss: {out['loss'].item():.4f}")
    print(f"  [OK] Logits shape: {out['logits'].shape}")
    print(f"  [OK] Aux loss type: {type(out['aux_loss'])}")
except Exception as e:
    errors.append(("TEST2-forward", str(e)))
    print(f"  [FAIL] {e}")

# TEST 3
print("\n[TEST 3] Backward pass...")
try:
    out['loss'].backward()
    num_with_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    num_total = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  [OK] Grads: {num_with_grad}/{num_total} params have gradients")
except Exception as e:
    errors.append(("TEST3-backward", str(e)))
    print(f"  [FAIL] {e}")

# TEST 4
print("\n[TEST 4] BitLinear mode...")
try:
    config_bit = ModelConfig(
        num_hidden_layers=1, hidden_size=64, moe_intermediate_size=32,
        n_routed_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        num_attention_heads=2, q_lora_rank=32, kv_lora_rank=16,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        vocab_size=500, use_bitnet=True
    )
    model_bit = SotaDecoderCausalLM(config_bit)
    out_bit = model_bit(torch.randint(0, 500, (1, 4)), labels=torch.randint(0, 500, (1, 4)))
    out_bit['loss'].backward()
    print(f"  [OK] BitNet forward+backward OK, loss={out_bit['loss'].item():.4f}")
except Exception as e:
    errors.append(("TEST4-bitnet", str(e)))
    print(f"  [FAIL] {e}")

# TEST 5
print("\n[TEST 5] PagedAttentionEngine...")
try:
    from engine import PagedAttentionEngine
    engine = PagedAttentionEngine(num_blocks=10, block_size=16, num_kv_heads=1, head_dim=128)
    engine.allocate(seq_id=1, num_tokens=20)
    dummy_k = torch.randn(5, 1, 128, dtype=torch.float16)
    dummy_v = torch.randn(5, 1, 128, dtype=torch.float16)
    engine.write_to_cache(1, 0, dummy_k, dummy_v)
    q = torch.randn(1, 1, 128, dtype=torch.float16)
    result = engine.paged_attention_forward(1, q, logical_length=5)
    print(f"  [OK] Paged attn output shape: {result.shape}")
except Exception as e:
    errors.append(("TEST5-engine", str(e)))
    print(f"  [FAIL] {e}")

# TEST 6
print("\n[TEST 6] EagleHead...")
try:
    from generate import EagleHead
    head = EagleHead(hidden_size=256, vocab_size=500)
    h = torch.randn(1, 256)
    tok = torch.tensor([42])
    out_h = head(h, tok)
    print(f"  [OK] EagleHead output: {out_h.shape}")
except Exception as e:
    errors.append(("TEST6-eagle", str(e)))
    print(f"  [FAIL] {e}")

# TEST 7
print("\n[TEST 7] SubAgentDispatcher...")
try:
    from swarm import SubAgentDispatcher
    s = SubAgentDispatcher()
    result = s.run_swarm("Test swarm dispatch")
    print(f"  [OK] Swarm returned string, len={len(result)}")
except Exception as e:
    errors.append(("TEST7-swarm", str(e)))
    print(f"  [FAIL] {e}")

# TEST 8
print("\n[TEST 8] PagedAttention Advanced GC & Prefix Cache...")
try:
    from engine import PagedAttentionEngine
    # Engine với 10 blocks x 16 tokens (Tổng 160 tokens sức chứa)
    engine2 = PagedAttentionEngine(num_blocks=10, block_size=16, num_kv_heads=1, head_dim=128, enable_prefix_caching=True)
    
    # 1. Cấp phát chuỗi lệch cỡ (logical_length = 21, tức cần 2 blocks)
    prompt_tokens = [i for i in range(21)]
    engine2.allocate(seq_id=1, num_tokens=21, token_ids=prompt_tokens)
    
    # 2. Ghi cache để lưu Prefix vào RadixTree
    dummy_k = torch.randn(21, 1, 128, dtype=torch.float16)  
    dummy_v = torch.randn(21, 1, 128, dtype=torch.float16)
    engine2.write_to_cache(1, 0, dummy_k, dummy_v, token_ids=prompt_tokens)
    
    # 3. Request thứ 2 có cùng 16 tokens đầu (Prefix HIT)
    query2_tokens = [i for i in range(16)] + [99, 99, 99] 
    # Cấp phát 19 tokens -> Được thừa hưởng mảng 16 của sequence cũ!
    engine2.allocate(seq_id=2, num_tokens=19, token_ids=query2_tokens)
    
    print(f"  [OK] Seq 2 (Prefix Hit) đã xin mượn block vật lý. Bảng khối: {engine2.block_tables[2]}")
    
    # 4. Trình dọn rác (GC Module)
    engine2.free_sequence(seq_id=2, token_ids=query2_tokens) # Giải phóng sequence 2, ref_count của Block gốc vẫn còn nên Node chưa bị rụng
    engine2.free_sequence(seq_id=1, token_ids=prompt_tokens) # Giải phóng nốt sequence 1
    
    assert len(engine2.free_blocks) == 10, f"Rò rỉ bộ nhớ, chỉ còn {len(engine2.free_blocks)}/10 blocks"
    print("  [OK] Garbage Collection thu hồi đầy đủ 10 Physical Blocks về mảng khả dụng!")
    
except Exception as e:
    errors.append(("TEST8-gc", str(e)))
    print(f"  [FAIL] {e}")

print("\n" + "=" * 60)
if errors:
    print(f"FAILED: {len(errors)} tests")
    for name, err in errors:
        print(f"  - {name}: {err[:80]}")
else:
    print("ALL 8 TESTS PASSED")
print("=" * 60)
