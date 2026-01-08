import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from util.case_util import save_all_cases_by_name
from cases.softmax.case import Case


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    case_list = []

    # Case 1: Normal range values (typical attention logits in BF16)
    case1 = Case()
    case1.t_input = torch.randn(4, 128) * 5
    case_list.append(case1)

    # Case 2: Large positive values (BF16 has same range as FP32)
    case2 = Case()
    case2.t_input = torch.rand(4, 128) * 50 + 50
    case_list.append(case2)

    # Case 3: Large negative values (underflow in exp())
    case3 = Case()
    case3.t_input = torch.rand(4, 128) * (-50) - 50
    case_list.append(case3)

    # Case 4: Mixed extreme values (stability test)
    case4 = Case()
    case4.t_input = torch.randn(4, 128) * 50
    case_list.append(case4)

    # Case 5: Very large positive values (overflow test)
    case5 = Case()
    case5.t_input = torch.rand(4, 128) * 40 + 80
    case_list.append(case5)

    # Case 6: One very large value dominates (common in attention)
    case6 = Case()
    case6.t_input = torch.ones(4, 128) * (-10)
    case6.t_input[:, 0] = 50
    case_list.append(case6)

    # Case 7: All equal values (should give uniform distribution)
    case7 = Case()
    case7.t_input = torch.ones(4, 128) * 5.0
    case_list.append(case7)

    # Case 8: Very small differences (BF16 precision ~3-4 digits)
    # BF16 can only represent ~3 decimal digits, so differences < 0.01 may be lost
    case8 = Case()
    base = torch.ones(4, 128) * 10.0
    noise = torch.randn(4, 128) * 0.01  # Smaller than FP32 case
    case8.t_input = base + noise
    case_list.append(case8)

    # Case 9: Large sequence length (typical for LLM with BF16)
    case9 = Case()
    case9.t_input = torch.randn(2, 2048) * 10
    case_list.append(case9)

    # Case 10: Alternating large/small values (dynamic range test)
    case10 = Case()
    case10.t_input = torch.zeros(4, 128)
    case10.t_input[:, ::2] = 50  # Even positions
    case10.t_input[:, 1::2] = -50  # Odd positions
    case_list.append(case10)

    # Case 11: Exponential distribution (attention pattern)
    case11 = Case()
    case11.t_input = torch.exp(torch.linspace(-10, 0, 128)).repeat(4, 1)
    case_list.append(case11)

    # Case 12: Near-zero values (small gradients)
    case12 = Case()
    case12.t_input = torch.randn(4, 128) * 1e-3
    case_list.append(case12)

    # Case 13: Large batch with masked attention (real-world scenario)
    case13 = Case()
    batch_size, seq_len = 16, 512
    case13.t_input = torch.randn(batch_size, seq_len) * 5
    for i in range(batch_size):
        mask_end = torch.randint(seq_len//2, seq_len, (1,)).item()
        case13.t_input[i, mask_end:] = -1e4
    case_list.append(case13)

    # Case 14: Values at BF16 precision boundary
    # BF16 mantissa has 7 bits, so 1.0 + 2^-7 = 1.0078125 is smallest distinguishable
    case14 = Case()
    case14.t_input = torch.tensor([[1.0, 1.0078125, 1.015625, 1.03125]] * 32)
    case_list.append(case14)

    # Case 15: Powers of 2 (exactly representable in BF16)
    case15 = Case()
    case15.t_input = torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]] * 16)
    case_list.append(case15)

    # Case 16: Dense values in moderate range (typical inference)
    case16 = Case()
    case16.t_input = torch.randn(8, 256) * 3
    case_list.append(case16)

    # Case 17: Subnormal-adjacent values (BF16 min normal ≈ 1.175e-38)
    case17 = Case()
    case17.t_input = torch.randn(4, 128) * 1e-6
    case_list.append(case17)

    # Case 18: Very wide range (tests normalization stability)
    case18 = Case()
    case18.t_input = torch.cat([
        torch.ones(4, 32) * 80,     # Very large
        torch.randn(4, 32) * 10,    # Medium
        torch.ones(4, 32) * (-80),  # Very small (after exp)
        torch.randn(4, 32) * 0.1    # Near zero
    ], dim=1)
    case_list.append(case18)

    # Case 19: Systematic rounding test (adjacent BF16 values)
    case19 = Case()
    # Create values that differ by the smallest BF16 step
    base = 16.0  # Nice power of 2 for BF16
    case19.t_input = torch.arange(0, 128).float().reshape(4, 32) * 0.125 + base
    case_list.append(case19)

    # Case 20: Extreme overflow scenario (near BF16 max ≈ 3.39e38)
    case20 = Case()
    case20.t_input = torch.tensor([[1e30, 5e30, 1e31, 5e31]] * 32)
    case_list.append(case20)

    # Case 21: Gradient-like small values (BF16 precision matters)
    case21 = Case()
    case21.t_input = torch.rand(4, 128) * 0.1 + 1e-3
    case_list.append(case21)

    # Case 22: Causal mask pattern (lower triangular attention)
    case22 = Case()
    seq_len = 64
    case22.t_input = torch.randn(4, seq_len) * 3
    # Apply causal mask: upper triangle gets -inf-like values
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    for i in range(4):
        case22.t_input[i, mask[0]] = -1e4
    case_list.append(case22)

    # Case 23: Challenging precision case - values differ by < BF16 epsilon
    case23 = Case()
    base = torch.ones(4, 128) * 100.0
    # BF16 epsilon at 100 is ~0.0625, add smaller noise
    noise = torch.randn(4, 128) * 0.01
    case23.t_input = base + noise
    case_list.append(case23)

    # Case 24: Real transformer attention logits simulation
    case24 = Case()
    # Simulate QK^T / sqrt(d_k) pattern
    d_k = 64
    q = torch.randn(2, 128, d_k)
    k = torch.randn(2, 128, d_k)
    case24.t_input = (q @ k.transpose(-2, -1)) / (d_k ** 0.5)
    case24.t_input = case24.t_input.reshape(2, -1)[:, :128]  # Flatten for consistency
    case_list.append(case24)

    # Case 25: Sparse large values (few high-attention tokens)
    case25 = Case()
    case25.t_input = torch.ones(4, 128) * (-20)
    # Randomly select 5% positions to be high attention
    high_attention_mask = torch.rand(4, 128) < 0.05
    case25.t_input[high_attention_mask] = torch.rand(high_attention_mask.sum()) * 10 + 10
    case_list.append(case25)

    # Save the cases to the specified path
    save_all_cases_by_name(case_list, "softmax", "bf16")
    print(f"Successfully generated {len(case_list)} test cases")
