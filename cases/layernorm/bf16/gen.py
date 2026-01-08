import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from util.case_util import save_all_cases_by_name
from cases.layernorm.case import Case


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    case_list = []
    hidden_size = 768  # Typical LLM hidden size

    # Helper function to create default weight and bias
    def create_params(size):
        weight = torch.ones(size)
        bias = torch.zeros(size)
        return weight, bias

    # Case 1: Normal distribution (typical activations in BF16)
    case1 = Case()
    case1.t_input = torch.randn(4, 128, hidden_size)
    case1.t_weight, case1.t_bias = create_params(hidden_size)
    case_list.append(case1)

    # Case 2: Large positive values (tests mean/variance stability)
    case2 = Case()
    case2.t_input = torch.rand(4, 128, hidden_size) * 50 + 50
    case2.t_weight, case2.t_bias = create_params(hidden_size)
    case_list.append(case2)

    # Case 3: Large negative values
    case3 = Case()
    case3.t_input = torch.rand(4, 128, hidden_size) * (-50) - 50
    case3.t_weight, case3.t_bias = create_params(hidden_size)
    case_list.append(case3)

    # Case 4: Mixed extreme values (wide dynamic range)
    case4 = Case()
    case4.t_input = torch.randn(4, 128, hidden_size) * 50
    case4.t_weight, case4.t_bias = create_params(hidden_size)
    case_list.append(case4)

    # Case 5: Very small variance (BF16 precision challenge)
    # BF16 can lose precision when variance is very small
    case5 = Case()
    base = torch.ones(4, 128, hidden_size) * 10.0
    noise = torch.randn(4, 128, hidden_size) * 1e-3  # Larger than FP32 to be representable
    case5.t_input = base + noise
    case5.t_weight, case5.t_bias = create_params(hidden_size)
    case_list.append(case5)

    # Case 6: Near-zero values (small gradients)
    case6 = Case()
    case6.t_input = torch.randn(4, 128, hidden_size) * 1e-3
    case6.t_weight, case6.t_bias = create_params(hidden_size)
    case_list.append(case6)

    # Case 7: All equal values (zero variance - division by epsilon)
    case7 = Case()
    case7.t_input = torch.ones(4, 128, hidden_size) * 5.0
    case7.t_weight, case7.t_bias = create_params(hidden_size)
    case_list.append(case7)

    # Case 8: Large hidden dimension (4096 for large LLMs with BF16)
    case8 = Case()
    large_hidden = 4096
    case8.t_input = torch.randn(2, 64, large_hidden) * 2
    case8.t_weight, case8.t_bias = create_params(large_hidden)
    case_list.append(case8)

    # Case 9: Non-unit weight and bias (amplifies rounding errors)
    case9 = Case()
    case9.t_input = torch.randn(4, 128, hidden_size) * 5
    case9.t_weight = torch.randn(hidden_size) * 2 + 1
    case9.t_bias = torch.randn(hidden_size) * 0.5
    case_list.append(case9)

    # Case 10: Extreme weight values (overflow risk in output)
    case10 = Case()
    case10.t_input = torch.randn(4, 128, hidden_size)
    case10.t_weight = torch.ones(hidden_size) * 100
    case10.t_bias = torch.ones(hidden_size) * 50
    case_list.append(case10)

    # Case 11: Very small weight values (underflow risk)
    case11 = Case()
    case11.t_input = torch.randn(4, 128, hidden_size) * 10
    case11.t_weight = torch.ones(hidden_size) * 1e-3  # BF16 representable
    case11.t_bias = torch.ones(hidden_size) * 1e-3
    case_list.append(case11)

    # Case 12: Outliers in input (affects mean/variance)
    case12 = Case()
    case12.t_input = torch.randn(4, 128, hidden_size)
    case12.t_input[0, 0, 0] = 1000.0  # Outlier
    case12.t_input[1, 10, 10] = -1000.0  # Outlier
    case12.t_weight, case12.t_bias = create_params(hidden_size)
    case_list.append(case12)

    # Case 13: Gradient-like small values
    case13 = Case()
    case13.t_input = torch.rand(4, 128, hidden_size) * 0.01 + 1e-4
    case13.t_weight, case13.t_bias = create_params(hidden_size)
    case_list.append(case13)

    # Case 14: Long sequence (typical for LLM inference)
    case14 = Case()
    case14.t_input = torch.randn(2, 2048, hidden_size) * 3
    case14.t_weight, case14.t_bias = create_params(hidden_size)
    case_list.append(case14)

    # Case 15: Alternating high/low variance across features
    case15 = Case()
    case15.t_input = torch.randn(4, 128, hidden_size)
    case15.t_input[:, :, :hidden_size//2] *= 0.01  # Low variance
    case15.t_input[:, :, hidden_size//2:] *= 10.0  # High variance
    case15.t_weight, case15.t_bias = create_params(hidden_size)
    case_list.append(case15)

    # Case 16: Values at BF16 precision boundary
    # Test rounding at BF16's 7-bit mantissa limit
    case16 = Case()
    base = torch.ones(4, 128, hidden_size) * 16.0  # Power of 2
    # Add noise at BF16 precision boundary (~0.125 for value 16)
    noise = torch.randn(4, 128, hidden_size) * 0.01
    case16.t_input = base + noise
    case16.t_weight, case16.t_bias = create_params(hidden_size)
    case_list.append(case16)

    # Case 17: Catastrophic cancellation test (large mean, small variance)
    case17 = Case()
    base = 1000.0
    case17.t_input = torch.ones(4, 128, hidden_size) * base + torch.randn(4, 128, hidden_size) * 0.1
    case17.t_weight, case17.t_bias = create_params(hidden_size)
    case_list.append(case17)

    # Case 18: Negative weight (unusual but valid)
    case18 = Case()
    case18.t_input = torch.randn(4, 128, hidden_size) * 5
    case18.t_weight = -torch.ones(hidden_size) * 2
    case18.t_bias = torch.zeros(hidden_size)
    case_list.append(case18)

    # Case 19: Single batch dimension
    case19 = Case()
    case19.t_input = torch.randn(1, 512, hidden_size) * 2
    case19.t_weight, case19.t_bias = create_params(hidden_size)
    case_list.append(case19)

    # Case 20: Powers of 2 (exactly representable in BF16)
    case20 = Case()
    powers = torch.tensor([2**i for i in range(-5, 5)])  # 2^-5 to 2^4
    case20.t_input = powers.repeat(4, 128, hidden_size // len(powers) + 1)[:, :, :hidden_size]
    case20.t_weight, case20.t_bias = create_params(hidden_size)
    case_list.append(case20)

    # Case 21: Mixed precision-critical weight and bias
    case21 = Case()
    case21.t_input = torch.randn(4, 128, hidden_size) * 3
    # Weight varies from very small to moderate
    case21.t_weight = torch.linspace(0.01, 2.0, hidden_size)
    case21.t_bias = torch.linspace(-1.0, 1.0, hidden_size)
    case_list.append(case21)

    # Case 22: Variance computation overflow test
    # Large values where x^2 might overflow in BF16 intermediate computation
    case22 = Case()
    case22.t_input = torch.randn(4, 128, hidden_size) * 1e3
    case22.t_weight, case22.t_bias = create_params(hidden_size)
    case_list.append(case22)

    # Case 23: Near-subnormal values (BF16 min normal ~1.175e-38)
    case23 = Case()
    case23.t_input = torch.randn(4, 128, hidden_size) * 1e-6
    case23.t_weight, case23.t_bias = create_params(hidden_size)
    case_list.append(case23)

    # Case 24: Systematic rounding test (adjacent BF16 values)
    case24 = Case()
    # Create values that differ by smallest BF16 step at scale 1.0
    case24.t_input = torch.arange(hidden_size).float().unsqueeze(0).unsqueeze(0) * 0.01
    case24.t_input = case24.t_input.repeat(4, 128, 1)
    case24.t_weight, case24.t_bias = create_params(hidden_size)
    case_list.append(case24)

    # Case 25: Simulated post-attention activations
    # Typical pattern after softmax @ V in transformer
    case25 = Case()
    # Softmax outputs sum to 1, so weighted values are moderate
    case25.t_input = torch.randn(4, 128, hidden_size).abs() * 2
    case25.t_weight, case25.t_bias = create_params(hidden_size)
    case_list.append(case25)

    # Case 26: Sparse large values (few outliers)
    case26 = Case()
    case26.t_input = torch.randn(4, 128, hidden_size) * 0.5
    # Add sparse outliers
    outlier_mask = torch.rand(4, 128, hidden_size) < 0.01
    case26.t_input[outlier_mask] = torch.randn(outlier_mask.sum()) * 50
    case26.t_weight, case26.t_bias = create_params(hidden_size)
    case_list.append(case26)

    # Case 27: Bimodal distribution (two distinct value clusters)
    case27 = Case()
    case27.t_input = torch.zeros(4, 128, hidden_size)
    mask = torch.rand(4, 128, hidden_size) < 0.5
    case27.t_input[mask] = torch.randn(mask.sum()) * 2 + 10
    case27.t_input[~mask] = torch.randn((~mask).sum()) * 2 - 10
    case27.t_weight, case27.t_bias = create_params(hidden_size)
    case_list.append(case27)

    # Case 28: Extreme variance (very wide spread)
    case28 = Case()
    case28.t_input = torch.randn(4, 128, hidden_size) * 100
    case28.t_weight, case28.t_bias = create_params(hidden_size)
    case_list.append(case28)

    # Case 29: Weight and bias at BF16 epsilon boundary
    case29 = Case()
    case29.t_input = torch.randn(4, 128, hidden_size) * 5
    # BF16 epsilon at 1.0 is 2^-7 = 0.0078125
    case29.t_weight = torch.ones(hidden_size) + torch.randn(hidden_size) * 0.001
    case29.t_bias = torch.randn(hidden_size) * 0.001
    case_list.append(case29)

    # Case 30: Very large batch (memory and accumulation precision)
    case30 = Case()
    case30.t_input = torch.randn(32, 256, hidden_size) * 3
    case30.t_weight, case30.t_bias = create_params(hidden_size)
    case_list.append(case30)

    # Save the cases to the specified path
    save_all_cases_by_name(case_list, "layernorm", "bf16")
    print(f"Successfully generated {len(case_list)} test cases")
