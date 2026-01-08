import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from util.case_util import save_all_cases_by_name
from cases.crossentropy.case import Case


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)

    case_list = []
    batch_size = 16
    seq_len = 128
    vocab_size = 50000  # Typical LLM vocab size

    # Case 1: Normal distribution logits (typical case)
    case1 = Case()
    case1.t_logits = torch.randn(batch_size, vocab_size) * 2
    case1.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case1)

    # Case 2: Large positive logits (overflow risk in softmax)
    case2 = Case()
    case2.t_logits = torch.rand(batch_size, vocab_size) * 50 + 50
    case2.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case2)

    # Case 3: Large negative logits (underflow in exp())
    case3 = Case()
    case3.t_logits = torch.rand(batch_size, vocab_size) * (-50) - 50
    case3.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case3)

    # Case 4: Mixed extreme logits (dynamic range test)
    case4 = Case()
    case4.t_logits = torch.randn(batch_size, vocab_size) * 30
    case4.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case4)

    # Case 5: One dominant logit per sample (tests softmax with single very high value)
    case5 = Case()
    case5.t_logits = torch.randn(batch_size, vocab_size) * 0.1
    for i in range(batch_size):
        dominant_idx = torch.randint(0, vocab_size, (1,)).item()
        case5.t_logits[i, dominant_idx] = 50.0
    case5.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case5)

    # Case 6: All equal logits (maximum uncertainty, tests log(1/vocab_size))
    case6 = Case()
    case6.t_logits = torch.ones(batch_size, vocab_size) * 5.0
    case6.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case6)

    # Case 7: Near-zero logits
    case7 = Case()
    case7.t_logits = torch.randn(batch_size, vocab_size) * 1e-3
    case7.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case7)

    # Case 8: Large vocabulary (accumulation error in logsumexp)
    case8 = Case()
    large_vocab = 100000
    case8.t_logits = torch.randn(8, large_vocab) * 3
    case8.t_targets = torch.randint(0, large_vocab, (8,))
    case_list.append(case8)

    # Case 9: Small vocabulary
    case9 = Case()
    small_vocab = 100
    case9.t_logits = torch.randn(batch_size, small_vocab) * 3
    case9.t_targets = torch.randint(0, small_vocab, (batch_size,))
    case_list.append(case9)

    # Case 10: Correct class has very negative logit (worst case, high loss)
    case10 = Case()
    case10.t_logits = torch.randn(batch_size, vocab_size) * 2
    case10.t_targets = torch.randint(0, vocab_size, (batch_size,))
    for i in range(batch_size):
        case10.t_logits[i, case10.t_targets[i]] = -50.0
    case_list.append(case10)

    # Case 11: Correct class has very positive logit (best case, near-zero loss)
    case11 = Case()
    case11.t_logits = torch.randn(batch_size, vocab_size) * 2
    case11.t_targets = torch.randint(0, vocab_size, (batch_size,))
    for i in range(batch_size):
        case11.t_logits[i, case11.t_targets[i]] = 50.0
    case_list.append(case11)

    # Case 12: Gradient-like small values
    case12 = Case()
    case12.t_logits = torch.rand(batch_size, vocab_size) * 0.01 + 1e-4
    case12.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case12)

    # Case 13: Alternating high/low confidence
    case13 = Case()
    case13.t_logits = torch.randn(batch_size, vocab_size) * 0.5
    case13.t_targets = torch.randint(0, vocab_size, (batch_size,))
    for i in range(0, batch_size, 2):
        case13.t_logits[i, case13.t_targets[i]] = 30.0  # High confidence
    for i in range(1, batch_size, 2):
        case13.t_logits[i, case13.t_targets[i]] = -30.0  # Low confidence
    case_list.append(case13)

    # Case 14: Single sample
    case14 = Case()
    case14.t_logits = torch.randn(1, vocab_size) * 5
    case14.t_targets = torch.randint(0, vocab_size, (1,))
    case_list.append(case14)

    # Case 15: Sparse high logits (few confident predictions)
    case15 = Case()
    case15.t_logits = torch.ones(batch_size, vocab_size) * (-10)
    for i in range(batch_size):
        high_indices = torch.randint(0, vocab_size, (5,))
        case15.t_logits[i, high_indices] = 20.0
    case15.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case15)

    # Case 16: All targets same class
    case16 = Case()
    case16.t_logits = torch.randn(batch_size, vocab_size) * 5
    case16.t_targets = torch.ones(batch_size, dtype=torch.long) * (vocab_size // 2)
    case_list.append(case16)

    # Case 17: Logits near overflow boundary
    case17 = Case()
    case17.t_logits = torch.rand(batch_size, vocab_size) * 10 + 80
    case17.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case17)

    # Case 18: Small differences in logits (BF16 precision test)
    # BF16 can only distinguish differences > ~0.01 at this scale
    case18 = Case()
    base_logit = 10.0
    case18.t_logits = torch.ones(batch_size, vocab_size) * base_logit
    case18.t_logits += torch.randn(batch_size, vocab_size) * 0.01  # Larger than FP32
    case18.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case18)

    # Case 19: BF16 precision boundary values
    # Test logits at BF16's 7-bit mantissa limit
    case19 = Case()
    case19.t_logits = torch.ones(batch_size, vocab_size) * 16.0  # Power of 2
    case19.t_logits += torch.randn(batch_size, vocab_size) * 0.125  # BF16 epsilon at 16
    case19.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case19)

    # Case 20: Powers of 2 (exactly representable in BF16)
    case20 = Case()
    powers = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    case20.t_logits = powers.repeat(batch_size, vocab_size // len(powers) + 1)[:, :vocab_size]
    case20.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case20)

    # Case 21: Logsumexp stability test (very similar logits)
    # Tests numerical stability when all logits are nearly equal
    case21 = Case()
    case21.t_logits = torch.ones(batch_size, vocab_size) * 100.0
    case21.t_logits += torch.randn(batch_size, vocab_size) * 0.1
    case21.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case21)

    # Case 22: Target class exactly at median logit (moderate loss)
    case22 = Case()
    case22.t_logits = torch.randn(batch_size, vocab_size) * 5
    case22.t_targets = torch.randint(0, vocab_size, (batch_size,))
    for i in range(batch_size):
        median_val = case22.t_logits[i].median()
        case22.t_logits[i, case22.t_targets[i]] = median_val
    case_list.append(case22)

    # Case 23: Extreme overflow scenario (near BF16 max)
    case23 = Case()
    case23.t_logits = torch.rand(batch_size, vocab_size) * 1e3 + 1e4
    case23.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case23)

    # Case 24: Log(small probability) test - target has second-lowest logit
    case24 = Case()
    case24.t_logits = torch.randn(batch_size, vocab_size) * 3
    case24.t_targets = torch.randint(0, vocab_size, (batch_size,))
    for i in range(batch_size):
        sorted_vals, sorted_idx = torch.sort(case24.t_logits[i])
        case24.t_logits[i, case24.t_targets[i]] = sorted_vals[1].item()
    case_list.append(case24)

    # Case 25: Bimodal logit distribution
    case25 = Case()
    case25.t_logits = torch.zeros(batch_size, vocab_size)
    mask = torch.rand(batch_size, vocab_size) < 0.5
    case25.t_logits[mask] = torch.randn(mask.sum()) * 2 + 10
    case25.t_logits[~mask] = torch.randn((~mask).sum()) * 2 - 10
    case25.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case25)

    # Case 26: Systematic rounding test (adjacent BF16 values)
    case26 = Case()
    # Create logits with systematic BF16 spacing
    base = 8.0  # Nice BF16 value
    case26.t_logits = (torch.arange(vocab_size).float() % 256) * 0.0625 + base
    case26.t_logits = case26.t_logits.unsqueeze(0).repeat(batch_size, 1)
    case26.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case26)

    # Case 27: Real LLM inference pattern (post-transformer logits)
    case27 = Case()
    # Simulate typical distribution after softmax temperature scaling
    case27.t_logits = torch.randn(batch_size, vocab_size) * 3
    # Add some high-probability tokens (common words)
    top_k = vocab_size // 100
    for i in range(batch_size):
        top_indices = torch.randint(0, vocab_size, (top_k,))
        case27.t_logits[i, top_indices] += 5.0
    case27.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case27)

    # Case 28: Target always has highest logit (perfect prediction)
    case28 = Case()
    case28.t_logits = torch.randn(batch_size, vocab_size) * 3
    case28.t_targets = torch.randint(0, vocab_size, (batch_size,))
    for i in range(batch_size):
        max_val = case28.t_logits[i].max() + 10.0
        case28.t_logits[i, case28.t_targets[i]] = max_val
    case_list.append(case28)

    # Case 29: Target always has lowest logit (worst prediction)
    case29 = Case()
    case29.t_logits = torch.randn(batch_size, vocab_size) * 3
    case29.t_targets = torch.randint(0, vocab_size, (batch_size,))
    for i in range(batch_size):
        min_val = case29.t_logits[i].min() - 10.0
        case29.t_logits[i, case29.t_targets[i]] = min_val
    case_list.append(case29)

    # Case 30: Very wide dynamic range within each sample
    case30 = Case()
    case30.t_logits = torch.zeros(batch_size, vocab_size)
    for i in range(batch_size):
        # Quarter of vocab at different scales
        q = vocab_size // 4
        case30.t_logits[i, :q] = torch.randn(q) * 0.1
        case30.t_logits[i, q:2*q] = torch.randn(q) * 5
        case30.t_logits[i, 2*q:3*q] = torch.randn(q) * 20
        case30.t_logits[i, 3*q:] = torch.randn(vocab_size - 3*q) * 50
    case30.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case30)

    # Case 31: Temperature-scaled logits (common in sampling)
    case31 = Case()
    temperature = 0.7
    case31.t_logits = torch.randn(batch_size, vocab_size) * 5 / temperature
    case31.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case31)

    # Case 32: Subnormal-adjacent values in BF16
    case32 = Case()
    case32.t_logits = torch.randn(batch_size, vocab_size) * 1e-6
    case32.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case32)

    # Save the cases to the specified path
    save_all_cases_by_name(case_list, "crossentropy", "bf16")
    print(f"Successfully generated {len(case_list)} test cases")
