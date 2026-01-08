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

    # Case 2: Large positive logits (overflow risk)
    case2 = Case()
    case2.t_logits = torch.rand(batch_size, vocab_size) * 50 + 50
    case2.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case2)

    # Case 3: Large negative logits (underflow risk)
    case3 = Case()
    case3.t_logits = torch.rand(batch_size, vocab_size) * (-50) - 50
    case3.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case3)

    # Case 4: Mixed extreme logits
    case4 = Case()
    case4.t_logits = torch.randn(batch_size, vocab_size) * 30
    case4.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case4)

    # Case 5: One dominant logit (confident prediction)
    case5 = Case()
    case5.t_logits = torch.randn(batch_size, vocab_size) * 0.1
    for i in range(batch_size):
        dominant_idx = torch.randint(0, vocab_size, (1,)).item()
        case5.t_logits[i, dominant_idx] = 50.0
    case5.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case5)

    # Case 6: All equal logits (maximum uncertainty)
    case6 = Case()
    case6.t_logits = torch.ones(batch_size, vocab_size) * 5.0
    case6.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case6)

    # Case 7: Near-zero logits
    case7 = Case()
    case7.t_logits = torch.randn(batch_size, vocab_size) * 1e-3
    case7.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case7)

    # Case 8: Large vocabulary (stress test)
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

    # Case 10: Near fp32 precision limit
    case10 = Case()
    case10.t_logits = torch.ones(batch_size, vocab_size) * 16777216.0 + torch.randn(batch_size, vocab_size)
    case10.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case10)

    # Case 11: Correct class has very negative logit (worst case)
    case11 = Case()
    case11.t_logits = torch.randn(batch_size, vocab_size) * 2
    case11.t_targets = torch.randint(0, vocab_size, (batch_size,))
    for i in range(batch_size):
        case11.t_logits[i, case11.t_targets[i]] = -50.0
    case_list.append(case11)

    # Case 12: Correct class has very positive logit (best case)
    case12 = Case()
    case12.t_logits = torch.randn(batch_size, vocab_size) * 2
    case12.t_targets = torch.randint(0, vocab_size, (batch_size,))
    for i in range(batch_size):
        case12.t_logits[i, case12.t_targets[i]] = 50.0
    case_list.append(case12)

    # Case 13: Gradient-like small values
    case13 = Case()
    case13.t_logits = torch.rand(batch_size, vocab_size) * 0.01 + 1e-4
    case13.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case13)

    # Case 14: Moderate range logits (balanced distribution)
    case14 = Case()
    case14.t_logits = torch.randn(batch_size, vocab_size) * 5
    case14.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case14)

    # Case 15: Medium variance logits (typical range)
    case15 = Case()
    case15.t_logits = torch.randn(batch_size, vocab_size) * 5
    case15.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case15)

    # Case 16: Alternating high/low confidence
    case16 = Case()
    case16.t_logits = torch.randn(batch_size, vocab_size) * 0.5
    case16.t_targets = torch.randint(0, vocab_size, (batch_size,))
    for i in range(0, batch_size, 2):
        case16.t_logits[i, case16.t_targets[i]] = 30.0  # High confidence
    for i in range(1, batch_size, 2):
        case16.t_logits[i, case16.t_targets[i]] = -30.0  # Low confidence
    case_list.append(case16)

    # Case 17: All positive logits (half-normal distribution)
    case17 = Case()
    case17.t_logits = torch.randn(batch_size, vocab_size).abs() * 10
    case17.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case17)

    # Case 18: Single sample
    case18 = Case()
    case18.t_logits = torch.randn(1, vocab_size) * 5
    case18.t_targets = torch.randint(0, vocab_size, (1,))
    case_list.append(case18)

    # Case 19: Sparse high logits
    case19 = Case()
    case19.t_logits = torch.ones(batch_size, vocab_size) * (-10)
    for i in range(batch_size):
        high_indices = torch.randint(0, vocab_size, (5,))
        case19.t_logits[i, high_indices] = 20.0
    case19.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case19)

    # Case 20: All targets same class
    case20 = Case()
    case20.t_logits = torch.randn(batch_size, vocab_size) * 5
    case20.t_targets = torch.ones(batch_size, dtype=torch.long) * (vocab_size // 2)
    case_list.append(case20)

    # Case 21: Logits near overflow boundary
    case21 = Case()
    case21.t_logits = torch.rand(batch_size, vocab_size) * 10 + 80
    case21.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case21)

    # Case 22: Small differences in logits (precision test)
    case22 = Case()
    base_logit = 10.0
    case22.t_logits = torch.ones(batch_size, vocab_size) * base_logit
    case22.t_logits += torch.randn(batch_size, vocab_size) * 1e-6
    case22.t_targets = torch.randint(0, vocab_size, (batch_size,))
    case_list.append(case22)

    # Save the cases to the specified path
    save_all_cases_by_name(case_list, "crossentropy", "fp32")
    print(f"Successfully generated {len(case_list)} test cases")
