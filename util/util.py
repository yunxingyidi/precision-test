import sys
import os
import importlib.util
import torch

if importlib.util.find_spec("torch_npu") is not None:
    try:
        print("Importing torch_npu")
        import torch_npu
        torch.npu.set_device(0)
        assert torch.npu.is_available()
    except:
        exit(1)
