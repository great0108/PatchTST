import numpy as np
import torch

L = 5
a = torch.arange(L).repeat(L,1)
_mask = torch.abs(torch.arange(L).reshape(-1, 1) - a) < 3
_mask = _mask.bool()
print(_mask.dtype)