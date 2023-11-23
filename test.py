import numpy as np
import torch

# instance norm
# mae loss
# flatten head -> channel mix
# add std input

L = 5
a = torch.arange(L).repeat(L,1)
_mask = torch.abs(torch.arange(L).reshape(-1, 1) - a) < 3
_mask = _mask.bool()
print(_mask.dtype)