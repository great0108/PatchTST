import numpy as np
import torch

L = 5
a = torch.arange(L).repeat(L,1)
mask = torch.abs(torch.arange(L).reshape(-1, 1) - a) > 1

print(mask)