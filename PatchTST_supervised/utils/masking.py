import torch


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class LocalMask():
    def __init__(self, L, tf_kernel_size, device="cpu"):
        with torch.no_grad():
            a = torch.arange(L).repeat(L,1)
            self._mask = torch.abs(torch.arange(L).reshape(-1, 1) - a) > tf_kernel_size
            self._mask = self._mask.bool().to(device)

    @property
    def mask(self):
        return self._mask
    
def localMask(L, kernel_size):
    with torch.no_grad():
        a = torch.arange(L).repeat(L,1)
        mask = torch.abs(torch.arange(L).reshape(-1, 1) - a) > kernel_size
        mask = mask.bool()
        return mask