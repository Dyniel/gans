import torch, numpy as np

# ──────────────────
#  Noise injection
# ──────────────────
def noise_injection(x, sigma_start=0.1, sigma_end=0.0,
                    step=None, total=None):
    """Dodaje gaussowski szum do wejścia Discriminatora."""
    if sigma_start == 0:
        return x
    if step is not None and total is not None:
        sigma = sigma_start + (sigma_end - sigma_start) * step / total
    else:
        sigma = sigma_start
    return x + torch.randn_like(x) * sigma