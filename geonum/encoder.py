"""
GeoNumEncoder — polar-coordinate numerical encoder (paper §3).

Polar feature vector (Eq. 5):
    r(x) = [s,  d̄₀, …, d̄_{N-1},  cos 2πf,  sin 2πf]

Position-aware digit feature (Eq. 4):
    d̄ᵢ = dᵢ + (Rᵢ + f) / 10^i

Learnable type embedding p (Eq. 6) is added before the shared MLP.

Joint pretraining loss (Eq. 10):
    L = α·Ls  +  β·Σᵢ ωᵢ·Lᵢ  +  γ·Lf  +  δ·Lm
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeoNumEncoder(nn.Module):
    """Polar numerical encoder (§3, Eq. 2–10).

    Args:
        embed_dim: Output embedding dimension d.
        n_digits:  Number of integer-digit positions N.
    """

    def __init__(self, embed_dim: int = 256, n_digits: int = 6):
        super().__init__()
        self.n_digits  = n_digits
        self.input_dim = n_digits + 3      # s + N·d̄ᵢ + cos2πf + sin2πf

        # Learnable type embedding p added to r(x) before encoding (Eq. 6)
        self.type_embed = nn.Parameter(torch.zeros(self.input_dim))

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, embed_dim),      nn.LayerNorm(embed_dim),
        )

        # Prediction heads for self-supervised pretraining (Eq. 7–9)
        self.sign_head      = nn.Linear(embed_dim, 2)
        self.digit_heads    = nn.ModuleList(
            [nn.Linear(embed_dim, 10) for _ in range(n_digits)])
        self.frac_head      = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(),
            nn.Linear(128, 1),         nn.Sigmoid())
        self.magnitude_head = nn.Linear(embed_dim, 1)

    def polar_encode(self, x: torch.Tensor):
        """Decompose scalar x into polar feature vector r(x) (Eq. 4–5)."""
        sign     = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        abs_x    = x.abs()
        int_part = abs_x.floor().long()
        frac     = abs_x - int_part.float()

        cos_f = torch.cos(2 * math.pi * frac)
        sin_f = torch.sin(2 * math.pi * frac)

        # d̄ᵢ = dᵢ + (Rᵢ + f) / 10^i  (Eq. 4)
        d_bars = []
        for i in range(self.n_digits):
            p_i = 10 ** i
            di  = (int_part // p_i) % 10
            Ri  = int_part % p_i
            d_bars.append(di.float() + (Ri.float() + frac) / p_i)

        r           = torch.stack([sign] + d_bars + [cos_f, sin_f], dim=1)
        true_digits = [(int_part // (10 ** i)) % 10 for i in range(self.n_digits)]
        return r, sign, int_part, frac, true_digits

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [B] scalar tensor.
        Returns dict: h, sign_logits, digit_logits, frac_pred, mag_pred,
                      true_sign, true_int, true_frac, true_digits.
        """
        r, sign, int_part, frac, true_digits = self.polar_encode(x)
        h = self.encoder(r + self.type_embed)
        return dict(
            h            = h,
            sign_logits  = self.sign_head(h),
            digit_logits = [head(h) for head in self.digit_heads],
            frac_pred    = self.frac_head(h).squeeze(-1),
            mag_pred     = self.magnitude_head(h).squeeze(-1),
            true_sign    = sign,
            true_int     = int_part,
            true_frac    = frac,
            true_digits  = true_digits,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return embedding h without auxiliary outputs."""
        r, *_ = self.polar_encode(x)
        return self.encoder(r + self.type_embed)


def compute_loss(out: dict,
                 alpha: float = 2.0,
                 beta:  float = 1.0,
                 gamma: float = 20.0,
                 delta: float = 0.5) -> dict:
    """Joint classification-regression loss L (Eq. 10).

    L = α·Ls  +  β·Σᵢ ωᵢ·Lᵢ  +  γ·Lf  +  δ·Lm
    where ωᵢ = 1 + 0.2·i weights higher-order digit positions more.
    """
    sign_loss  = F.cross_entropy(out["sign_logits"], (out["true_sign"] > 0).long())

    digit_loss = out["sign_logits"].new_zeros(1).squeeze()
    for i, (logits, tgt) in enumerate(zip(out["digit_logits"], out["true_digits"])):
        digit_loss = digit_loss + (1.0 + 0.2 * i) * F.cross_entropy(logits, tgt)

    frac_loss  = F.mse_loss(out["frac_pred"], out["true_frac"])

    x_val    = out["true_sign"] * (out["true_int"].float() + out["true_frac"])
    mag_loss = F.mse_loss(out["mag_pred"],
                          torch.sign(x_val) * torch.log(x_val.abs() + 1.0))

    total = alpha * sign_loss + beta * digit_loss + gamma * frac_loss + delta * mag_loss
    return dict(total=total, sign=sign_loss, digit=digit_loss, frac=frac_loss, mag=mag_loss)
