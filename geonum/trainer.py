"""
Training utilities: logging, checkpointing, loss/decode helpers.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


# ── Logging ──────────────────────────────────────────────────────────────────

class Tee:
    """Write simultaneously to a file and another stream (e.g. stderr).
    File is opened in write mode so each run starts fresh."""

    def __init__(self, path: str, stream=None):
        self._f = open(path, "w")
        self._s = stream or sys.stderr

    def write(self, msg):
        self._f.write(msg); self._f.flush()
        self._s.write(msg)

    def flush(self):
        self._f.flush(); self._s.flush()

    def close(self):
        self._f.close()


def log(file, msg: str):
    """Write a timestamped line to an open log file and stdout."""
    file.write(msg + "\n"); file.flush()
    print(msg, flush=True)


def append_jsonl(path: str, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Numerical output head: loss and decode ───────────────────────────────────

def numerical_loss(logits, targets, n_digits,
                   w_sign=2.0, w_digit=1.0, w_frac=10.0, w_mag=0.5):
    """Joint classification-regression loss on polar logits (Eq. 10)."""
    device = logits.device
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.float32, device=device)
    else:
        targets = targets.to(device)

    loss = w_sign * F.cross_entropy(logits[:, :2], (targets >= 0).long())

    abs_int = targets.abs().long()
    for d in range(n_digits):
        digit_tgt = (abs_int // (10 ** d)) % 10
        loss = loss + w_digit * (1.0 + 0.2 * d) * F.cross_entropy(
            logits[:, 2 + d * 10: 2 + (d + 1) * 10], digit_tgt)

    frac_tgt = targets.abs() - targets.abs().floor()
    loss = loss + w_frac * F.mse_loss(torch.sigmoid(logits[:, -2]), frac_tgt)

    mag_tgt = torch.sign(targets) * torch.log(targets.abs() + 1.0)
    loss = loss + w_mag  * F.mse_loss(logits[:, -1], mag_tgt)
    return loss


def decode_logits(logits, n_digits) -> torch.Tensor:
    """Decode polar logits to scalar predictions."""
    sign = torch.where(logits[:, :2].argmax(dim=1) == 1,
                       torch.ones(logits.shape[0],  device=logits.device),
                       -torch.ones(logits.shape[0], device=logits.device))
    int_part = torch.zeros(logits.shape[0], device=logits.device)
    for d in range(n_digits):
        int_part += (logits[:, 2 + d * 10: 2 + (d + 1) * 10]
                     .argmax(dim=1).float() * (10 ** d))
    return sign * (int_part + torch.sigmoid(logits[:, -2]))
