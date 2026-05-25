"""
Dataset and evaluation utilities for GeoNum Stage I and Stage II/III.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset

OPERATORS  = ["+", "-", "*"]
OP_TO_IDX  = {op: i for i, op in enumerate(OPERATORS)}


# ── Stage I: scalar operand corpus ──────────────────────────────────────────

class ScalarDataset(Dataset):
    """Operand corpus for Stage I encoder pretraining.

    NUPA records carry explicit 'a'/'b' fields; operands are read directly.
    NumericBench / FERMAT records lack operand fields, so scalars are sampled
    from ±[scalar_lo, scalar_hi] to match the dataset's value range.
    """

    def __init__(self, json_path: str,
                 scalar_lo: float = None, scalar_hi: float = None,
                 log_uniform: bool = False, seed: int = 42):
        with open(json_path) as f:
            records = json.load(f)

        if "a" in records[0] and "b" in records[0]:
            scalars = []
            for r in records:
                scalars.append(r["a"])
                scalars.append(r["b"])
            self.values = torch.tensor(scalars, dtype=torch.float32)
        else:
            n   = len(records) * 2
            rng = np.random.default_rng(seed)
            if log_uniform:
                mags = np.exp(rng.uniform(np.log(scalar_lo), np.log(scalar_hi), n))
            else:
                mags = rng.uniform(scalar_lo, scalar_hi, n)
            signs = rng.choice([-1.0, 1.0], size=n)
            self.values = torch.tensor((signs * mags).astype(np.float32))

    def __len__(self):  return len(self.values)
    def __getitem__(self, idx):  return self.values[idx]


@torch.no_grad()
def evaluate_encoder(model, loader, device) -> dict:
    """Evaluate encoder reconstruction accuracy.  Returns mae, acc5/1/01, arrays."""
    model.eval()
    true_list, pred_list = [], []
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch)
        int_p = sum(torch.argmax(dl, dim=1).float() * (10 ** i)
                    for i, dl in enumerate(out["digit_logits"]))
        sign_p = (torch.argmax(out["sign_logits"], dim=1) * 2 - 1).float()
        pred   = sign_p * (int_p + out["frac_pred"])
        true_list.append(batch.cpu())
        pred_list.append(pred.cpu())

    true_arr  = torch.cat(true_list).numpy()
    pred_arr  = torch.cat(pred_list).numpy()
    rel_error = np.abs((pred_arr - true_arr) / (np.abs(true_arr) + 1e-8))
    return dict(
        mae   = float(np.mean(np.abs(pred_arr - true_arr))),
        acc5  = float(np.mean(rel_error < 0.05)),
        acc1  = float(np.mean(rel_error < 0.01)),
        acc01 = float(np.mean(rel_error < 0.001)),
        true  = true_arr,
        pred  = pred_arr,
        rel   = rel_error,
    )


# ── Stage II/III: arithmetic pair datasets ───────────────────────────────────

class ArithmeticDataset(Dataset):
    """Arithmetic pair dataset for Stage II and Stage III.

    Supports NUPA (a/b/op/result fields), NumericBench (question/answer text),
    and FERMAT (operands sampled from [0.1, 500]).
    """

    def __init__(self, json_path: str, dataset_type: str = "nupa", seed: int = 42):
        with open(json_path) as f:
            records = json.load(f)

        if dataset_type == "nupa":
            self.data = [{"a": r["a"], "b": r["b"],
                          "op_idx": OP_TO_IDX[r["op"]], "result": r["result"]}
                         for r in records]

        elif dataset_type == "numericbench":
            self.data = []
            for r in records:
                inner = r["question"].replace("What is ", "").rstrip("?").strip()
                if " + " in inner:
                    a, b = inner.split(" + ");  op = "+"
                else:
                    a, b = inner.split(" - ");  op = "-"
                self.data.append({"a": float(a), "b": float(b),
                                  "op_idx": OP_TO_IDX[op], "result": float(r["answer"])})

        elif dataset_type == "fermat":
            rng     = np.random.default_rng(seed)
            n       = len(records)
            a_vals  = rng.uniform(0.1, 500.0, n).astype(np.float32)
            b_vals  = rng.uniform(0.1, 500.0, n).astype(np.float32)
            self.data = [{"a": float(a_vals[i]), "b": float(b_vals[i]),
                          "op_idx": 0, "result": float(a_vals[i] + b_vals[i])}
                         for i in range(n)]

    def __len__(self):  return len(self.data)

    def __getitem__(self, idx):
        r = self.data[idx]
        return {"input_nums": [r["a"], r["b"]],
                "op_idx":     r["op_idx"],
                "output_num": r["result"]}


def evaluate_predictions(predictions, targets) -> dict:
    """Compute MAE, ACC@1%, ACC@0.1% from prediction lists."""
    true_arr  = np.array(targets)
    pred_arr  = np.array(predictions)
    rel_error = np.abs((pred_arr - true_arr) / (np.abs(true_arr) + 1e-8))
    return dict(
        mae  = float(np.mean(np.abs(pred_arr - true_arr))),
        acc1 = float(np.mean(rel_error < 0.01)),
        acc01= float(np.mean(rel_error < 0.001)),
    )
