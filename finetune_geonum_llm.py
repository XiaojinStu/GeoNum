"""Stage II and III: projection alignment and end-to-end LLM fine-tuning."""
import os, sys, json, argparse
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from geonum.encoder  import GeoNumEncoder
from geonum.data     import ArithmeticDataset, evaluate_predictions, OPERATORS, OP_TO_IDX
from geonum.trainer  import Tee, log, append_jsonl, numerical_loss, decode_logits
from geonum.viz      import plot_stage23_training

NUM_TOKEN = "<|num|>"


# ── Shared modules ────────────────────────────────────────────────────────────

class OperandProjection(nn.Module):
    """Projects GeoNum embeddings (R^d) to the LLM hidden dimension (R^D)."""
    def __init__(self, embed_dim: int, llm_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, 512), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(512, llm_dim))
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, h):
        return self.norm(self.proj(h))


class NumericalHead(nn.Module):
    """Maps an LLM-dim representation to polar numerical logits."""
    def __init__(self, llm_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, llm_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(llm_dim, 512),     nn.ReLU(),
            nn.Linear(512, out_dim))

    def forward(self, h):
        return self.mlp(h)


# ── Stage II model ────────────────────────────────────────────────────────────

class AlignmentDecoder(nn.Module):
    """MLP decoder for Stage II projection alignment.
    The intermediate h_dec serves as a frozen reference anchor in Stage III."""

    def __init__(self, embed_dim: int, n_digits: int, llm_dim: int):
        super().__init__()
        self.n_digits       = n_digits
        self.projection     = OperandProjection(embed_dim, llm_dim)
        self.operator_embed = nn.Embedding(len(OPERATORS), llm_dim)
        self.mixer          = nn.Sequential(
            nn.Linear(llm_dim * 3, llm_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(llm_dim, llm_dim))
        self.head           = NumericalHead(llm_dim, 2 + 10 * n_digits + 2)

    def forward(self, h_a, h_b, op_idx):
        h_dec = self.mixer(torch.cat([
            self.projection(h_a),
            self.projection(h_b),
            self.operator_embed(op_idx)], dim=1))
        return self.head(h_dec), h_dec


# ── Stage III model ───────────────────────────────────────────────────────────

class GeoNumLLM(nn.Module):
    """LLM with LoRA and GeoNum operand injection (Stage III).

    GeoNum embeddings replace <|num|> token embeddings in the prompt.
    h_fused = w_LLM · h_LLM + w_dec · h_dec  (Eq. 11)
    """

    def __init__(self, llm_path: str, stage2_ckpt: str,
                 embed_dim: int, n_digits: int, llm_dim: int):
        super().__init__()
        self.n_digits = n_digits
        self.llm_dim  = llm_dim

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({"additional_special_tokens": [NUM_TOKEN]})
        self.num_token_id = self.tokenizer.convert_tokens_to_ids(NUM_TOKEN)

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path, dtype=torch.float32, attn_implementation="eager")
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm = get_peft_model(self.llm, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))

        self.projection = OperandProjection(embed_dim, llm_dim)
        self.head       = NumericalHead(llm_dim, 2 + 10 * n_digits + 2)

        # Learned scalar blend weights w_LLM, w_dec (Eq. 11)
        self.w_llm_raw = nn.Parameter(torch.ones(1))
        self.w_dec_raw = nn.Parameter(torch.ones(1))

        if stage2_ckpt and os.path.exists(stage2_ckpt):
            saved = torch.load(stage2_ckpt, map_location="cpu", weights_only=False)["net"]
            self.projection.load_state_dict(
                {k[len("projection."):]: v for k, v in saved.items()
                 if k.startswith("projection.")})
            self.head.load_state_dict(
                {k[len("head."):]: v for k, v in saved.items()
                 if k.startswith("head.")})
            print(f"  Loaded projection + head from {stage2_ckpt}", flush=True)

        self._prompt_tokens = {}
        for op in OPERATORS:
            text = (f"<|begin_of_text|>"
                    f"<|start_header_id|>system<|end_header_id|>\n"
                    f"You are a precise arithmetic calculator.<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n"
                    f"Compute {NUM_TOKEN} {op} {NUM_TOKEN}. "
                    f"Give only the numeric result.<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n"
                    f"{NUM_TOKEN}<|eot_id|>")
            self._prompt_tokens[op] = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=False)["input_ids"]

    def forward(self, h_a, h_b, op_idx, h_dec_frozen):
        B, dev = h_a.shape[0], h_a.device
        emb    = self.llm.get_input_embeddings()
        proj_a = self.projection(h_a)
        proj_b = self.projection(h_b)

        op_names = [OPERATORS[int(i)] for i in op_idx]
        max_len  = max(self._prompt_tokens[op].shape[1] for op in set(op_names))
        input_embeds   = torch.zeros(B, max_len, self.llm_dim, device=dev)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long, device=dev)
        ans_positions  = torch.zeros(B, dtype=torch.long, device=dev)

        for op in set(op_names):
            ids     = self._prompt_tokens[op][0].to(dev)
            seq_len = ids.shape[0]
            pad_len = max_len - seq_len
            num_pos = (ids == self.num_token_id).nonzero(as_tuple=False).squeeze(-1)
            pa, pb, pans = num_pos[0].item(), num_pos[1].item(), num_pos[2].item()

            bidx = torch.tensor([b for b, o in enumerate(op_names) if o == op], device=dev)
            ge   = emb(ids).unsqueeze(0).expand(len(bidx), -1, -1).clone()
            ge[:, pa,   :] = proj_a[bidx]
            ge[:, pb,   :] = proj_b[bidx]
            ge[:, pans, :] = 0.0

            if pad_len > 0:
                ge = torch.cat([torch.zeros(len(bidx), pad_len, self.llm_dim, device=dev), ge], dim=1)
                mask_row = torch.cat([torch.zeros(pad_len, dtype=torch.long, device=dev),
                                      torch.ones(seq_len,  dtype=torch.long, device=dev)])
                ans_positions[bidx] = pans + pad_len
            else:
                mask_row = torch.ones(seq_len, dtype=torch.long, device=dev)
                ans_positions[bidx] = pans

            input_embeds[bidx]   = ge
            attention_mask[bidx] = mask_row.unsqueeze(0).expand(len(bidx), -1)

        hidden = self.llm(inputs_embeds=input_embeds, attention_mask=attention_mask,
                          output_hidden_states=True).hidden_states[-1]
        h_LLM  = hidden.gather(1, ans_positions.view(B, 1, 1).expand(B, 1, self.llm_dim)).squeeze(1)

        # Eq. 11: h_fused = w_LLM · h_LLM + w_dec · h_dec
        w = torch.softmax(torch.stack([self.w_llm_raw, self.w_dec_raw]), dim=0)
        return self.head(w[0] * h_LLM + w[1] * h_dec_frozen.to(dev))


# ── Encoder embedding cache ───────────────────────────────────────────────────

@torch.no_grad()
def cache_embeddings(encoder, dataset, device, desc=""):
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4,
                        collate_fn=lambda b: {
                            "input_nums":  [x["input_nums"]  for x in b],
                            "op_idx":      [x["op_idx"]      for x in b],
                            "output_nums": [x["output_num"]  for x in b]})
    ha, hb, ops, res = [], [], [], []
    for batch in tqdm(loader, desc=desc, ncols=90, ascii=False, file=sys.stderr):
        na = torch.tensor([x[0] for x in batch["input_nums"]], dtype=torch.float32, device=device)
        nb = torch.tensor([x[1] for x in batch["input_nums"]], dtype=torch.float32, device=device)
        ha.append(encoder(na)["h"].cpu())
        hb.append(encoder(nb)["h"].cpu())
        ops.append(torch.tensor(batch["op_idx"]))
        res.append(torch.tensor(batch["output_nums"], dtype=torch.float32))
    return torch.cat(ha), torch.cat(hb), torch.cat(ops), torch.cat(res)


# ── Main training ─────────────────────────────────────────────────────────────

def main(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{cfg.gpus[0]}")

    train_log    = open(os.path.join(cfg.out_dir, "train.log"), "a")
    progress_tee = Tee(os.path.join(cfg.out_dir, "progress.log"))

    log(train_log, f"GPU: {cfg.gpus[0]}  |  Stage II steps: {cfg.stage2_steps}"
                   f"  |  Stage III steps: {cfg.stage3_steps}")

    _dir = os.path.basename(os.path.normpath(cfg.data_dir)).lower()
    dtype = "numericbench" if "numericbench" in _dir else "fermat" if "fermat" in _dir else "nupa"
    log(train_log, f"Loading {dtype.upper()} dataset...")

    train_ds = ArithmeticDataset(os.path.join(cfg.data_dir, "train.json"), dtype)
    val_ds   = ArithmeticDataset(os.path.join(cfg.data_dir, "val.json"),   dtype)
    test_ds  = ArithmeticDataset(os.path.join(cfg.data_dir, "test.json"),  dtype)
    log(train_log, f"  train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")

    encoder = GeoNumEncoder(embed_dim=cfg.embed_dim, n_digits=cfg.enc_n_digits).to(device)
    encoder.load_state_dict(torch.load(cfg.encoder_ckpt, map_location="cpu", weights_only=False))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad = False

    log(train_log, "Caching GeoNum encoder embeddings...")
    tr_ha, tr_hb, tr_ops, tr_res = cache_embeddings(encoder, train_ds, device, "  train")
    va_ha, va_hb, va_ops, va_res = cache_embeddings(encoder, val_ds,   device, "  val  ")
    te_ha, te_hb, te_ops, te_res = cache_embeddings(encoder, test_ds,  device, "  test ")

    log_s2, log_s3 = [], []
    s2_path = os.path.join(cfg.out_dir, "log_stage2.jsonl")
    s3_path = os.path.join(cfg.out_dir, "log_stage3.jsonl")

    # ── Stage II ──────────────────────────────────────────────────────────────
    if cfg.stage in ("2", "23"):
        log(train_log, "\n" + "=" * 60)
        log(train_log, "STAGE II  —  Numerical-Textual Alignment")
        log(train_log, "=" * 60)

        decoder = AlignmentDecoder(cfg.embed_dim, cfg.n_digits, cfg.llm_dim).to(device)
        if len(cfg.gpus) > 1:
            dec_par = nn.DataParallel(decoder, device_ids=cfg.gpus)
        else:
            dec_par = decoder
        log(train_log, f"  AlignmentDecoder parameters: "
                       f"{sum(p.numel() for p in decoder.parameters())/1e6:.2f}M")

        opt2 = torch.optim.AdamW(decoder.parameters(), lr=cfg.lr_2, weight_decay=1e-4)
        sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, cfg.stage2_steps, eta_min=cfg.lr_2*0.1)

        @torch.no_grad()
        def eval_s2(ha, hb, ops, res, n=8192):
            dec_par.eval()
            idx = torch.randperm(len(res))[:n]
            logits, _ = dec_par(ha[idx].to(device), hb[idx].to(device), ops[idx].to(device))
            return evaluate_predictions(decode_logits(logits, cfg.n_digits).cpu().tolist(),
                                        res[idx].tolist())

        step, best = 0, 0.0
        loss_buf   = []
        pbar = tqdm(total=cfg.stage2_steps, desc="Stage II", ncols=90, ascii=False, file=progress_tee)

        while step < cfg.stage2_steps:
            idx = torch.randperm(len(tr_res))[:cfg.bs_2]
            dec_par.train(); opt2.zero_grad()
            logits, _ = dec_par(tr_ha[idx].to(device), tr_hb[idx].to(device), tr_ops[idx].to(device))
            loss = numerical_loss(logits, tr_res[idx].tolist(), cfg.n_digits)
            loss.backward(); nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt2.step(); sch2.step()
            loss_buf.append(loss.item()); step += 1; pbar.update(1)

            if step % cfg.log_every == 0:
                vm  = eval_s2(va_ha, va_hb, va_ops, va_res)
                tm  = eval_s2(te_ha, te_hb, te_ops, te_res)
                avg = float(np.mean(loss_buf[-cfg.log_every:]))
                entry = {"step": step, "loss": avg,
                         "acc1": vm["acc1"], "acc01": vm["acc01"], "test_acc1": tm["acc1"]}
                log_s2.append(entry); append_jsonl(s2_path, entry)
                log(train_log, f"  [II] step={step:5d} | loss={avg:.4f} | "
                               f"val ACC@1%={vm['acc1']:.3f} ACC@0.1%={vm['acc01']:.3f} | "
                               f"test ACC@1%={tm['acc1']:.3f}")
                if vm["acc1"] > best:
                    best = vm["acc1"]
                    torch.save({"net": decoder.state_dict(), "step": step},
                               os.path.join(cfg.out_dir, "stage2_best.pth"))
                plot_stage23_training(log_s2, [], cfg.out_dir)

        pbar.close()
        log(train_log, f"Stage II done.  Best val ACC@1% = {best:.3f}")
        cfg.stage2_ckpt = os.path.join(cfg.out_dir, "stage2_best.pth")

    else:
        if os.path.exists(s2_path):
            with open(s2_path) as f:
                log_s2 = [json.loads(l) for l in f if l.strip()]

    # ── Stage III ─────────────────────────────────────────────────────────────
    if cfg.stage in ("3", "23"):
        log(train_log, "\n" + "=" * 60)
        log(train_log, "STAGE III  —  End-to-End Fine-Tuning")
        log(train_log, "=" * 60)

        model = GeoNumLLM(cfg.llm_path, cfg.stage2_ckpt,
                          cfg.embed_dim, cfg.n_digits, cfg.llm_dim).to(device)

        frozen_dec = AlignmentDecoder(cfg.embed_dim, cfg.n_digits, cfg.llm_dim).to(device)
        if cfg.stage2_ckpt and os.path.exists(cfg.stage2_ckpt):
            frozen_dec.load_state_dict(
                torch.load(cfg.stage2_ckpt, map_location="cpu", weights_only=False)["net"])
        frozen_dec.eval()
        for p in frozen_dec.parameters(): p.requires_grad = False

        lora_p  = [p for n, p in model.named_parameters() if "lora_" in n]
        other_p = [p for n, p in model.named_parameters() if "lora_" not in n and p.requires_grad]
        log(train_log, f"  LoRA params: {sum(p.numel() for p in lora_p)/1e6:.2f}M  "
                       f"| other: {sum(p.numel() for p in other_p)/1e6:.2f}M")

        opt3 = torch.optim.AdamW([{"params": lora_p,  "lr": cfg.lr_3_lora},
                                   {"params": other_p, "lr": cfg.lr_3_other}], weight_decay=1e-4)
        sch3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt3, cfg.stage3_steps, eta_min=1e-5)

        @torch.no_grad()
        def eval_s3(ha, hb, ops, res, n=512):
            model.eval()
            idx   = torch.randperm(len(res))[:n]
            ha_b, hb_b, ops_b = ha[idx].to(device), hb[idx].to(device), ops[idx].to(device)
            _, h_dec = frozen_dec(ha_b, hb_b, ops_b)
            logits   = model(ha_b, hb_b, ops_b, h_dec_frozen=h_dec.detach())
            return evaluate_predictions(decode_logits(logits, cfg.n_digits).cpu().tolist(),
                                        res[idx].tolist())

        resume = 0
        if cfg.resume_ckpt and os.path.exists(cfg.resume_ckpt):
            saved = torch.load(cfg.resume_ckpt, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["net"]); resume = saved.get("step", 0)
            log(train_log, f"  Resumed from {cfg.resume_ckpt} at step {resume}")
        if resume > 0 and os.path.exists(s3_path):
            with open(s3_path) as f: log_s3 = [json.loads(l) for l in f if l.strip()]
        best = max((e["acc1"] for e in log_s3), default=0.0)

        step, accum, loss_buf = resume, 0, []
        pbar = tqdm(total=cfg.stage3_steps, desc="Stage III",
                    ncols=90, ascii=False, file=progress_tee, initial=resume)
        model.train(); opt3.zero_grad()

        # Initial eval before any fine-tuning (step=0), plotted at the II→III boundary.
        if resume == 0:
            vm = eval_s3(va_ha, va_hb, va_ops, va_res)
            tm = eval_s3(te_ha, te_hb, te_ops, te_res)
            w  = torch.softmax(torch.stack([model.w_llm_raw, model.w_dec_raw]), dim=0)
            entry = {"step": 0, "loss": float("nan"),
                     "acc1": vm["acc1"], "acc01": vm["acc01"],
                     "test_acc1": tm["acc1"],
                     "w_llm": float(w[0]), "w_dec": float(w[1])}
            log_s3.append(entry); append_jsonl(s3_path, entry)
            log(train_log, f"  [III] step=    0 | loss=  init | "
                           f"val ACC@1%={vm['acc1']:.3f} ACC@0.1%={vm['acc01']:.3f} | "
                           f"test ACC@1%={tm['acc1']:.3f}")
            plot_stage23_training(log_s2, log_s3, cfg.out_dir)
            model.train()

        while step < cfg.stage3_steps:
            idx   = torch.randperm(len(tr_res))[:cfg.bs_3]
            ha_b  = tr_ha[idx].to(device)
            hb_b  = tr_hb[idx].to(device)
            ops_b = tr_ops[idx].to(device)
            model.train()
            with torch.no_grad():
                _, h_dec = frozen_dec(ha_b, hb_b, ops_b)
            loss = numerical_loss(model(ha_b, hb_b, ops_b, h_dec.detach()),
                                  tr_res[idx].tolist(), cfg.n_digits) / cfg.grad_accum
            loss.backward(); accum += 1

            if accum == cfg.grad_accum:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt3.step(); sch3.step(); opt3.zero_grad()
                accum = 0; loss_buf.append(loss.item() * cfg.grad_accum)
                step += 1; pbar.update(1)

                if step % cfg.log_every == 0:
                    vm  = eval_s3(va_ha, va_hb, va_ops, va_res)
                    tm  = eval_s3(te_ha, te_hb, te_ops, te_res)
                    avg = float(np.mean(loss_buf[-cfg.log_every:]))
                    w   = torch.softmax(torch.stack([model.w_llm_raw, model.w_dec_raw]), dim=0)
                    w_llm, w_dec = float(w[0]), float(w[1])
                    entry = {"step": step, "loss": avg,
                             "acc1": vm["acc1"], "acc01": vm["acc01"],
                             "test_acc1": tm["acc1"], "w_llm": w_llm, "w_dec": w_dec}
                    log_s3.append(entry); append_jsonl(s3_path, entry)
                    log(train_log, f"  [III] step={step:5d} | loss={avg:.4f} | "
                                   f"val ACC@1%={vm['acc1']:.3f} ACC@0.1%={vm['acc01']:.3f} | "
                                   f"test ACC@1%={tm['acc1']:.3f} | "
                                   f"w_llm={w_llm:.3f} w_dec={w_dec:.3f}")
                    if vm["acc1"] > best:
                        best = vm["acc1"]
                        torch.save({"net": model.state_dict(), "step": step},
                                   os.path.join(cfg.out_dir, "stage3_best.pth"))
                    plot_stage23_training(log_s2, log_s3, cfg.out_dir)

        pbar.close()
        log(train_log, f"Stage III done.  Best val ACC@1% = {best:.3f}")

    plot_stage23_training(log_s2, log_s3, cfg.out_dir)
    train_log.close(); progress_tee.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",        default="23", choices=["2", "3", "23"])
    p.add_argument("--llm_path",     default="/path/to/Llama-3.2-1B-Instruct")
    p.add_argument("--encoder_ckpt", default="results/stage1/encoder.pth")
    p.add_argument("--stage2_ckpt",  default="")
    p.add_argument("--resume_ckpt",  default="")
    p.add_argument("--out_dir",      default="results/stage2")
    p.add_argument("--data_dir",     default="datasets/nupa")
    p.add_argument("--embed_dim",    type=int,   default=256)
    p.add_argument("--enc_n_digits", type=int,   default=6)
    p.add_argument("--n_digits",     type=int,   default=7)
    p.add_argument("--llm_dim",      type=int,   default=2048,
                   help="LLM hidden dim: 2048 (1B), 3072 (3B)")
    p.add_argument("--gpus",         type=int,   nargs="+", default=[0])
    p.add_argument("--stage2_steps", type=int,   default=4_000)
    p.add_argument("--stage3_steps", type=int,   default=8_000)
    p.add_argument("--bs_2",         type=int,   default=4096)
    p.add_argument("--bs_3",         type=int,   default=16)
    p.add_argument("--grad_accum",   type=int,   default=4)
    p.add_argument("--lr_2",         type=float, default=5e-4)
    p.add_argument("--lr_3_lora",    type=float, default=1e-4)
    p.add_argument("--lr_3_other",   type=float, default=3e-4)
    p.add_argument("--log_every",    type=int,   default=200)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
