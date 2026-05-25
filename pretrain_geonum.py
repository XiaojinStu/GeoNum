"""Stage I: self-supervised pretraining of GeoNumEncoder."""
import os, sys, json, argparse
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from geonum.encoder import GeoNumEncoder, compute_loss
from geonum.data    import ScalarDataset, evaluate_encoder
from geonum.trainer import log, append_jsonl
from geonum.viz     import plot_stage1_training, plot_scale_quality


def train(cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)

    def make_loader(split, shuffle, seed):
        ds = ScalarDataset(os.path.join(cfg.data_dir, f"{split}.json"),
                           scalar_lo=cfg.scalar_lo, scalar_hi=cfg.scalar_hi,
                           log_uniform=cfg.log_uniform, seed=seed)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=4, pin_memory=True), ds

    train_loader, train_ds = make_loader("train", True,  42)
    val_loader,   _        = make_loader("val",   False, 1000)
    test_loader,  _        = make_loader("test",  False, 2000)

    device = torch.device(f"cuda:{cfg.gpus[0]}")
    model  = GeoNumEncoder(embed_dim=cfg.embed_dim, n_digits=cfg.n_digits)
    if len(cfg.gpus) > 1:
        model = nn.DataParallel(model, device_ids=cfg.gpus)
    model = model.to(device)

    scalar_info = (f"±[{cfg.scalar_lo}, {cfg.scalar_hi}] "
                   f"({'log-uniform' if cfg.log_uniform else 'uniform'})"
                   if cfg.scalar_lo else "from JSON a/b fields (NUPA)")
    header = (f"Train {len(train_ds):,} | batch {cfg.batch_size} | "
              f"{len(train_loader)} steps/epoch | scalars {scalar_info}")
    print(header, flush=True)

    with open(os.path.join(cfg.out_dir, "train.log"), "w") as train_log, \
         open(os.path.join(cfg.out_dir, "progress.log"), "w") as prog_log:

        log(train_log, header)
        log(train_log, f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.epochs)

        step_log, epoch_evals, global_step = [], [], 0
        loss_buf = {k: 0.0 for k in ["total", "sign", "digit", "frac", "mag", "count"]}

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch:>2}/{cfg.epochs}",
                        ncols=110, ascii=False, file=prog_log)

            for batch in pbar:
                batch = batch.to(device)
                optimizer.zero_grad()
                losses = compute_loss(model(batch))
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1

                for k in ["total", "sign", "digit", "frac", "mag"]:
                    loss_buf[k] += losses[k].item()
                loss_buf["count"] += 1

                if global_step % cfg.log_every == 0:
                    n = loss_buf["count"]
                    entry = {k: loss_buf[k] / n for k in ["total", "sign", "digit", "frac", "mag"]}
                    entry["step"] = global_step
                    step_log.append(entry)
                    loss_buf = {k: 0.0 for k in loss_buf}
                    append_jsonl(os.path.join(cfg.out_dir, "log.jsonl"), entry)

                if global_step % 10 == 0:
                    pbar.set_postfix(loss=f"{losses['total'].item():.3f}",
                                     frac=f"{losses['frac'].item():.5f}")

            # Epoch-end evaluation
            core = model.module if hasattr(model, "module") else model
            val_r  = evaluate_encoder(core, val_loader,  device)
            test_r = evaluate_encoder(core, test_loader, device)
            val_loss = sum(compute_loss(model(b.to(device)))["total"].item()
                           for b in val_loader) / len(val_loader)
            if step_log:
                step_log[-1]["val_loss"] = val_loss

            ev = {**val_r, "epoch": epoch, "step": global_step, "val_loss": val_loss,
                  "test_acc5": test_r["acc5"], "test_acc1": test_r["acc1"],
                  "test_acc01": test_r["acc01"], "test_mae": test_r["mae"]}
            epoch_evals.append(ev)
            scheduler.step()

            line = (f"  ep {epoch:>2d} | loss={val_loss:.4f} | "
                    f"val  ACC@5%={val_r['acc5']:.4f}  ACC@1%={val_r['acc1']:.4f}  "
                    f"ACC@0.1%={val_r['acc01']:.4f} | "
                    f"test ACC@1%={test_r['acc1']:.4f}  ACC@0.1%={test_r['acc01']:.4f}")
            log(train_log, line)
            append_jsonl(os.path.join(cfg.out_dir, "log.jsonl"), {
                "epoch": epoch, "step": global_step,
                "val_loss": val_loss, **{f"val_{k}": val_r[k] for k in ["acc5","acc1","acc01","mae"]},
                **{f"test_{k}": test_r[k] for k in ["acc5","acc1","acc01","mae"]},
            })

            plot_stage1_training(step_log, epoch_evals, cfg.out_dir)
            plot_scale_quality(epoch_evals, cfg.out_dir, cfg.epochs)

        ckpt = os.path.join(cfg.out_dir, "encoder.pth")
        torch.save(core.state_dict(), ckpt)
        log(train_log, f"\nEncoder saved → {ckpt}")

        with open(os.path.join(cfg.out_dir, "metrics.json"), "w") as f:
            json.dump([{k: v for k, v in ev.items() if k not in ("true", "pred", "rel")}
                       for ev in epoch_evals], f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default="datasets/nupa")
    p.add_argument("--out_dir",     default="results/stage1")
    p.add_argument("--embed_dim",   type=int,   default=256)
    p.add_argument("--n_digits",    type=int,   default=6)
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=1024)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--log_every",   type=int,   default=50)
    p.add_argument("--gpus",        type=int,   nargs="+", default=[0])
    p.add_argument("--scalar_lo",   type=float, default=None)
    p.add_argument("--scalar_hi",   type=float, default=None)
    p.add_argument("--log_uniform", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
