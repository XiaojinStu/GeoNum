"""
Visualization utilities for GeoNum training curves and encoder geometry.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         20,
    "axes.titlesize":    22,
    "axes.labelsize":    20,
    "xtick.labelsize":   18,
    "ytick.labelsize":   18,
    "legend.fontsize":   17,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

# Color palette
_C = dict(
    sign="  #7C3AED", digit="#D97706", frac="#059669", mag="#0891B2",
    total="#111827", val="#DC2626",  train="#2563EB",
    stage2="#3B82F6", stage3="#22C55E",
)
_BLUE, _RED = "#2563EB", "#DC2626"

MAGNITUDE_RANGES = [
    ("<0.1",    0,       0.1),
    ("0.1–1",   0.1,     1.0),
    ("1–10",    1.0,    10.0),
    ("10–100", 10.0,   100.0),
    ("100–1K", 100,  1_000),
    ("1K–10K", 1_000, 10_000),
    ("10K–100K", 10_000, 100_000),
]


def _smooth(values, window=7):
    if len(values) < window:
        return np.array(values, dtype=float)
    arr    = np.array(values, dtype=float)
    padded = np.pad(arr, window // 2, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")[:len(arr)]


# ── Stage I training curves ───────────────────────────────────────────────────

def plot_stage1_training(step_log: list, epoch_evals: list, out_dir: str):
    """Save training.png — overwritten each epoch."""
    if not step_log:
        return

    steps      = [e["step"] for e in step_log]
    val_points = [(e["step"], e["val_loss"]) for e in step_log if "val_loss" in e]
    val_steps, val_losses = zip(*val_points) if val_points else ([], [])

    num_rows = 4 if epoch_evals else 2
    fig = plt.figure(figsize=(16, 5.5 * num_rows))
    gs  = gridspec.GridSpec(num_rows, 2, figure=fig,
                            hspace=0.56, wspace=0.32,
                            top=0.95, bottom=0.06, left=0.08, right=0.97)
    fig.suptitle("GeoNum Stage I Pretraining", fontsize=26, fontweight="bold", y=0.99)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(steps, _smooth([e["total"] for e in step_log]),
            color=_C["train"], lw=2.5, label="Train loss (smoothed)")
    if val_steps:
        ax.scatter(val_steps, val_losses, color=_C["val"], s=80, zorder=5,
                   label="Val loss")
        ax.plot(val_steps, val_losses, color=_C["val"], lw=2.0, ls="--", alpha=0.8)
    ax.set_yscale("log"); ax.set_ylabel("Total Loss (log)")
    ax.set_title("Total Loss"); ax.legend(); ax.grid(True, alpha=0.25)

    ax = fig.add_subplot(gs[1, :])
    for key, label, color in [
        ("sign",  "Sign  (α·Ls)",     _C["sign"].strip()),
        ("digit", "Digit (β·ΣωᵢLᵢ)", _C["digit"]),
        ("frac",  "Frac  (γ·Lf)",     _C["frac"]),
        ("mag",   "Mag   (δ·Lm)",     _C["mag"]),
    ]:
        ax.plot(steps, _smooth([e[key] for e in step_log]),
                color=color, lw=2.5, label=label)
    ax.set_yscale("log"); ax.set_ylabel("Loss (log)")
    ax.set_title("Loss Components (Eq. 10)"); ax.legend(ncol=2); ax.grid(True, alpha=0.25)

    if not epoch_evals:
        fig.savefig(os.path.join(out_dir, "training.png"), bbox_inches="tight")
        plt.close(fig); return

    eval_steps = [ev["step"] for ev in epoch_evals]

    ax = fig.add_subplot(gs[2, :])
    if "test_acc5" in epoch_evals[0]:
        ax.plot(eval_steps, [ev["test_acc5"]  * 100 for ev in epoch_evals],
                color="#93C5FD", lw=2.5, marker="D", ms=6, label="Test ACC@5%")
        ax.plot(eval_steps, [ev["test_acc1"]  * 100 for ev in epoch_evals],
                color="#2563EB", lw=2.5, marker="o", ms=6, label="Test ACC@1%")
        ax.plot(eval_steps, [ev["test_acc01"] * 100 for ev in epoch_evals],
                color="#1E3A8A", lw=2.5, marker="^", ms=6, label="Test ACC@0.1%")
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Reconstruction Accuracy")
    ax.set_xlabel("Global Step"); ax.legend(ncol=2); ax.grid(True, alpha=0.25)

    num_epochs  = len(epoch_evals)
    cdf_step    = max(1, num_epochs // 10)
    cdf_indices = set(range(0, num_epochs, cdf_step)) | {num_epochs - 1}
    cdf_evals   = [ev for i, ev in enumerate(epoch_evals) if i in cdf_indices]
    color_map   = plt.cm.Blues(np.linspace(0.25, 1.0, max(len(cdf_evals), 1)))
    final_rel   = np.sort(epoch_evals[-1]["rel"])
    x_full = min(float(np.percentile(final_rel, 99.5)), 0.05)
    x_zoom = max(min(float(np.percentile(final_rel, 95)), 0.005), 0.0005)

    for col_idx, (x_limit, title) in enumerate([
        (x_full, f"Relative Error CDF (0–{x_full*100:.3g}%)"),
        (x_zoom, f"Relative Error CDF (0–{x_zoom*100:.3g}%) zoomed"),
    ]):
        ax = fig.add_subplot(gs[3, col_idx])
        for i, ev in enumerate(cdf_evals):
            is_final = (ev["epoch"] == epoch_evals[-1]["epoch"])
            rel  = np.sort(ev["rel"])
            cdf  = np.arange(1, len(rel) + 1) / len(rel)
            mask = rel <= x_limit
            if mask.sum() > 0:
                ax.plot(rel[mask], cdf[mask], color=color_map[i],
                        lw=3.0 if is_final else 1.2,
                        label=f"ep {ev['epoch']}" + (" (final)" if is_final else ""))
        for thr, col, ls, lbl in [(0.01, "#F59E0B", "--", "1%"), (0.001, "#EF4444", ":", "0.1%")]:
            if thr <= x_limit:
                ax.axvline(thr, color=col, ls=ls, lw=2.0, label=f"{lbl} error")
        ax.set_xlim(0, x_limit); ax.set_ylim(0, 1.01)
        ax.set_ylabel("Cumulative Fraction"); ax.set_title(title)
        ax.legend(ncol=2, fontsize=14); ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.4g}%"))
        ax.set_xlabel("Relative Error")
        for tl in ax.get_xticklabels():
            tl.set_rotation(30); tl.set_ha("right")

    fig.savefig(os.path.join(out_dir, "training.png"), bbox_inches="tight")
    plt.close(fig)


def plot_scale_quality(epoch_evals: list, out_dir: str, total_epochs: int = 30):
    """Save scale_quality.png — embedding evolution across training epochs."""
    if not epoch_evals:
        return

    n_cols   = 10
    step     = max(1, total_epochs // n_cols)
    selected = [ev for ev in epoch_evals if ev["epoch"] % step == 0]
    last     = epoch_evals[-1]
    if not selected or selected[-1]["epoch"] != last["epoch"]:
        selected.append(last)
    selected = selected[-n_cols:]
    if not selected:
        return

    final_true   = last["true"]
    active_ranges = [(lbl, lo, hi) for lbl, lo, hi in MAGNITUDE_RANGES
                     if np.sum((np.abs(final_true) >= lo) & (np.abs(final_true) < hi)) > 0]
    if not active_ranges:
        return

    n_ranges    = len(active_ranges)
    row_heights = [3.0, 2.5] * n_ranges + [3.0]
    fig, axes   = plt.subplots(n_ranges * 2 + 1, len(selected),
                               figsize=(3.6 * len(selected), sum(row_heights)),
                               gridspec_kw={"height_ratios": row_heights},
                               squeeze=False)
    fig.suptitle("GeoNum Embedding Evolution", fontsize=26, fontweight="bold", y=1.005)

    rng = np.random.default_rng(42)
    range_scales = {}
    for lbl, lo, hi in active_ranges:
        mask = (np.abs(selected[0]["true"]) >= lo) & (np.abs(selected[0]["true"]) < hi)
        if mask.sum() > 0:
            res = selected[0]["pred"][mask] - selected[0]["true"][mask]
            range_scales[lbl] = max(np.percentile(np.abs(res), 95), 1e-3)
        else:
            range_scales[lbl] = 1.0

    for r_idx, (lbl, lo, hi) in enumerate(active_ranges):
        y_scale = range_scales[lbl]
        for col, ev in enumerate(selected):
            mask    = (np.abs(ev["true"]) >= lo) & (np.abs(ev["true"]) < hi)
            t, p    = ev["true"][mask], ev["pred"][mask]
            n_show  = min(400, len(t))
            if n_show == 0:
                for row in [r_idx * 2, r_idx * 2 + 1]:
                    axes[row][col].text(0.5, 0.5, "no data", ha="center", va="center",
                                       transform=axes[row][col].transAxes,
                                       fontsize=11, color="#9CA3AF")
                    axes[row][col].set_xticks([]); axes[row][col].set_yticks([])
                continue
            idx       = rng.choice(len(t), n_show, replace=False)
            ts, ps    = t[idx], p[idx]
            res       = ps - ts
            rel_err   = np.abs(res) / (np.abs(ts) + 1e-8)
            acc1      = float(np.mean(rel_err < 0.01))
            log_x     = np.sign(ts) * np.log10(np.abs(ts) + 1e-9)

            ax_d = axes[r_idx * 2][col]
            pos  = ts >= 0
            if pos.sum()  > 0: ax_d.scatter(ts[pos],  ps[pos],  color=_BLUE, s=16, alpha=0.72, edgecolors="white", linewidths=0.3, rasterized=True)
            if (~pos).sum()>0: ax_d.scatter(ts[~pos], ps[~pos], color=_RED,  s=16, alpha=0.72, edgecolors="white", linewidths=0.3, rasterized=True)
            v0, v1 = float(ts.min()), float(ts.max()); pad = (v1 - v0) * 0.05
            ax_d.plot([v0-pad, v1+pad], [v0-pad, v1+pad], color="#374151", lw=1.0, ls="--", alpha=0.65)
            ax_d.set_title(f"ep {ev['epoch']}  {acc1*100:.1f}%", fontsize=12, pad=3)
            if col == 0: ax_d.set_ylabel(f"|x|∈[{lbl}]\npred", fontsize=12)
            ax_d.tick_params(labelsize=10); ax_d.grid(True, alpha=0.18)

            ax_r = axes[r_idx * 2 + 1][col]
            sc   = ax_r.scatter(log_x, res, c=np.log10(rel_err + 1e-9),
                                cmap="RdYlGn_r", vmin=np.log10(1e-4), vmax=np.log10(0.1),
                                s=10, alpha=0.75, rasterized=True)
            ax_r.axhline(0, color="#374151", lw=1.0, alpha=0.55, ls="--")
            ax_r.set_ylim(-y_scale * 1.15, y_scale * 1.15)
            if col == 0: ax_r.set_ylabel("residual", fontsize=12)
            if r_idx == n_ranges - 1: ax_r.set_xlabel("sign·log₁₀|x|", fontsize=12)
            ax_r.tick_params(labelsize=10); ax_r.grid(True, alpha=0.18)
            if col == len(selected) - 1:
                cbar = plt.colorbar(sc, ax=ax_r, fraction=0.07, pad=0.03)
                cbar.set_label("log₁₀(rel err)", fontsize=10)
                cbar.ax.tick_params(labelsize=9)

    dec_edges  = np.arange(0, 1.05, 0.1)
    bin_centers = [(a + b) / 2 for a, b in zip(dec_edges[:-1], dec_edges[1:])]
    bot_row = n_ranges * 2
    for col, ev in enumerate(selected):
        ax      = axes[bot_row][col]
        t, p    = ev["true"], ev["pred"]
        rel_err = np.abs((p - t) / (np.abs(t) + 1e-8))
        frac    = np.abs(t) - np.floor(np.abs(t))
        a1, a01 = [], []
        for lo_d, hi_d in zip(dec_edges[:-1], dec_edges[1:]):
            mask = (frac >= lo_d) & (frac < hi_d)
            a1.append(float(np.mean(rel_err[mask] < 0.01)  * 100) if mask.sum() > 0 else np.nan)
            a01.append(float(np.mean(rel_err[mask] < 0.001) * 100) if mask.sum() > 0 else np.nan)
        x = np.arange(len(bin_centers)); w = 0.38
        ax.bar(x - w/2, a1,  w, color=_BLUE,    alpha=0.82, label="ACC@1%")
        ax.bar(x + w/2, a01, w, color="#1E3A8A", alpha=0.82, label="ACC@0.1%")
        ax.set_ylim(0, 108)
        ax.set_xticks(x); ax.set_xticklabels([f"{c:.1f}" for c in bin_centers],
                                              rotation=45, ha="right", fontsize=10)
        ax.set_xlabel("Fractional part", fontsize=12)
        if col == 0: ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(f"ep {ev['epoch']}  decimal", fontsize=12, pad=3)
        ax.tick_params(labelsize=10); ax.grid(True, axis="y", alpha=0.22)
        if col == len(selected) - 1: ax.legend(fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(os.path.join(out_dir, "scale_quality.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Stage II/III training curves ─────────────────────────────────────────────

def plot_stage23_training(log_s2: list, log_s3: list, out_dir: str):
    """Save training.png — 3-panel horizontal: Loss | ACC@1% | ACC@0.1%."""
    if not log_s2 and not log_s3:
        return

    has_s2  = bool(log_s2)
    has_s3  = bool(log_s3)
    s2_end  = log_s2[-1]["step"] if has_s2 else 0
    steps_2 = [e["step"] for e in log_s2]
    steps_3 = [e["step"] + s2_end for e in log_s3]

    if has_s2 and has_s3:
        title = "GeoNum Stage II and Stage III"
    elif has_s2:
        title = "GeoNum Stage II Projection Alignment"
    else:
        title = "GeoNum Stage III End-to-End Fine-Tuning"

    boundary = s2_end if has_s2 and has_s3 else None
    mk2 = dict(marker="o", ms=5, lw=2.0)
    mk3 = dict(marker="s", ms=5, lw=2.0)

    def _vline(ax):
        if boundary:
            ax.axvline(boundary, color="gray", ls="--", lw=1.4, alpha=0.7, label="II→III")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.03)

    for ax, key, ylabel in zip(
        axes,
        ["loss", "acc1", "acc01"],
        ["Loss", "ACC@1% (%)", "ACC@0.1% (%)"],
    ):
        if has_s2:
            vals = _smooth([e["loss"] for e in log_s2]) if key == "loss" \
                   else [e[key] * 100 for e in log_s2]
            ax.plot(steps_2, vals, color=_C["stage2"], label="Stage II", **mk2)
        if has_s3:
            if key == "loss":
                # skip nan entries (e.g. step=0 initial eval has no loss yet)
                pairs = [(s, v) for s, e in zip(steps_3, log_s3)
                         for v in [e["loss"]] if v == v]  # nan != nan
                if pairs:
                    sx, sy = zip(*pairs)
                    ax.plot(sx, _smooth(list(sy)), color=_C["stage3"], label="Stage III", **mk3)
            else:
                vals = [e[key] * 100 for e in log_s3]
                ax.plot(steps_3, vals, color=_C["stage3"], label="Stage III", **mk3)
        _vline(ax)
        ax.set_xlabel("Global Step"); ax.set_ylabel(ylabel); ax.set_title(ylabel)
        ax.legend(); ax.grid(True, alpha=0.25)

    fig.tight_layout()
    path = os.path.join(out_dir, "training.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path}", flush=True)
