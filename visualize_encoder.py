"""Visualize the geometry of a trained GeoNumEncoder checkpoint.

Produces:
    embedding_structure.png  — t-SNE scatter colored by value / magnitude / leading digit
    decimal_structure.png    — t-SNE + PCA circularity of fractional-part embeddings
"""
import os, sys, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.patches import Circle

from geonum.encoder import GeoNumEncoder

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 22,
    "axes.labelsize": 26, "axes.titlesize": 28,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
})


def _embeddings(encoder, values, device):
    with torch.no_grad():
        return encoder.encode(
            torch.tensor(values, dtype=torch.float32, device=device)).cpu().numpy()


def _tsne(emb, perplexity=50, n_iter=2000, seed=42):
    return TSNE(n_components=2, perplexity=perplexity,
                max_iter=n_iter, random_state=seed).fit_transform(emb)


def plot_embedding_structure(encoder, device, out_dir, n_points=800):
    print("  Computing embedding structure ...", flush=True)
    rng    = np.random.default_rng(42)
    values = np.exp(rng.uniform(np.log(100.0), np.log(1e5), n_points)).astype(np.float32)
    emb    = _embeddings(encoder, values, device)
    coords = _tsne(emb)

    int_vals    = np.floor(np.abs(values)).astype(int)
    log_mags    = np.log10(np.abs(values) + 1)
    lead_digits = (int_vals // (10 ** np.floor(
        np.log10(np.maximum(int_vals, 1))).astype(int))) % 10

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, (cvals, cmap, clabel, title) in zip(axes, [
        (values,      "plasma",  "Scalar Value",  "Value"),
        (log_mags,    "viridis", "Log₁₀(|x|+1)", "Log Magnitude"),
        (lead_digits, "tab10",   "Leading Digit", "Leading Digit"),
    ]):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=cvals, cmap=cmap,
                        s=120, alpha=0.85, edgecolors="white", linewidth=0.8)
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(clabel, fontsize=24, fontweight="bold"); cb.ax.tick_params(labelsize=18)
        ax.set_title(title, fontsize=28, fontweight="bold", pad=12)
        ax.tick_params(labelsize=18); ax.grid(alpha=0.25, linewidth=0.8)

    plt.tight_layout(pad=1.5)
    path = os.path.join(out_dir, "embedding_structure.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"    Saved → {path}", flush=True)


def _circularity(pts, decimals):
    if len(pts) < 8:
        return 0.0, np.zeros(2), 1.0
    try:
        x, y   = pts[:, 0], pts[:, 1]
        A      = np.column_stack([x, y, np.ones(len(x))])
        params = np.linalg.lstsq(A, x**2 + y**2, rcond=None)[0]
        center = params[:2] / 2
        radius = np.sqrt(params[2] + center[0]**2 + center[1]**2)
        dists  = np.linalg.norm(pts - center, axis=1)
        cv     = np.std(dists) / (np.mean(dists) + 1e-8)
        angles = (np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]) + 2*np.pi) % (2*np.pi)
        corr   = abs(np.corrcoef(angles, decimals * 2*np.pi)[0, 1])
        score  = float(np.clip((1/(1+cv))*0.6 + (0 if np.isnan(corr) else corr)*0.4, 0, 1))
        return score, center, radius
    except Exception:
        return 0.0, np.zeros(2), 1.0


def plot_decimal_structure(encoder, device, out_dir, n_points=600):
    print("  Computing decimal structure ...", flush=True)
    rng  = np.random.default_rng(0)
    main = rng.uniform(0.01, 100.0, int(n_points * 0.7)).astype(np.float32)
    bnd  = np.concatenate([rng.uniform(b-0.5, b+0.5, int(n_points*0.3/4))
                           for b in [1, 10, 100, 1000]]).astype(np.float32)
    vals = np.concatenate([main, bnd])[:n_points]
    rng.shuffle(vals)
    decs = vals - np.floor(vals)

    emb    = _embeddings(encoder, vals, device)
    coords = _tsne(emb, perplexity=30, n_iter=1500)

    mask   = (decs > 0.05) & (decs < 0.95)
    if mask.sum() < 20:
        print("    Not enough fractional samples — skipping.", flush=True); return

    pca_pts          = PCA(n_components=2).fit_transform(emb[mask])
    circ, center, r  = _circularity(pca_pts, decs[mask])

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, data, crds, title in [
        (axes[0], decs,      coords,  "t-SNE  (by Fractional Part)"),
        (axes[1], decs[mask], pca_pts, f"PCA  (Circularity = {circ:.3f}){'  ✓' if circ > 0.15 else ''}"),
    ]:
        sc = ax.scatter(crds[:, 0], crds[:, 1], c=data, cmap="Greens",
                        s=130, alpha=0.85, edgecolors="white", linewidth=0.8)
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Decimal", fontsize=24, fontweight="bold"); cb.ax.tick_params(labelsize=18)
        ax.set_title(title, fontsize=28, fontweight="bold", pad=12)
        ax.tick_params(labelsize=18); ax.grid(alpha=0.2, linewidth=0.8)

    if circ > 0.15:
        axes[1].add_patch(Circle(center, r, fill=False, color="#DC2626",
                                 linewidth=2.0, linestyle="--", alpha=0.9))
        axes[1].scatter(*center, c="#DC2626", s=80, marker="x", linewidths=2.5, zorder=5)
    axes[1].set_aspect("equal", adjustable="box")

    plt.tight_layout(pad=1.5, w_pad=2.0)
    path = os.path.join(out_dir, "decimal_structure.png")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"    Saved → {path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_ckpt", required=True)
    p.add_argument("--out_dir",   default="figures/encoder")
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--n_digits",  type=int, default=6)
    p.add_argument("--n_points",  type=int, default=800)
    p.add_argument("--gpu",       type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    cfg    = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    encoder = GeoNumEncoder(embed_dim=cfg.embed_dim, n_digits=cfg.n_digits).to(device)
    encoder.load_state_dict(torch.load(cfg.encoder_ckpt, map_location="cpu", weights_only=False))
    encoder.eval()
    print(f"Loaded encoder from {cfg.encoder_ckpt}", flush=True)
    plot_embedding_structure(encoder, device, cfg.out_dir, cfg.n_points)
    plot_decimal_structure(encoder, device, cfg.out_dir, min(cfg.n_points, 600))
    print(f"\nAll figures saved to {cfg.out_dir}/", flush=True)
