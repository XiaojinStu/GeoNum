"""Plot Stage II / III training curves from saved log files."""
import os, json, argparse
from geonum.viz import plot_stage23_training


def load_logs(out_dir):
    log2, log3 = [], []
    for stems, target in [(["log_stage2", "log_A"], log2),
                          (["log_stage3", "log_B"], log3)]:
        for stem in stems:
            for ext in (".jsonl", ".json"):
                path = os.path.join(out_dir, stem + ext)
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    if ext == ".jsonl":
                        target.extend(json.loads(l) for l in f if l.strip())
                    else:
                        target.extend(json.load(f))
                break
            if target:
                break
    return log2, log3


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", required=True)
    p.add_argument("--save",    default="training.png")
    cfg = p.parse_args()
    log2, log3 = load_logs(cfg.out_dir)
    if not log2 and not log3:
        print("No logs found in", cfg.out_dir)
    else:
        plot_stage23_training(log2, log3, cfg.out_dir)
