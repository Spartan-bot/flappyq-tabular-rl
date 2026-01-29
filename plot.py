"""
plot.py

Plot learning curve (frames/pipes) and visit heatmap (x vs y).
"""

from __future__ import annotations
import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from env_simple import X_BINS, Y_BINS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Run directory under results/")
    args = ap.parse_args()

    run_dir = args.dir
    scores_path = os.path.join(run_dir, "scores.csv")
    visit_path  = os.path.join(run_dir, "visit.npy")

    episodes, frames, pipes = [], [], []
    with open(scores_path, "r") as f:
        next(f)
        for line in f:
            ep_s, fr_s, pi_s = line.strip().split(",")
            episodes.append(int(ep_s))
            frames.append(int(fr_s))
            pipes.append(int(pi_s))

    episodes = np.array(episodes)
    frames   = np.array(frames, dtype=float)
    pipes    = np.array(pipes, dtype=float)

    def moving_avg(x, w=100):
        if len(x) < w: return x
        kernel = np.ones(w)/w
        return np.convolve(x, kernel, mode="valid")

    plt.figure()
    plt.plot(episodes, frames, alpha=0.2, label="frames (raw)")
    if len(frames) >= 100:
        plt.plot(episodes[99:], moving_avg(frames), label="frames (100-ep mean)")
        plt.plot(episodes[99:], moving_avg(pipes),  label="pipes (100-ep mean)")
    plt.xlabel("Episode"); plt.ylabel("Frames / Pipes")
    plt.title("Learning curve"); plt.legend(); plt.tight_layout()
    out_curve = os.path.join(run_dir, "learning_curve.png")
    plt.savefig(out_curve); print("✓", out_curve)

    visit = np.load(visit_path)   # (nx, ny, nz, 2)
    vis2d = visit.sum(axis=(2,3))

    plt.figure()
    plt.imshow(vis2d.T, origin="lower", aspect="auto")
    plt.colorbar(label="visit count")
    plt.xticks(
        ticks=np.linspace(0, len(X_BINS)-1, 6),
        labels=np.linspace(X_BINS[0], X_BINS[-1], 6, dtype=int),
    )
    plt.yticks(
        ticks=np.linspace(0, len(Y_BINS)-1, 5),
        labels=np.linspace(Y_BINS[0], Y_BINS[-1], 5, dtype=int),
    )
    plt.xlabel("x-bin (distance)"); plt.ylabel("y-bin (gap offset)")
    plt.title("State visit heatmap (x vs y)")
    plt.tight_layout()
    out_heat = os.path.join(run_dir, "visit_heatmap.png")
    plt.savefig(out_heat); print("✓", out_heat)

if __name__ == "__main__":
    main()
