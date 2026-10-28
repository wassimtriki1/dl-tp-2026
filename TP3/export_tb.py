"""Exporte les courbes TensorBoard en PNG + tableau Markdown (pratique sur le cluster,
sans navigateur ni capture d'écran).

Usage :
    python export_tb.py runs/mlp_sub20000_lr0.0001 runs/alexnet_sub20000_lr0.0001 -o figures/ex3_mlp_vs_alexnet.png
    python export_tb.py runs/alexnet_subNone_lr0.0001 -o figures/ex4_full.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TAGS = ["Loss/Train", "Loss/Validation", "Accuracy/Train", "Accuracy/Validation"]


def load(run_dir):
    ea = EventAccumulator(run_dir)
    ea.Reload()
    available = ea.Tags()["scalars"]
    return {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in TAGS if t in available}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("-o", "--output", default="figures/courbes.png")
    args = p.parse_args()

    data = {os.path.basename(r.rstrip("/")): load(r) for r in args.runs}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, d in data.items():
        for tag, ax, style in [("Loss/Train", axes[0], "-"), ("Loss/Validation", axes[0], "--"),
                               ("Accuracy/Train", axes[1], "-"), ("Accuracy/Validation", axes[1], "--")]:
            if tag in d:
                steps, vals = zip(*d[tag])
                ax.plot([s + 1 for s in steps], vals, style, marker="o", ms=3,
                        label=f"{name} – {tag.split('/')[1]}")
    axes[0].set_title("Loss"); axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Époque"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(args.output, dpi=130)
    print(f"Figure enregistrée : {args.output}\n")

    for name, d in data.items():
        print(f"### {name}\n")
        print("| Époque | Train Loss | Val Loss | Train Acc | Val Acc |")
        print("|---|---|---|---|---|")
        n = max(len(v) for v in d.values())
        for i in range(n):
            row = [f"{d[t][i][1]:.4f}" if t in d and i < len(d[t]) else "-" for t in TAGS]
            print(f"| {i+1} | {' | '.join(row)} |")
        print()


if __name__ == "__main__":
    main()
