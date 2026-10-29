"""Exporte toutes les courbes TensorBoard de runs_rnn/ dans un seul fichier CSV
(sans matplotlib, utilisable directement sur le cluster).

Usage : python export_tb_csv.py            -> écrit tb_scalars.csv
"""
import csv
import glob
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

rows = []
for run_dir in sorted(glob.glob("runs_rnn/*")):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    for tag in ea.Tags()["scalars"]:
        for e in ea.Scalars(tag):
            rows.append([os.path.basename(run_dir), tag, e.step, f"{e.value:.6f}"])

with open("tb_scalars.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run", "tag", "step", "value"])
    w.writerows(rows)

print(f"{len(rows)} valeurs exportées dans tb_scalars.csv")
for run in sorted({r[0] for r in rows}):
    print(" -", run)
