"""TP6 : affiche les fichiers results_*.csv sous forme de tableaux Markdown (pour le rapport)."""
import csv
import glob

for path in sorted(glob.glob("results_*.csv")):
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        continue
    print(f"\n### {path}\n")
    print("| " + " | ".join(rows[0]) + " |")
    print("|" + "---|" * len(rows[0]))
    for r in rows[1:]:
        print("| " + " | ".join(r) + " |")
