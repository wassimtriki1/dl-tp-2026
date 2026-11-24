#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=csc_8607
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:30:00
#SBATCH -J tp6-ex3
#SBATCH -o logs/%x-%j.out

set -euo pipefail
source tp6_env.sh
# 1) Calcul seul (données synthétiques) : l'effet de chaque levier sur le GPU
python -u bench.py --tag fp32-synthetic
python -u bench.py --tag tf32-synthetic --precision tf32
python -u bench.py --tag bf16-synthetic --precision bf16
python -u bench.py --tag bf16-cl-synthetic --precision bf16 --channels-last
python -u bench.py --tag bf16-cl-compile-synthetic --precision bf16 --channels-last --compile
# 2) Avec de vraies images : le chargement suit-il quand le GPU accélère ?
#    1 worker = un chargement volontairement limité (cf. exercice 2, question 5)
#    Avec 1 worker, la mesure dépend du CPU, partagé avec les voisins : 5 répétitions au lieu de 3
for D in raw small; do
    for W in 1 3; do
        R=$([ "$W" = 1 ] && echo 5 || echo 3)
        python -u bench.py --tag bf16-cl-$D-w$W --precision bf16 --channels-last --data $D \
               --workers $W --repeat $R
        python -u bench.py --tag bf16-cl-compile-$D-w$W --precision bf16 --channels-last --compile \
               --data $D --workers $W --repeat $R
    done
done
# 3) Plus grand batch qui tient en bf16
python -u bench.py --tag maxbatch-bf16 --precision bf16 --channels-last --find-max-batch
