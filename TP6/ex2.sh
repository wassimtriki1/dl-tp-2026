#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=csc_8607
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:25:00
#SBATCH -J tp6-ex2
#SBATCH -o logs/%x-%j.out

set -euo pipefail
source tp6_env.sh
for W in 0 1 2 3; do
    python -u bench.py --tag raw-w$W --data raw --workers $W --steps 20 --repeat 2
done
python -u bench.py --tag raw-w3-nopin --data raw --workers 3 --no-pin --steps 20 --repeat 2
for W in 0 3; do
    python -u bench.py --tag small-w$W --data small --workers $W --steps 20 --repeat 2
done
python -u profile_run.py --tag raw-w3 --data raw --workers 3
python -u profile_run.py --tag small-w3 --data small --workers 3
