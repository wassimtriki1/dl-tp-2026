#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=csc_8607
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:20:00
#SBATCH -J tp6-train
#SBATCH -o logs/%x-%j.out

set -euo pipefail
source tp6_env.sh
# Adaptez les options à VOTRE meilleure configuration de l'exercice 3
python -u train.py --tag best --precision bf16 --channels-last --epochs 2
