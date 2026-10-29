#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -J music-lstm
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# Usage : sbatch run_music.sh            -> Exercice 4.b (LSTM h=1024, 50 000 partitions, 30 époques + génération)
#         sbatch run_music.sh explore    -> Exercices 1 et 2 (statistiques, vocabulaire, paramètres)
#         sbatch run_music.sh quick      -> Exercice 3 (Run 1, 2, 3)
#         sbatch run_music.sh generate   -> Exercice 4.a (génération avec le LSTM du Run 3)

set -euo pipefail
mkdir -p logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

MODE=${1:-full}
echo "Job $SLURM_JOB_ID sur $SLURM_NODELIST, mode=$MODE"
nvidia-smi

python -u train_music.py --mode "$MODE"
