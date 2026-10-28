#!/bin/bash
#SBATCH --account=csc_8607
#SBATCH --qos=normal
# Entraînements rapides de l'Exercice 3 (20 000 images, 10 époques) sur un nœud GPU.
# Usage : sbatch run_quick.sh quick   (MLP puis AlexNet)
#         sbatch run_quick.sh deep    (AlexNet + 3 convolutions)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH -J tp3-quick
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail
mkdir -p logs

source ~/miniforge3/etc/profile.d/conda.sh
conda activate deeplearning

MODE=${1:-quick}
echo "Job $SLURM_JOB_ID sur $SLURM_NODELIST, mode=$MODE"
nvidia-smi

python -u train_imagenet.py --mode "$MODE"
