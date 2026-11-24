#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=csc_8607
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:10:00
#SBATCH -J tp6-ex0
#SBATCH -o logs/%x-%j.out

set -euo pipefail
source tp6_env.sh
nvidia-smi -L
python -u check_env.py
