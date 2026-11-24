#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=csc_8607
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=00:25:00
#SBATCH -J tp6-ex1
#SBATCH -o logs/%x-%j.out

set -euo pipefail
source tp6_env.sh
python -u bench.py --tag ref-synth                       # fp32, batch 64, données synthétiques
python -u bench.py --tag ref-async --async-demo          # un pas, avec et sans synchronisation
python -u bench.py --tag ref-maxbatch --find-max-batch   # plus grand batch qui tient en fp32
python -u bench.py --tag ref-raw --data raw --workers 3 --epoch   # vraies images, configuration « CI 3 »
