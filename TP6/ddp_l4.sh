#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --account=csc_8607
#SBATCH --nodes=2                    # deux machines…
#SBATCH --ntasks-per-node=1          # … un lanceur torchrun par machine
#SBATCH --gres=gpu:nvidia_l4:1       # … une L4 entière par machine (pas de MIG)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH -J tp6-ddp
#SBATCH -o logs/%x-%j.out

set -euo pipefail
export TP6_NO_COPY=1                 # données synthétiques : pas besoin de copier les images
source tp6_env.sh

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))   # ex. starfighter-slurm-node-03-1 …
export MASTER_ADDR=${NODES[0]}       # le rendez-vous a lieu sur le premier nœud
PORT=$((20000 + SLURM_JOB_ID % 10000))
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK   # srun n'hérite pas de --cpus-per-task
export NCCL_DEBUG=WARN
# export NCCL_SOCKET_IFNAME=eth0     # seulement si NCCL choisit une mauvaise interface réseau
echo "Nœuds : ${NODES[*]} ; rendez-vous sur $MASTER_ADDR"

run2() {  # $1 = port du rendez-vous, puis le script et ses arguments : torchrun sur les 2 nœuds
    local port=$1; shift
    srun torchrun --nnodes=2 --nproc_per_node=1 --rdzv_backend=c10d \
         --rdzv_id="$SLURM_JOB_ID-$port" --rdzv_endpoint="$MASTER_ADDR:$port" "$@"
}

# A. Communication seule : correction et débit d'un all-reduce de 102 Mo entre les deux nœuds
export NCCL_DEBUG=INFO
if ! run2 "$PORT" check_dist.py > "logs/nccl-$SLURM_JOB_ID.log" 2>&1; then
    echo "ÉCHEC du test de communication, fin du log :"; tail -n 30 "logs/nccl-$SLURM_JOB_ID.log"; exit 1
fi
grep -E "rang|all-reduce|Using network|NET/" "logs/nccl-$SLURM_JOB_ID.log" | sort -u | head -n 20 || true
export NCCL_DEBUG=WARN

# B. Référence : un seul processus, sur le premier nœud
srun --nodes=1 --ntasks=1 --nodelist="${NODES[0]}" \
     torchrun --standalone --nproc_per_node=1 ddp_bench.py --tag l4-x1 --channels-last

# C. DDP sur les deux nœuds
run2 $((PORT + 1)) ddp_bench.py --tag l4-x2 --channels-last

# D. DDP sur les deux nœuds, gradients compressés en bf16 pendant l'all-reduce
run2 $((PORT + 2)) ddp_bench.py --tag l4-x2-bf16hook --channels-last --bf16-hook

# E. Référence avec accumulation : 8 micro-batchs par pas d'optimiseur, un seul processus
srun --nodes=1 --ntasks=1 --nodelist="${NODES[0]}" \
     torchrun --standalone --nproc_per_node=1 ddp_bench.py --tag l4-x1-accum8 --channels-last \
     --accum 8 --warmup 3 --steps 10

# F. DDP sur les deux nœuds : accumulation (une communication tous les 8 micro-batchs) + bf16
run2 $((PORT + 3)) ddp_bench.py --tag l4-x2-accum8-bf16hook --channels-last --bf16-hook \
     --accum 8 --warmup 3 --steps 10
