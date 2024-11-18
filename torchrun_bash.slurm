#!/bin/bash
#SBATCH --job-name=train_transformer    # Job name
#SBATCH --partition=gpu                 # Partition name
#SBATCH --account=bb1438                # Account name
#SBATCH --nodes=2                       # Number of nodes
#SBATCH --ntasks-per-node=4             # Number of tasks per node (GPUs per node)
#SBATCH --gres=gpu:4                    # Number of GPUs per node
#SBATCH --cpus-per-task=32              # CPU cores per task
#SBATCH --mem=512G                      # Memory per node
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec
#SBATCH --output=logs/%x-%j.out         # Standard output and error log

module load pytorch


# Set OMP_NUM_THREADS to match the number of CPU cores per task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Get the master address from SLURM
MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR

# Set a master port (choose an open port)
MASTER_PORT=12345
export MASTER_PORT

# Set node rank using SLURM_NODEID
NODE_RANK=$SLURM_NODEID
export NODE_RANK

# (Optional) Set NCCL environment variables if you know your hardware supports it
# If unsure, it's safer to omit these settings
# export NCCL_DEBUG=INFO
# export NCCL_P2P_LEVEL=NVL
# export NCCL_SOCKET_IFNAME=^lo,docker0
# export NCCL_IB_GDR=0

# Run the training script using torchrun
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --node_rank=$NODE_RANK \
    main_dino.py \
    --arch vit_small \