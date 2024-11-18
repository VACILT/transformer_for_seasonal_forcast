#!/bin/bash
#SBATCH --job-name=train_transformer    # Job name
#SBATCH --partition=gpu                 # Partition name
#SBATCH --account=bb1438                # Account name
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=4             # Number of tasks per node (GPUs per node)
#SBATCH --gpus=4                        # Number of GPUs per node
#SBATCH --cpus-per-task=32              # CPU cores per task
#SBATCH --mem=0                         # Request all memory available on all nodes
#SBATCH --time=12:00:00                 # Time limit hrs:min:sec
#SBATCH --output=logs/%x-%j.out         # Standard output and error log

module load pytorch

# Set OMP_NUM_THREADS
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Since we're on a single node, we can set MASTER_ADDR to localhost
MASTER_ADDR=127.0.0.1
export MASTER_ADDR
echo "MASTER_ADDR: $MASTER_ADDR"

# Set a master port (choose any free port)
MASTER_PORT=12355
export MASTER_PORT
echo "MASTER_PORT: $MASTER_PORT"

# Set node rank to 0 since there's only one node
NODE_RANK=0
export NODE_RANK
echo "NODE_RANK: $NODE_RANK"

echo "SLURM_NODELIST: $SLURM_NODELIST"

# Run the training script using torchrun
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    main_dino.py \
    --arch vit_small \