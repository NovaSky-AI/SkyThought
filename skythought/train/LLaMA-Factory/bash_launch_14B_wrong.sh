
#!/bin/bash
# Define a unique log file for each job based on dataset name
log_file="./slurm_14B_wrong.log"

command="WANDB_API_KEY=e357e4ac1b5cace6b76e7857c2d97f6a84405006 FORCE_TORCHRUN=1 NNODES=1 NODE_RANK=0 MASTER_PORT=29501 llamafactory-cli train examples/train_full/qwen2_full_sft_14B_wrong.yaml"
# Submit the sbatch job with a time limit of 24 hours and redirect output to the log file
sbatch -p sky -N 1 --gres gpu:8 --exclusive --output=$log_file --wrap="$command"
