#!/bin/bash

#SBATCH --job-name=sst2_classification
#SBATCH --partition=whartonstat
#SBATCH --output=/home/jrudoler/logs/sst2_classification_%A_%a.out
#SBATCH --error=/home/jrudoler/logs/sst2_classification_%A_%a.err
#SBATCH --array=0-39
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2

# Define the parameter combinations
# declare -a model_ids=("meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "google/gemma-2-2b-it" "google/gemma-2-9b-it")
declare -a model_ids=("meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "google/gemma-2-2b-it" "google/gemma-2-9b-it" "tiiuae/Falcon3-10B-Instruct")
# declare -a model_ids=("mistralai/Mistral-7B-Instruct-v0.3")
# declare -a model_ids=("google/gemma-2-2b-it")
# declare -a model_ids=("meta-llama/Llama-3.3-70B-Instruct")
# declare -a n_contexts=(16 32 64 128 256 512)
declare -a n_contexts=(0 1 2 4 8 16 32 64)
batch_size=2  # Use a single batch size
seed=42  # Use a single seed
n_target=1500

# Calculate the total number of combinations
total_combinations=$((${#model_ids[@]} * ${#n_contexts[@]}))

# Ensure the array index is within the range of combinations
if [ $SLURM_ARRAY_TASK_ID -ge $total_combinations ]; then
    echo "Array index $SLURM_ARRAY_TASK_ID is out of range for $total_combinations combinations."
    exit 1
fi

# Calculate the specific combination for this task
model_id_index=$(($SLURM_ARRAY_TASK_ID / ${#n_contexts[@]}))
n_context_index=$(($SLURM_ARRAY_TASK_ID % ${#n_contexts[@]}))

model_id=${model_ids[$model_id_index]}
n_context=${n_contexts[$n_context_index]}

# Run the Python script with the selected parameters
poetry run python classify_sst2.py --model_id $model_id --batch_size $batch_size --seed $seed --n_context $n_context --n_target $n_target