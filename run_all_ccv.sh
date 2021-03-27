#!/bin/bash
#
# This is a half-day long job
#SBATCH -t 11:00:00
#
# Uses 1 GPU
#SBATCH -p gpu --gres=gpu:1
#
# Uses 32 gb ram
#SBATCH --mem=32G
#
# Uses 8 cpu cores
#SBATCH -c 4
#
# Array
#SBATCH --array=10

ID=$(($SLURM_ARRAY_TASK_ID - 1))
NUM_ITERATIONS=$((ID * 10))
exp_type=separatecritic_hardcodedattention_rhardcodedwithobstaclesneg0.5_iterations${NUM_ITERATIONS}_fourrooms_biggerh

source ~/miniconda3/bin/activate && conda activate level-replay && WANDB_RUN_GROUP=${exp_type} python3 -m train --env_name MiniGrid-FourRooms-v0 --num_processes=64 --level_replay_strategy='random' --level_replay_score_transform='rank' --level_replay_temperature=0.1 --staleness_coef=0.1 --log_interval=10 --log_dir=../vin_logs/${exp_type}_${ID} --num_env_steps=100000000 --num_train_seeds=500 --use_vin --vin_num_iterations=${NUM_ITERATIONS} --wandb
