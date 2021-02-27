#!/bin/bash
#
# Execute from cwd
#$ -cwd
#
# This is a day long job
#$ -l day
#
# Uses 32GB memory
#$ -l vf=32G
#
# Uses 32GB memory
#$ -pe smp 8
#
# Uses 1 GPU
#$ -l gpus=1
#
# Runs 48 jobs
#$ -t 1-5
#
# Runs at most 20 jobs at once
#$ -tc 16

ID=$(($SGE_TASK_ID - 1))
exp_type=minigrid_multiroomsmall_trainlevels3000_ppo

source ~/miniconda3/bin/activate && conda activate vin && WANDB_RUN_GROUP=${exp_type} python -m train --env_name MiniGrid-MultiRoom-N2-S4-v0 --num_processes=64 --level_replay_strategy='random' --level_replay_score_transform='rank' --level_replay_temperature=0.1 --staleness_coef=0.1 --log_interval=10 --wandb --log_dir=/data/people/jroy1/vin/inigridmultiroomtest_${ID} --num_env_steps=100000000 --num_train_seeds=3000
