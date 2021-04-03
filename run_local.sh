#! /bin/bash

for i in 10;
	do WANDB_RUN_GROUP=dynamicsvinconvbias_hardcodedattention_rhardcoded_iterations${i} python -m train --env_name MiniGrid-FourRooms-v0 --num_processes=64 --level_replay_strategy='random' --level_replay_score_transform='rank' --level_replay_temperature=0.1 --staleness_coef=0.1 --log_interval=10 --log_dir=../vin_logs/${exp_type}_${ID} --num_env_steps=20000000 --num_train_seeds=500 --use_vin --vin_num_iterations=${i} --wandb;
done
