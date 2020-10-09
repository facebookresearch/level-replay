# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import sys
import time
from collections import deque
import timeit
import logging

import numpy as np
import torch
from baselines.logger import HumanOutputFormat

from level_replay import algo, utils
from level_replay.model import model_for_env_name
from level_replay.storage import RolloutStorage
from level_replay.file_writer import FileWriter
from level_replay.envs import make_lr_venv
from level_replay.arguments import parser
from test import evaluate


os.environ["OMP_NUM_THREADS"] = "1"

last_checkpoint_time = None


def train(args, seeds):
    global last_checkpoint_time
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if 'cuda' in device.type:
        print('Using CUDA\n')

    torch.set_num_threads(1)

    utils.seed(args.seed)

    # Configure logging
    if args.xpid is None:
        args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.expandvars(os.path.expanduser(args.log_dir))
    plogger = FileWriter(
        xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir,
        seeds=seeds,
    )
    stdout_logger = HumanOutputFormat(sys.stdout)

    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid, "model.tar"))
    )

    # Configure actor envs
    start_level = 0
    if args.full_train_distribution:
        num_levels = 0
        level_sampler_args = None
        seeds = None
    else:
        num_levels = 1
        level_sampler_args = dict(
            num_actors=args.num_processes,
            strategy=args.level_replay_strategy,
            replay_schedule=args.level_replay_schedule,
            score_transform=args.level_replay_score_transform,
            temperature=args.level_replay_temperature,
            eps=args.level_replay_eps,
            rho=args.level_replay_rho,
            nu=args.level_replay_nu, 
            alpha=args.level_replay_alpha,
            staleness_coef=args.staleness_coef,
            staleness_transform=args.staleness_transform,
            staleness_temperature=args.staleness_temperature
        )
    envs, level_sampler = make_lr_venv(
        num_envs=args.num_processes, env_name=args.env_name,
        seeds=seeds, device=device,
        num_levels=num_levels, start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        level_sampler_args=level_sampler_args)
    
    is_minigrid = args.env_name.startswith('MiniGrid')

    actor_critic = model_for_env_name(args, envs)       
    actor_critic.to(device)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space,
                                actor_critic.recurrent_hidden_state_size)
        
    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)

    def checkpoint():
        if args.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": actor_critic.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "args": vars(args),
            },
            checkpointpath,
        )

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        env_name=args.env_name)

    level_seeds = torch.zeros(args.num_processes)
    if level_sampler:
        obs, level_seeds = envs.reset()
    else:
        obs = envs.reset()
    level_seeds = level_seeds.unsqueeze(-1)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    timer = timeit.default_timer
    update_start_time = timer()
    for j in range(num_updates):
        actor_critic.train()
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                obs_id = rollouts.obs[step]
                value, action, action_log_dist, recurrent_hidden_states = actor_critic.act(
                    obs_id, rollouts.recurrent_hidden_states[step], rollouts.masks[step])
                action_log_prob = action_log_dist.gather(-1, action)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            # Reset all done levels by sampling from level sampler
            for i, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

                if level_sampler:
                    level_seeds[i][0] = info['level_seed']

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts.insert(
                obs, recurrent_hidden_states, 
                action, action_log_prob, action_log_dist, 
                value, reward, masks, bad_masks, level_seeds)

        with torch.no_grad():
            obs_id = rollouts.obs[-1]
            next_value = actor_critic.get_value(
                obs_id, rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
            
        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        # Update level sampler
        if level_sampler:
            level_sampler.update_with_rollouts(rollouts)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()
        if level_sampler:
            level_sampler.after_update()

        # Log stats every log_interval updates or if it is the last update
        if (j % args.log_interval == 0 and len(episode_rewards) > 1) or j == num_updates - 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps

            update_end_time = timer()
            num_interval_updates = 1 if j == 0 else args.log_interval
            sps = num_interval_updates*(args.num_processes * args.num_steps) / (update_end_time - update_start_time)
            update_start_time = update_end_time

            logging.info(f"\nUpdate {j} done, {total_num_steps} steps\n  ")
            logging.info(f"\nEvaluating on {args.num_test_seeds} test levels...\n  ")
            eval_episode_rewards = evaluate(args, actor_critic, args.num_test_seeds, device)

            logging.info(f"\nEvaluating on {args.num_test_seeds} train levels...\n  ")
            train_eval_episode_rewards = evaluate(args, actor_critic, args.num_test_seeds, device, start_level=0, num_levels=args.num_train_seeds, seeds=seeds)

            stats = { 
                "step": total_num_steps,
                "pg_loss": action_loss,
                "value_loss": value_loss,
                "dist_entropy": dist_entropy,
                "train:mean_episode_return": np.mean(episode_rewards),
                "train:median_episode_return": np.median(episode_rewards),
                "test:mean_episode_return": np.mean(eval_episode_rewards),
                "test:median_episode_return": np.median(eval_episode_rewards),
                "train_eval:mean_episode_return": np.mean(train_eval_episode_rewards),
                "train_eval:median_episode_return": np.median(train_eval_episode_rewards),
                "sps": sps,
            }
            if is_minigrid:
                stats["train:success_rate"] = np.mean(np.array(episode_rewards) > 0)
                stats["train_eval:success_rate"] = np.mean(np.array(train_eval_episode_rewards) > 0)
                stats["test:success_rate"] = np.mean(np.array(eval_episode_rewards) > 0)

            if j == num_updates - 1:
                logging.info(f"\nLast update: Evaluating on {args.num_test_seeds} test levels...\n  ")
                final_eval_episode_rewards = evaluate(args, actor_critic, args.final_num_test_seeds, device)

                mean_final_eval_episode_rewards = np.mean(final_eval_episode_rewards)
                median_final_eval_episide_rewards = np.median(final_eval_episode_rewards)
                
                plogger.log_final_test_eval({
                    'num_test_seeds': args.final_num_test_seeds,
                    'mean_episode_return': mean_final_eval_episode_rewards,
                    'median_episode_return': median_final_eval_episide_rewards
                })

            plogger.log(stats)
            if args.verbose:
                stdout_logger.writekvs(stats)

        # Log level weights
        if level_sampler and j % args.weight_log_interval == 0:
            plogger.log_level_weights(level_sampler.sample_weights())

        # Checkpoint 
        timer = timeit.default_timer
        if last_checkpoint_time is None:
            last_checkpoint_time = timer()
        try:
            if j == num_updates - 1 or \
                (args.save_interval > 0 and timer() - last_checkpoint_time > args.save_interval * 60):  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()
        except KeyboardInterrupt:
            return


def generate_seeds(num_seeds, base_seed=0):
    return [base_seed + i for i in range(num_seeds)]


def load_seeds(seed_path):
    seed_path = os.path.expandvars(os.path.expanduser(seed_path))
    seeds = open(seed_path).readlines()
    return [int(s) for s in seeds] 


if __name__ == "__main__":
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    if args.seed_path:
        train_seeds = load_seeds(args.seed_path)
    else:
        train_seeds = generate_seeds(args.num_train_seeds)

    train(args, train_seeds)
