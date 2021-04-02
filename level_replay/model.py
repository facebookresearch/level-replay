# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/auto-drac/blob/master/ucb_rl2_meta/model.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import wandb
import plotly.express as px
import torch.autograd.profiler as profiler

from level_replay.distributions import Categorical
from level_replay.utils import init
from level_replay.envs import PROCGEN_ENVS


init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0))

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), nn.init.calculate_gain('relu'))

init_tanh_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), np.sqrt(2))

def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def model_for_env_name(args, env):
    if args.env_name in PROCGEN_ENVS:
        model = Policy(
            env.observation_space.shape, env.action_space.n,
            arch=args.arch,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size})
    elif args.env_name.startswith('MiniGrid'):
        model = MinigridPolicy(
            env.observation_space.shape, env.action_space.n,
            arch=args.arch, vin=args.use_vin, num_iterations=args.vin_num_iterations, wandb=args.wandb, spatial_transformer=args.vin_spatial_transformer)
    else:
        raise ValueError(f'Unsupported env {env}')

    return model


class Flatten(nn.Module):
    """
    Flatten a tensor
    """
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class Policy(nn.Module):
    """
    Actor-Critic module 
    """
    def __init__(self, obs_shape, num_actions, arch='small', base_kwargs=None):
        super(Policy, self).__init__()
        
        if base_kwargs is None:
            base_kwargs = {}
        
        if len(obs_shape) == 3:
            if arch == 'small':
                base = SmallNetBase
            else:
                base = ResNetBase
        elif len(obs_shape) == 1:
            base = MLPBase

        self.base = base(obs_shape[0], **base_kwargs)
        self.dist = Categorical(self.base.output_size, num_actions)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # action_log_probs = dist.log_probs(action)
        action_log_dist = dist.logits
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    """
    Actor-Critic network (base class)
    """
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class MLPBase(NNBase):
    """
    Multi-Layer Perceptron
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        self.actor = nn.Sequential(
            init_tanh_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_tanh_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_tanh_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_tanh_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class BasicBlock(nn.Module):
    """
    Residual Network Block
    """
    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1,1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class ResNetBase(NNBase):
    """
    Residual Network 
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=256, channels=[16,32,32]):
        super(ResNetBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=1))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class SmallNetBase(NNBase):
    """
    Residual Network 
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=256):
        super(SmallNetBase, self).__init__(recurrent, num_inputs, hidden_size)

        self.conv1 = Conv2d_tf(3, 16, kernel_size=8, stride=4)
        self.conv2 = Conv2d_tf(16, 32, kernel_size=4, stride=2)

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc(x))

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MinigridPolicy(nn.Module):
    """
    Actor-Critic module 
    """
    def __init__(self, obs_shape, num_actions, arch='small', base_kwargs=None, vin=False, num_iterations=10, wandb=False, spatial_transformer=False):
        super(MinigridPolicy, self).__init__()

        self.vin = vin
        self.num_iterations = num_iterations
        self.wandb = wandb
        self.spatial_transformer = spatial_transformer

        if self.wandb:
            self.image_log_freq = 1024
            self.image_log_i = 0
        
        if base_kwargs is None:
            base_kwargs = {}
        
        self.final_channels = 32 if arch == 'small' else 64

        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            # nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, self.final_channels, (3, 3), padding=(1, 1)),
            nn.ReLU()
        )
        self.image_conv_critic = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, self.final_channels, (3, 3), padding=(1, 1)),
            nn.ReLU()
        )

        n = obs_shape[-2]
        m = obs_shape[-1]

        zeros_in = torch.zeros(obs_shape).unsqueeze(0)
        zeros_out = self.image_conv(zeros_in)
        self.image_embedding_shape = tuple(zeros_out.shape[1:])

        zeros_in = torch.zeros(obs_shape).unsqueeze(0)
        zeros_out = self.image_conv_critic(zeros_in)
        self.critic_embedding_shape = tuple(zeros_out.shape[1:])

        if self.vin and self.spatial_transformer:
            self.image_embedding_size = self.image_embedding_shape[1] * self.image_embedding_shape[2]
            self.actor_embedding_size = self.image_embedding_size + self.image_embedding_size * self.final_channels
        elif self.vin and not self.spatial_transformer:
            self.image_embedding_size = self.critic_embedding_shape[1] * self.critic_embedding_shape[2] * self.final_channels
            if True:
                self.actor_embedding_size = num_actions# + self.image_embedding_size * self.final_channels
            else:
                self.actor_embedding_size = 1 + self.image_embedding_size * self.final_channels
        else:
            self.image_embedding_size = self.critic_embedding_shape[1] * self.critic_embedding_shape[2] * self.final_channels
            self.actor_embedding_size = self.image_embedding_shape[1] * self.image_embedding_shape[2] * self.final_channels

        # Define VIN
        if self.vin:
            self.h = nn.Sequential(
                nn.Conv2d(in_channels=self.final_channels, out_channels=self.final_channels, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.final_channels, out_channels=self.final_channels, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.final_channels, out_channels=150, kernel_size=(3, 3), stride=1, padding=1),
                nn.ReLU(),
            )

            self.r = nn.Conv2d(in_channels=150, out_channels=1, kernel_size=(1, 1), stride=1, padding=0, bias=False)
            self.q = nn.Conv2d(in_channels=1, out_channels=num_actions, kernel_size=(3, 3), stride=1, padding=1, bias=False)

            self.p_condensor = nn.Conv2d(in_channels=self.final_channels, out_channels=1, kernel_size=(3, 3), stride=1, padding=1)
            self.p = nn.parameter.Parameter(torch.zeros(1, num_actions, 3, 3), requires_grad=True)

            self.w = nn.parameter.Parameter(torch.zeros(num_actions, 1, 3, 3), requires_grad=True)


            if self.spatial_transformer:
                self.localization = nn.Sequential(
                    nn.Conv2d(self.final_channels, out_channels=8, kernel_size=7),
                    nn.MaxPool2d(2, stride=2),
                    nn.ReLU(True),
                    nn.Conv2d(8, out_channels=10, kernel_size=5),
                    nn.MaxPool2d(2, stride=2),
                    nn.ReLU(True)
                )

                # zeros_localized = self.localization(zeros_out)
                zeros_localized = zeros_out
                self.localization_out_dim = zeros_localized.shape[2] * zeros_localized.shape[3] * zeros_localized.shape[1]

                # Regressor for the 3 * 2 affine matrix
                self.fc_loc = nn.Sequential(
                    nn.Linear(self.localization_out_dim, 32),
                    nn.ReLU(True),
                    nn.Linear(32, 3 * 2)
                )

                # Initialize weights/bias with identity transform
                self.fc_loc[2].weight.data.zero_()
                self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
            else:
                self.critic_attention = nn.Sequential(
                    nn.Conv2d(in_channels=self.final_channels, out_channels=self.final_channels, kernel_size=(3, 3), padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=self.final_channels, out_channels=self.final_channels, kernel_size=(3, 3), padding=1),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=self.final_channels, out_channels=1, kernel_size=(3, 3), padding=1),
                )

        # Define actor's model
        self.actor_base = nn.Sequential(
            init_tanh_(nn.Linear(self.actor_embedding_size, 64)),
            nn.Tanh(),
        )

        # Define critic's model
        self.critic = nn.Sequential(
            init_tanh_(nn.Linear(self.image_embedding_size, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 1))
        )

        self.dist = Categorical(64, num_actions)

# New VIN Bits
        self.state_shape = self.image_embedding_shape[1:3]
        self.half_kernel = self.p.shape[2] // 2

        rows = np.arange(self.state_shape[0])
        cols = np.arange(self.state_shape[0])

        self.trans_window_row_starts = np.maximum(rows - self.half_kernel, 0)
        self.trans_window_row_ends = np.minimum(rows + self.half_kernel + 1, self.state_shape[0] - 1)
        self.trans_window_col_starts = np.maximum(cols - self.half_kernel, 0)
        self.trans_window_col_ends = np.minimum(cols + self.half_kernel + 1, self.state_shape[0] - 1)

        self.rep_window_row_starts = np.maximum(self.trans_window_row_starts-rows, 0)
        self.rep_window_row_ends = self.rep_window_row_starts + (self.trans_window_row_ends - self.trans_window_row_starts)
        self.rep_window_col_starts = np.maximum(self.trans_window_col_starts-cols, 0)
        self.rep_window_col_ends = self.rep_window_col_starts + (self.trans_window_col_ends - self.trans_window_col_starts)
# End New VIN Bits

        apply_init_(self.modules())

        self.train()

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return 1

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def stn(self, img_in, values):
        # xs = self.localization(img_in)
        xs = img_in
        xs = xs.reshape(-1, self.localization_out_dim)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, values.size())
        values = F.grid_sample(values, grid)

        return values

    def make_image(self, x):
        img = x.detach().cpu().numpy().reshape((x.shape[1], x.shape[2]))
        fig = px.imshow(img)
        return fig

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        x = inputs
        img_out = self.image_conv(x)
        img_out_critic = self.image_conv_critic(x)
        img_out_actor = img_out
        x = img_out.flatten(1, -1)
        x_actor = img_out_actor.flatten(1, -1)
        x_critic = img_out_critic.flatten(1, -1)

        if self.vin:
            out_value_rep, out_reward, out_q, transition_info = self.get_value_vin(img_out, num_iterations=self.num_iterations, inputs=inputs)

            if self.spatial_transformer:
                weighted_out_value_rep = self.stn(img_out, out_value_rep)

                value_in = weighted_out_value_rep.flatten(1, -1)
                value = self.critic(value_in)
                # actor_in = torch.cat([x_actor, value_in.detach()], dim=1)
                # actor_in = q.detach()
                actor_in = q
            else:
                if True:
                    critic_attention = (inputs[:, 0, :, :] == 10).unsqueeze(1).float()
                else:
                    critic_attention = self.critic_attention(img_out)
                    critic_attention = F.softmax(critic_attention.view(critic_attention.shape[0], critic_attention.shape[1], -1), dim=2).view_as(critic_attention)
                assert(out_value_rep.shape == critic_attention.shape)
                weighted_out_value_rep = out_value_rep * critic_attention
                # value_in = weighted_out_value_rep.flatten(1, -1)
                # value = value_in.sum(dim=1, keepdim=True)
                value = self.critic(x_critic)

                q = out_q * critic_attention
                q = q.sum(dim=[2, 3])
                if True:
                    # actor_in = torch.cat([x_actor, q.detach()], dim=1)
                    # actor_in = q.detach()
                    actor_in = q
                else:
                    assert value.shape[1] == 1
                    actor_in = torch.cat([x_actor, value.detach()], dim=1)
                    assert actor_in.shape[1] == 1 + self.image_embedding_size * self.final_channels

            if self.wandb:
                if self.image_log_i % self.image_log_freq == 0:
                    if self.spatial_transformer:
                        wandb.log({"inputs": wandb.Image(inputs[0]), "out_value_rep": wandb.Image(out_value_rep[0]), "weighted_out_value_rep": wandb.Image(weighted_out_value_rep[0]), "r": wandb.Image(out_reward[0])})
                    else:
                        inputs_img = wandb.Image(inputs[0])
                        critic_attention_img = self.make_image(critic_attention[0])
                        out_value_rep_img = self.make_image(out_value_rep[0])
                        weighted_out_value_rep_img = self.make_image(weighted_out_value_rep[0])
                        out_reward_img = self.make_image(out_reward[0])
                        transition_info_img = self.make_image(transition_info[0])
                        wandb.log({"inputs": inputs_img, "critic_attention": critic_attention_img, "out_value_rep": out_value_rep_img, "weighted_out_value_rep": weighted_out_value_rep_img, "reward": out_reward_img, 'transition_info': transition_info_img})
                self.image_log_i += 1

        else:
            value_in = x_critic
            actor_in = x_actor
            value = self.critic(value_in)

        actor_features = self.actor_base(actor_in)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        # action_log_probs = dist.log_probs(action)
        action_log_dist = dist.logits
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_dist, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        x = inputs
        img_out = self.image_conv_critic(x)
        #if self.vin:
        #    x, _, _ = self.get_value_vin(img_out, num_iterations=self.num_iterations, inputs=inputs)

        #    if self.spatial_transformer:
        #        x = self.stn(img_out, x)
        #        x = x.flatten(1, -1)
        #        out = self.critic(x)
        #    else:
        #        if True:
        #            critic_attention = (inputs[:, 0, :, :] == 10).unsqueeze(1).float()
        #        else:
        #            critic_attention = self.critic_attention(img_out)
        #            critic_attention = F.softmax(critic_attention.view(critic_attention.shape[0], critic_attention.shape[1], -1), dim=2).view_as(critic_attention)
        #        x *= critic_attention
        #        x = x.flatten(1, -1)
        #        out = x.sum(dim=1, keepdim=True)
        #        assert out.shape[1] == 1
        #else:
        x = img_out
        x = x.flatten(1, -1)
        out = self.critic(x)

        return out

    def get_value_vin(self, representation, num_iterations, inputs):
        h = self.h(representation)
        if False:
            r = self.r(h)
        else:
            r = (inputs[:, 0, :, :] == 8).unsqueeze(1).float() - 0.1
            # r[(inputs[:, 0, :, :] == 2).unsqueeze(1)] = -0.5
        q = self.q(r)
        v, _ = torch.max(q, dim=1, keepdim=True)

        r_img = self.q(r)
        transition_info = self.p_condensor(representation)
        qt = torch.empty_like(r_img)

        if False:
            print("AAAAAAAAAAAAAA")
            with profiler.profile(record_shapes=True, profile_memory=True, use_cuda=True) as prof:
                with profiler.record_function("agent update"):
                    for i in range(num_iterations - 1):
                        q = self.eval_q(r_img, v, transition_info, qt)
                        v, _ = torch.max(q, dim=1, keepdim=True)
            print(prof.key_averages().table(sort_by='cuda_memory_usage'))
            assert False
        else:
            for i in range(num_iterations - 1):
                q = self.eval_q(r_img, v, transition_info, qt)
                v, _ = torch.max(q, dim=1, keepdim=True)

        return v, r, q, transition_info

    def eval_q(self, r_img, v, transition_info, qt):
        # Get reward image
        # qt = torch.empty_like(r_img)

        # Get qt-image
        for row in range(self.state_shape[0]):
            for col in range(self.state_shape[1]):
                trans_window_row_start = self.trans_window_row_starts[row]
                trans_window_row_end = self.trans_window_row_ends[row]
                trans_window_col_start = self.trans_window_col_starts[col]
                trans_window_col_end = self.trans_window_col_ends[col]

                rep_window_row_start = self.rep_window_row_starts[row]
                rep_window_row_end = self.rep_window_row_ends[row]
                rep_window_col_start = self.rep_window_col_starts[col]
                rep_window_col_end = self.rep_window_col_ends[col]

                transition_window = self.p[:, :, rep_window_row_start:rep_window_row_end, rep_window_col_start:rep_window_col_end] * transition_info[:, :, trans_window_row_start:trans_window_row_end, trans_window_col_start:trans_window_col_end]
                transition_window = F.softmax(transition_window.view(transition_window.shape[0], transition_window.shape[1], -1), dim=2).view_as(transition_window)
                qt[:, :, row, col] = (transition_window * v[:, :, trans_window_row_start:trans_window_row_end, trans_window_col_start:trans_window_col_end]).sum([2, 3])

        # Sum q-transition and reward image
        q = r_img + qt

        return q

        # return F.conv2d(
        #     torch.cat([r, v], 1),
        #     torch.cat([self.q.weight, self.w], 1),
        #     stride=1,
        #     padding=1)


    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        x = inputs
        img_out = self.image_conv(x)
        img_out_critic = self.image_conv_critic(x)
        # img_out_actor = self.image_conv_actor(x)
        img_out_actor = img_out
        x = img_out.flatten(1, -1)
        x_actor = img_out_actor.flatten(1, -1)
        x_critic = img_out_critic.flatten(1, -1)


        if self.vin:
            out_value_rep, _, out_q, _ = self.get_value_vin(img_out, num_iterations=self.num_iterations, inputs=inputs)

            if self.spatial_transformer:
                out_value_rep = self.stn(img_out, out_value_rep)
                value_in = out_value_rep.flatten(1, -1)
                value = self.critic(value_in)
                actor_in = torch.cat([x_actor, value_in.detach()], dim=1)
            else:
                if True:
                    critic_attention = (inputs[:, 0, :, :] == 10).unsqueeze(1).float()
                else:
                    critic_attention = self.critic_attention(img_out)
                    critic_attention = F.softmax(critic_attention.view(critic_attention.shape[0], critic_attention.shape[1], -1), dim=2).view_as(critic_attention)
                # out_value_rep *= critic_attention

                # value_in = out_value_rep.flatten(1, -1)
                # value = value_in.sum(dim=1, keepdim=True)
                value = self.critic(x_critic)
                q = out_q * critic_attention
                q = q.sum(dim=[2, 3])
                if True:
                    # actor_in = torch.cat([x_actor, q.detach()], dim=1)
                    # actor_in = q.detach()
                    actor_in = q
                else:
                    assert value.shape[1] == 1
                    actor_in = torch.cat([x_actor, value.detach()], dim=1)
                    assert actor_in.shape[1] == 1 + self.image_embedding_size * self.final_channels

        else:
            value_in = x_critic
            actor_in = x_actor
            value = self.critic(value_in)

        actor_features = self.actor_base(actor_in)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
