#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (C) 2019 Robert Bosch GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

@author: Andreas Doerr
"""

import time

import numpy as np

from garage.tf.misc import tensor_utils


class Sampler:
    """
    Sample paths in an environment according to a parameterized policy.
    """

    def __init__(self,
                 env,
                 policy_cls,
                 policy_args,
                 max_path_length=100,
                 deterministic=False,
                 sleep=0.0):

        self.env = env
        self.policy = policy_cls(**policy_args, name='sampler_policy')

        self.max_path_length = max_path_length
        self.deterministic = deterministic

        self._sleep = sleep

    def get_paths(self, params_list):
        """
        Sample paths according to the policy parameters that are passed to this function.
        args:
            params_list: list of policy parameters
        returns:
            paths that have been sampled in the environment.
        """

        params_list = np.asarray(params_list)
        if params_list.ndim == 1:
            params_list = params_list[None, :]

        return [self.get_path(params) for params in params_list]

    def get_path(self, params):
        """
        Sample a single path in the environment accoring to the policy parameters
        that are passed.
        returns:
            A path consists of a dictionary of observations, actions, etc.
        """

        policy = self.policy
        env = self.env
        env_spec = self.env.spec

        obs = env.reset()
        policy.reset()

        policy.set_param_values(params, trainable=True)

        path = dict(observations=[],
                    actions=[],
                    rewards=[],
                    env_infos=[],
                    agent_infos=[])

        for t in range(self.max_path_length):

            action, agent_info = policy.get_action(obs)
            if self.deterministic:
                action = agent_info['mean']

            next_obs, reward, done, env_info = self.env.step(action)

            if agent_info is None:
                agent_info = dict()
            if env_info is None:
                env_info = dict()
            path['observations'].append(obs)
            path['actions'].append(action)
            path['rewards'].append(reward)
            path['env_infos'].append(env_info)
            path['agent_infos'].append(agent_info)

            if done:
                break

            obs = next_obs

        path['observations'] = env_spec.observation_space.flatten_n(path['observations'])
        path['actions'] = env_spec.action_space.flatten_n(path['actions'])
        path['rewards'] = tensor_utils.stack_tensor_list(path['rewards'])
        path['env_infos'] = tensor_utils.stack_tensor_dict_list(path['env_infos'])
        path['agent_infos'] = tensor_utils.stack_tensor_dict_list(path['agent_infos'])
        path['policy_params'] = policy.get_param_values(trainable=True)

        return path
