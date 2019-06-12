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

import os
import time

from garage.tf.envs import TfEnv
from garage.experiment import run_experiment
from garage.tf.policies import GaussianMLPPolicy
from garage.envs.box2d.cartpole_env import CartpoleEnv

from policy_gradients import config
from policy_gradients.algos import DDOPG


def run_task(*_):

    env = TfEnv(CartpoleEnv())

    policy_cls = GaussianMLPPolicy
    policy_args = dict(env_spec=env.spec,
                       hidden_sizes=(16, 16),
                       learn_std=False)

    algo = DDOPG(env,
                 policy_cls,
                 policy_args)

    algo.train()

log_dir = os.path.join(config.RESULTS_DIR, "cartpole")

run_experiment(run_task,
               n_parallel=1,
               snapshot_mode="gap",
               snapshot_gap=200,
               seed=16,
               log_dir=log_dir)
