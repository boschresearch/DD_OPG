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

import numpy as np
import tensorflow as tf

from garage.tf.misc.tensor_utils import flatten_batch
from garage.tf.misc.tensor_utils import pad_tensor_n


class DDOPGModel:
    """
    Importance sampling inspired return model
    """

    def __init__(self,
                 policy_cls,
                 policy_args,
                 max_path_length=100,
                 log_std=2.0,
                 delta=0.1,
                 return_prior_max=1000,
                 eps=1e-8):

        self.policy_cls = policy_cls
        self.policy_args = policy_args

        self.max_path_length = max_path_length
        self.log_std = log_std
        self.delta = delta
        self.return_prior_max = return_prior_max
        self.eps = eps

        self._build_graph()

    def compute_logps(self, paths, paths_data):
        """
        Computes logps matrix for all policies and all trajectories in paths
        """
        sess = tf.get_default_session()

        feed = dict()
        feed[self.obs_var] = paths_data['obs']
        feed[self.action_var] = paths_data['acts']
        feed[self.valid_var] = paths_data['valids']
        feed[self.log_std_var] = self.log_std

        logps = []
        for path in paths:
            self.policy.set_param_values(path['policy_params'], trainable=True)
            logps.append(np.squeeze(sess.run(self.test_logps, feed)))
        return np.asarray(logps)

    def _build_graph(self):

        self.policy = self.policy_cls(**self.policy_args, name='ddopg_model_policy')

        self.var_list = self.policy.get_params(trainable=True)
        self.var_shapes = [var.shape for var in self.var_list]
        self.n_params = sum([shape.num_elements() for shape in self.var_shapes])

        self.policy_dist = self.policy.distribution
        self.policy_params_shapes = [param.shape for param in self.policy.get_params(trainable=True)]

        observation_space = self.policy.observation_space
        action_space = self.policy.action_space

        self.obs_var = observation_space.new_tensor_variable(name='obs', extra_dims=2)                       # (n_paths, H, Dy)
        self.action_var = action_space.new_tensor_variable(name='action', extra_dims=2)                      # (n_paths, H, Du)
        self.path_return_var = tf.placeholder(tf.float32, [None], 'path_return')                             # (n_paths, )
        self.valid_var = tf.placeholder(tf.float32, [None, None], 'valid')                                   # (n_paths, H)
        self.train_logps = tf.placeholder(dtype=tf.float32, shape=[None, None], name='train_logps_pre')      # (n_paths, n_train)

        self.log_std_var = tf.placeholder(dtype=tf.float32, shape=[], name='log_std')
        self.delta_var = tf.placeholder(dtype=tf.float32, shape=[], name='delta')

        self.input_vars = [self.obs_var,
                           self.action_var,
                           self.path_return_var,
                           self.valid_var,
                           self.train_logps,
                           self.delta_var,
                           self.log_std_var]

        # Flatten observation and actions for vectorized computations
        self.obs_flat = flatten_batch(self.obs_var, name='obs_flat')            # (n_paths * H, Dy)
        self.action_flat = flatten_batch(self.action_var, name='action_flat')   # (n_paths * H, Du)

        # Shape of training data: (# of paths, path horizon) = (N_train, H)
        self.batch_shape = tf.shape(self.obs_var)[0:2]

        # Compute logp for all policy
        dist_info_flat = self.policy.dist_info_sym(self.obs_flat, name='dist_info_flat')
        dist_info_flat['log_std'] = self.log_std_var * tf.ones_like(dist_info_flat['mean'])

        test_logp_flat = self.policy_dist.log_likelihood_sym(self.action_flat, dist_info_flat, name='logp_flat')
        test_logp_full = tf.reshape(test_logp_flat, self.batch_shape)           # (n_epochs, H)
        self.test_logps = tf.reduce_sum(test_logp_full * self.valid_var, axis=1)[None, :]

        self.all_logps = tf.concat((self.train_logps,                           # (n_train + n_test, n_paths)
                                    self.test_logps), axis=0)

        # Prevent exp() overflow by shifting logps
        self.logp_max = tf.reduce_max(self.all_logps, axis=0)                   # (n_paths, )

        self.train_logps_0 = self.train_logps - self.logp_max                   # (n_train, n_paths)
        self.test_logps_0 = self.test_logps - self.logp_max                     # (n_paths, )
        self.train_liks = tf.exp(self.train_logps_0)                            # (n_train, n_paths)
        self.test_liks = tf.exp(self.test_logps_0)                              # (n_paths, )

        # Mean traj lik for empirical mixture distribution
        self.train_mean_liks = tf.reduce_mean(self.train_liks, axis=0) + self.eps  # (n_paths, )

        # Compute prediction for all training policies
        train_res = self._compute_prediction_vec(self.train_liks)
        self.J_train = train_res[0]
        self.J2_train = train_res[1]
        self.J_var_train = train_res[2]
        self.J_unc_train = train_res[3]
        self.w_train = train_res[4]
        self.wn_train = train_res[5]
        self.ess_train = train_res[6]

        # Compute prediction for all test policies
        test_res = self._compute_prediction_vec(self.test_liks)
        self.J_test = test_res[0]
        self.J2_test = test_res[1]
        self.J_var_test = test_res[2]
        self.J_unc_test = test_res[3]
        self.w_test = test_res[4]
        self.wn_test = test_res[5]
        self.ess_test = test_res[6]

    def _compute_prediction_vec(self, target_liks):
        """
        Part of the tensorflow graphs that computes differnt target values based
        on the likelihoods, i.e. the objective function, ess, importance sampling
        weights etc.
        """
        eps = self.eps
        delta = self.delta_var
        R = self.path_return_var

        # Vectorized computation of IS prediction
        w = target_liks / self.train_mean_liks                                  # (N_targets, N_traj) / (N_traj, ) = (N_targets, N_traj)
        wn = w / (tf.reduce_sum(w, axis=1) + eps)[:, None]                      # (N_targets, N_traj) / (N_targets, 1) = (N_targets, N_traj)

        ess = tf.reduce_sum(wn, axis=1)**2 / (tf.reduce_sum(wn**2, axis=1) + eps) # (N_targets, ) / (N_targets, ) = (N_targets, )

        J = tf.reduce_sum(wn * R, axis=1)                                       # (N_targets, )
        J2 = tf.reduce_sum(wn * R**2, axis=1)                                   # (N_targets, )
        J_var = tf.clip_by_value(J2 - J**2, eps, np.inf)                        # (N_targets, )
        J_unc = self.return_prior_max * tf.sqrt((1 - delta) / (delta * ess + eps))    # (N_targets, )

        return J, J2, J_var, J_unc, w, wn, ess

    def _preprocess_paths(self, paths):
        """Preprocesses paths (patting, concatenation, returns)"""
        obs = pad_tensor_n([path['observations'] for path in paths], self.max_path_length)
        acts = pad_tensor_n([path['actions'] for path in paths], self.max_path_length)
        valids = pad_tensor_n([np.ones_like(path["rewards"]) for path in paths], self.max_path_length)
        path_rets = np.asarray([np.sum(p['rewards']) for p in paths])
        policy_params = np.asarray([path['policy_params'] for path in paths])

        paths_data = dict(obs=obs,
                          acts=acts,
                          path_rets=path_rets,
                          valids=valids,
                          policy_params=policy_params)

        return paths_data

    def _get_inputs(self, paths):
        """
        Generates the input variables for the tensorflow graphs such that they
        can be used to construct a feed_dict.
        args:
            paths, list, list of paths
        returns:
            list of input variables for a feed_dict.
        """
        paths_data = self._preprocess_paths(paths)
        train_logps = self.compute_logps(paths, paths_data)

        inputs = [paths_data['obs'],
                  paths_data['acts'],
                  paths_data['path_rets'],
                  paths_data['valids'],
                  train_logps,
                  self.delta,
                  self.log_std]

        return inputs

    def get_feed(self, paths):
        """
        Generates a feed_dict that can be fed to the tensorflow graph.
        args:
            paths, list, list of paths
        returns:
            a tensorflow feed dict.
        """
        inputs = self._get_inputs(paths)
        return dict((key, value) for key, value in zip(self.input_vars, inputs))
