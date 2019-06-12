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

import numpy as np

import matplotlib.pyplot as plt


def print_paths_info(paths):
    res = get_paths_info(paths)
    print(res)


def get_paths_info(paths):

    num_paths = len(paths)
    epoche_lengths = np.asarray([len(path['rewards']) for path in paths])
    Rs = np.asarray([sum(path['rewards']) for path in paths])

    params = np.asarray([path['policy_params'] for path in paths])
    num_unique_policies = len(np.unique(params, axis=0))

    res = ''

    res += '\n--------------------------------------\n'
    res += 'Number of paths:      {}\n'.format(num_paths)
    res += 'Epoche lengths:\n'
    res += '                mean: {:.3f}\n'.format(epoche_lengths.mean())
    res += '                std:  {:.3f}\n'.format(epoche_lengths.std())
    res += '                min:  {}\n'.format(epoche_lengths.min())
    res += '                max:  {}\n'.format(epoche_lengths.max())
    res += 'Number of steps:      {}\n'.format(epoche_lengths.sum())
    res += 'Undiscounted return:\n'
    res += '                mean: {:.3f}\n'.format(Rs.mean())
    res += '                std:  {:.3f}\n'.format(Rs.std())
    res += '                min:  {:.3f}\n'.format(Rs.min())
    res += '                max:  {:.3f}\n'.format(Rs.max())
    res += 'Unique policies:      {}\n'.format(num_unique_policies)
    res += '--------------------------------------\n'

    return res


def visualize_cartpole_paths(env, paths, figure_id=0):

    from garage.tf.envs import TfEnv

    if type(env) is TfEnv:
        env = env.unwrapped

    Hs = [len(path['rewards']) for path in paths]
    H_max = max(Hs)
    t_max = np.arange(H_max)

    plt.figure(figure_id)
    plt.clf()

    # Cart position
    x_max = env.max_cart_pos
    x_done = env.reset_range * x_max
    plt.subplot(3, 2, 1)

    plt.vlines(0, -x_done, x_done, 'r', label='reset range', linewidths=20)
    plt.plot(t_max, x_max * np.ones_like(t_max), ':r', label='done')
    plt.plot(t_max, -x_max * np.ones_like(t_max), ':r')

    for path in paths:
        H = len(path['rewards'])
        t = np.arange(H)
        plt.plot(t, path['observations'][:, 0], color='C0')

    plt.xlabel('Time t [-]')
    plt.ylabel('Cart position [m]')

    # Cart speed
    xd_max = env.max_cart_speed
    xd_done = env.reset_range * xd_max
    plt.subplot(3, 2, 2)

    plt.vlines(0, -xd_done, xd_done, 'r', label='reset range', linewidths=20)
    plt.plot(t_max, xd_max * np.ones_like(t_max), ':r', label='done')
    plt.plot(t_max, -xd_max * np.ones_like(t_max), ':r')

    for path in paths:
        H = len(path['rewards'])
        t = np.arange(H)
        plt.plot(t, path['observations'][:, 1], color='C0')
    plt.xlabel('Time t [-]')
    plt.ylabel('Cart speed [m/s]')

    # Pole position
    th_max = env.max_pole_angle
    th_done = env.reset_range * th_max
    plt.subplot(3, 2, 3)

    plt.vlines(0, -th_done, th_done, 'r', label='reset range', linewidths=20)
    plt.plot(t_max, th_max * np.ones_like(t_max), ':r', label='done')
    plt.plot(t_max, -th_max * np.ones_like(t_max), ':r')

    for path in paths:
        H = len(path['rewards'])
        t = np.arange(H)
        plt.plot(t, path['observations'][:, 2], color='C0')
    plt.xlabel('Time t [-]')
    plt.ylabel('Pole position [rad]')

    # Pole speed
    thd_max = env.max_pole_speed
    thd_done = env.reset_range * thd_max
    plt.subplot(3, 2, 4)

    plt.vlines(0, -thd_done, thd_done, 'r', label='reset range', linewidths=20)
    plt.plot(t_max, thd_max * np.ones_like(t_max), ':r', label='done')
    plt.plot(t_max, -thd_max * np.ones_like(t_max), ':r')

    for path in paths:
        H = len(path['rewards'])
        t = np.arange(H)
        plt.plot(t, path['observations'][:, 3], color='C0')
    plt.xlabel('Time t [-]')
    plt.ylabel('Pole speed [rad/s]')

    # Input
    ax = plt.subplot(3, 2, 5)
    for path in paths:
        H = len(path['rewards'])
        t = np.arange(H)
        plt.plot(t, path['actions'], color='C0')
    plt.xlabel('Time t [-]')
    plt.ylabel('Input u [-]')

    plt.tight_layout()

    path_info_str = get_paths_info(paths)
    path_info_str = 'Paths info:' + path_info_str
    plt.text(1.2, 1.3, path_info_str, fontsize=8, transform=ax.transAxes, verticalalignment='top')


def create_dated_directory(path):
    assert(os.path.exists(path))

    date_str = time.strftime('%y%m%d')
    time_str = time.strftime('%H%M')
    run = 0

    dir_path = os.path.join(path, date_str, time_str, 'run_%d' % run)
    path_exists = True

    while path_exists is True:
        if os.path.exists(dir_path):
            path_exists = True
            run += 1
            dir_path = os.path.join(path, date_str, time_str, 'run_%d' % run)
        else:
            os.makedirs(dir_path)
            path_exists = False

    return dir_path
