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


class FirstOrderOptimizer:
    """
    First order stochastic gradient descend style optimizer with momentum.
    """

    def __init__(self, fun_and_jac, ndim,
                 step_size_init=3e-2, beta=0.75,
                 maxiter=1000,
                 window=9, f_std_tol=1e-3):

        self.fun_and_jac = fun_and_jac
        self.ndim = ndim

        # Optimizer parameters
        self.step_size_init = step_size_init
        self.beta = beta

        # Termination settings
        self.window = window
        self.f_std_tol = f_std_tol
        self.maxiter = maxiter

        self.reset()

    def reset(self):
        self.m = np.zeros((self.ndim, ))

    def optimize(self, x, **kwargs):
        """
        Optimize the target x for a maximum number of steps or until convergence.
        """

        if 'step_size' in kwargs:
            step_size = kwargs['step_size']
        else:
            step_size = self.step_size_init
        step_size = np.clip(step_size, 1e-5, 3e-2)

        beta = self.beta

        fs = []

        g_norm = np.ones((self.ndim, ))

        for i in range(self.maxiter):
            f, g = self.fun_and_jac(x)

            fs.append(f)

            norm = np.linalg.norm(g)
            if norm > 0:
                g_norm = g / norm

            if i > self.window+1 and np.std(fs[-self.window:]) < self.f_std_tol:
                break

            self.m = beta * self.m + (1 - beta) * g_norm

            x = x + step_size * self.m

        return x, f, np.asarray(fs)
