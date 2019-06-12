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


class SoftmaxMemorySelection:
    """
    Select a subset of all available paths according to a softmax distribution
    over their corresponding path returns. Up to max_paths paths are selected
    where a number of last n_hist paths can optionally be added in every case.
    """

    def __init__(self, max_paths, softmax_temp, n_hist):
        """
        Construct softmax memory selection object.

        args:
            max_paths, int, number of paths to select from all available
            softmax_temp, float, softmax temperature
            n_hist, int, number of last (historic) paths to select
        """

        self.max_paths = max_paths
        self.softmax_temp = softmax_temp
        self.n_hist = n_hist

    def select_paths_subset(self, paths, Rs, return_indices=False):
        """
        Select a subset of paths (or their indices).

        args:
            paths, list, all available paths
            Rs, list, all corresponding path returns
            return_indices, bool, whether to return sublist or selected indices
        returns:
            list of path indices or list of paths
        """

        max_paths = self.max_paths
        softmax_temp = self.softmax_temp
        n_hist = self.n_hist

        n_paths = len(paths)

        if n_paths > max_paths:
            # Select last n_hist rollouts
            hist_ids = np.arange(n_paths - n_hist, n_paths)

            # Randomly select according to softmax return
            available_ids = np.arange(n_paths - n_hist)

            logits = np.asarray(Rs)[available_ids]
            logits -= logits.min()
            logits /= logits.max()

            softmax = np.exp((logits - logits.max()) / softmax_temp)
            softmax = softmax / np.sum(softmax)

            # Check how many non zero prob. samples are available
            n_softmax = np.count_nonzero(softmax > 0)
            n_softmax = min(max_paths - n_hist, n_softmax)

            softmax_ids = np.random.choice(available_ids,
                                           n_softmax,
                                           replace=False,
                                           p=softmax)

            # Add additional random paths
            if n_hist + n_softmax < max_paths:
                random_ids = np.random.choice(len(paths),
                                              max_paths - n_hist - n_softmax,
                                              replace=False)
                ids = np.concatenate((hist_ids, softmax_ids, random_ids))
            else:
                ids = np.concatenate((hist_ids, softmax_ids))
        else:
            ids = np.arange(n_paths)

        if return_indices:
            return ids
        else:
            return [paths[id] for id in ids]
