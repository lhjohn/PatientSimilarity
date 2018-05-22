#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manifold learning of non-linear data relationships
This file contains functions manifold learn the non-linear relationships in
the covariate data.

Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements. See the NOTICE file distributed with this
work for additional information regarding copyright ownership. The ASF
licenses this file to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
License for the specific language governing permissions and limitations
under the License."""
# --------------------------------------------------------------------------- #
#                  MODULE HISTORY                                             #
# --------------------------------------------------------------------------- #
# Version          1
# Date             2018-04-11
# Author           LH John
# Note             Original version
#
# --------------------------------------------------------------------------- #
#                  SYSTEM IMPORTS                                             #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  OTHER IMPORTS                                              #
# --------------------------------------------------------------------------- #
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
# from ggplot import *

# --------------------------------------------------------------------------- #
#                  OWN IMPORTS                                                #
# --------------------------------------------------------------------------- #
import load

# --------------------------------------------------------------------------- #
#                  META DATA                                                  #
# --------------------------------------------------------------------------- #
__author__ = 'LH John'
__copyright__ = 'Copyright 2018 The Apache Software Foundation'
__credits__ = ['LH John']
__license__ = 'Apache License, Version 2.0'
__version__ = '1'
__maintainer__ = 'LH John'
__email__ = 'l.john@erasmusmc.nl'
__status__ = 'Development'


# --------------------------------------------------------------------------- #
#                  CONSTANTS                                                  #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  GLOBAL VARIABLES                                           #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  EXPORTED FUNCTIONS                                         #
# --------------------------------------------------------------------------- #
def plot_sparsity(sparse, patient_subset=None, paper=False):
    non_zero_count = sparse.count_nonzero()
    dense = sparse.toarray()
    n_cov = dense.shape[1]
    n_pat = dense.shape[0]

    sparsity = non_zero_count/(n_cov*n_pat)*100

    if patient_subset is not None:
        subset = patient_subset
    else:
        subset = n_cov

    plt.imshow(sparse.toarray()[0:subset, 0:n_cov], cmap="Greys",
               interpolation="nearest")

    if not paper:
        plt.suptitle("Sparsity")
        plt.xlabel("Covariates")
        plt.ylabel("Patient subset")
    else:
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.tick_params(axis='both', which='minor', labelsize=22)
        plt.xlabel("Covariates", fontsize=22)
        plt.ylabel("Patient subset", fontsize=22)

    # plt.title("Non-zero percentage: {:5.3}%".format(sparsity))

    plt.show()


def plot_svd_redundancy(svd):
    plt.plot(np.cumsum(svd.explained_variance_ratio_))
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Redundancy")
    plt.suptitle("Singular-Value Decomposition")
    print(plt.show())

#
# def plot_manifold(df):
#     a_plot = ggplot(aes(x="D1", y="D2", color="outcome"), data=df) +\
#         geom_point(alpha=0.9, size=5) + \
#         scale_color_manual(values=["orange", "purple"]) +\
#         labs(title='Manifold') +\
#         theme_bw()
#     print(a_plot)
#
#
# def plot_cluster(df):
#     a_plot = ggplot(aes(x="D1", y="D2", color="Cluster"), data=df) +\
#         geom_point(alpha=0.9, size=5) + \
#         scale_color_brewer(type="qual", palette="Set1") +\
#         labs(title='Cluster') +\
#         theme_bw()
#     print(a_plot)
# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
