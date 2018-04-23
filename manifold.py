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
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import TSNE

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
def fit_manifold(pc, outcome, technique="isomap", dim=2):
    df = []
    if technique == "isomap":
        df = __manifold_isomap(pc, outcome, dim)
    elif technique == "tsne":
        df = __manifold_tsne(pc, outcome, dim=2, n_iterations=4000)
    elif technique == "lle":
        df = __manifold_lle(pc, outcome, dim)

    df = pd.concat([df, outcome], axis=1)
    return df


# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #
def __manifold_isomap(pc, outcome, dim=2):
    """Fit Isomap.
    :return: DataFrame of covariates"""
    isomap = Isomap(n_components=dim, n_jobs=-1)
    iso_out = isomap.fit_transform(pc)

    df_iso_out = pd.DataFrame(iso_out, columns=["D1", "D2"])
    return df_iso_out


def __manifold_lle(pc, outcome, dim=2):
    """Fit Locally Linear Embedding.
    :return: DataFrame of covariates"""
    lle = LocallyLinearEmbedding(n_components=dim, n_jobs=-1,
                                 method='standard')
    lle_out = lle.fit_transform(pc)

    df_lle_out = pd.DataFrame(lle_out, columns=["D1", "D2"])
    return df_lle_out


def __manifold_tsne(pc, outcome, dim=2, n_iterations=4000):
    """Fit t-distributed Stochastic Neighbor Embedding.
    :return: DataFrame of covariates"""
    lle = TSNE(n_components=dim, verbose=2, n_iter=n_iterations)
    lle_out = lle.fit_transform(pc)

    df_lle_out = pd.DataFrame(lle_out, columns=["D1", "D2"])
    return df_lle_out

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
