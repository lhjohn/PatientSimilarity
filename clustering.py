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
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

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
def fit_cluster(manifold_df, n_clusters, technique="kmeans"):
    df = []
    if technique == "kmeans":
        df = __cluster_kmeans(manifold_df, n_clusters)
    elif technique == "agglomerative":
        df = __cluster_agglomerative(manifold_df, n_clusters)

    return df


# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #
def __cluster_kmeans(manifold_df, n_clusters):
    """Fit Kmeans.
    :return: DataFrame of covariates"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_out = kmeans.fit_predict(manifold_df.loc[:, ["D1", "D2"]])

    kmeans_out = pd.concat([manifold_df,
                            pd.DataFrame(kmeans_out, columns=["Cluster"],
                                         dtype="category")], axis=1)
    return kmeans_out


def __cluster_agglomerative(manifold_df, n_clusters):
    """Fit Agglomerative.
    :return: DataFrame of covariates"""
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative_out = agglomerative.fit_predict(
        manifold_df.loc[:, ["D1", "D2"]])

    agglomerative_out = pd.concat([manifold_df,
                                   pd.DataFrame(
                                       agglomerative_out,
                                       columns=["Cluster"],
                                       dtype="category")],
                                  axis=1)
    return agglomerative_out

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
