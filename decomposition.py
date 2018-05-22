#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Linear decomposition of covariate data
This file contains functions to linearly decompose the covariate data. This
way, redundancy can be assessed.

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
from sklearn.decomposition import TruncatedSVD

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
svd = []

# --------------------------------------------------------------------------- #
#                  EXPORTED FUNCTIONS                                         #
# --------------------------------------------------------------------------- #
def decompose_svd(sparse, n_pc=0):
    """Load covariate data file.
    :return: DataFrame of covariates"""
    global svd

    if n_pc == 0:
        n_pc = sparse.toarray().shape[1] - 1

    svd = TruncatedSVD(n_components=n_pc, )
    svd.fit(sparse)
    pc = svd.transform(sparse)
    return pc


def get_svd():
    return svd

# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
