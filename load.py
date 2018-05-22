#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Loading and pre-processing covariates
This file contains functions to load and pre-process the data files used for
exploratory data analysis.

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
from scipy import sparse as sp
import numpy as np
# --------------------------------------------------------------------------- #
#                  OWN IMPORTS                                                #
# --------------------------------------------------------------------------- #

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
def load_covariates(filename):
    """Load covariate data file.
    :return: DataFrame of covariates"""
    covariates = pd.read_csv(filename, dtype={"rowId": int, "covariateId": int, "covariateValue": int})
    sparse = sp.coo_matrix((covariates.iloc[:, 2].values, (covariates.iloc[:, 0].values, covariates.iloc[:, 1].values)))
    return sparse


def load_outcome(filename):
    """Load outcome data file.
    :return: DataFrame of outcomes"""
    df = pd.read_csv(filename)
    df = df['outcomeCount'].astype(bool)
    return df


# --------------------------------------------------------------------------- #
#                  LOCAL FUNCTIONS                                            #
# --------------------------------------------------------------------------- #
def __private_function_example():
    """This is a private function example, which is indicated by the leading
    under scores.
    :return: tmp_bool"""
    tmp_bool = True
    return tmp_bool

# --------------------------------------------------------------------------- #
#                  END OF FILE                                                #
# --------------------------------------------------------------------------- #
