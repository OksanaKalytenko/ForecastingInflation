#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 20:22:50 2024

@author: oksanakalytenko
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from FredMD import FredMD

#Define the target name.
target_name = "CPIAUCSL"
#Define the target transformation code.
target_tcode = 5 #Should be 5 
#Define the start date of the forecasts.
start_date = pd.Timestamp("2010-01-01")
#Define the start date of the forecasts.
end_date = pd.Timestamp("2024-01-01")

date = "2024-02"

# Instantiate the FredMD object
fred = FredMD(vintage = date)
#change transform code for target var
fred.transforms[target_name] = target_tcode

transforms = fred.transforms

fred.estimate_factors()

data = fred.series_filled

factors = fred.factors

#start with 1970 year
data = data[data.index >= pd.Timestamp("1970-01-01")]
#drop colums due to many missing values
columns_to_drop = ['ACOGNO', 'ANDENOx', 'TWEXAFEGSMTHx', 'UMCSENTx', 'VIXCLSx']

final_data = data.drop(columns=columns_to_drop)

final_data[target_name]=final_data[target_name]*100

final_data.to_csv('data/fred_md_data.csv', index=True) 
