#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle


target_name = "CPIAUCSL"
nwindows = 156

dt = pd.read_csv("data/fredmd_data.csv", index_col=0)
dt.index = pd.to_datetime(dt.index)
dt.index.freq = 'MS'


y = dt[target_name]

roll_prod = lambda x, n: x.rolling(n).apply(np.prod) - 1

y = pd.concat([y,
               roll_prod(1+y, 3),
               roll_prod(1+y, 6),
               roll_prod(1+y, 12)], axis=1)
y.columns = ['y', 'roll_prod_3', 'roll_prod_6', 'roll_prod_12']

yout = y.tail(nwindows)
rw = np.full((nwindows, 12), np.nan)

for i in range(12):
   
    aux = dt[(len(dt) - nwindows - i-1):(len(dt) - i-1)]['CPIAUCSL']
    rw[:, i] = aux

rw3 = y['roll_prod_3'].shift(3).tail(nwindows)
rw6 = y['roll_prod_6'].shift(6).tail(nwindows)
rw12 = y['roll_prod_12'].shift(12).tail(nwindows)
rw = np.column_stack((rw, rw3, rw6, rw12))

colnames = [f"t+{i}" for i in range(1, 13)] + ["acc3", "acc6", "acc12"]
rw_df = pd.DataFrame(rw, columns=colnames)


with open(f"forecasts/yout_{nwindows}.pkl", "wb") as f:
    pickle.dump(yout, f)

with open(f"forecasts/rw_{nwindows}.pkl", "wb") as f:
    pickle.dump(rw_df, f)
    
yout.to_csv(f"forecasts/yout_{nwindows}.csv", index=False)
rw_df.to_csv(f"forecasts/rw_{nwindows}.csv", index=False)


# Create a new DataFrame containing only the index
index_df = pd.DataFrame(yout.index, columns=['Dates'])
# Save the index DataFrame to a CSV file
index_df.to_csv(f"forecasts/yout_index{nwindows}.csv", index=False)
