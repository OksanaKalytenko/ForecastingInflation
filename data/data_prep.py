#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 12:31:06 2024

@author: oksanakalytenko
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def dataprep(ind, df, variable, horizon, add_dummy=True, univar=False, factonly=False, nofact=False):
    df = df.iloc[ind, :]
    y = df[variable]
    
    if nofact:
        if not univar:
            x = df
        else:
            x = df[[variable]].values
    else:
        if not univar:
            pca = PCA(n_components=4)
            factors = pca.fit_transform(df)
            if factonly:
                x = np.hstack([df[[variable]], factors])
            else:
                x = np.hstack([df, factors])
        else:
            x = df[[variable]].values
    
    X = np.array([x[i:i+4].flatten() for i in range(len(x) - 3)])
    
    Xin = X[:-horizon, :]
    Xout = X[-1, :].reshape(1, -1)
    yin = y.iloc[-len(Xin):]
    
    if "2008-11-01" in yin.index:
        dummy = np.zeros(len(yin))
        intervention = yin.index.get_loc("2008-11-01")
        dummy[intervention] = 1
        if add_dummy:
            Xin = np.hstack([Xin, dummy.reshape(-1, 1)])
            Xout = np.hstack([Xout, [[0]]])
    else:
        dummy = np.zeros(len(yin))
        if add_dummy:
            Xin = np.hstack([Xin, dummy.reshape(-1, 1)])
            Xout = np.hstack([Xout, [[0]]])

    return {"dummy": dummy, "Xin": Xin, "Xout": Xout, "yin": yin}



