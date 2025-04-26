#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 13:31:14 2024

@author: oksanakalytenko
"""
import numpy as np
import pandas as pd
from data_prep import dataprep

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant



def rolling_window(fn, df, nwindow=1, horizon=None, variable=None, **kwargs):
    # Indices for the dataframe
    ind = np.arange(len(df))
    
    # Size of the window
    window_size = len(df) - nwindow
    
    # Create an index matrix
    indmat = np.full((window_size, nwindow), np.nan)
    
    indmat[0, :] = np.arange(nwindow)    
    
    # Fill the index matrix
    for i in range(1, len(indmat)):
        indmat[i, :] = indmat[i - 1, :] + 1
        
    indmat = indmat.T

    # Apply the function to each rolling window
    rw = [fn(indmat[i, :].astype(int), df=df, horizon=horizon, variable=variable, **kwargs) for i in range(len(indmat))]
    # Extract forecasts and outputs
    forecast = [x['forecast'] for x in rw]
    outputs = [x['outputs'] for x in rw]
    
    print("FINISHED")
    
    return {'forecast': forecast, 'outputs': outputs}


def runar(ind, df, variable, horizon, type="fixed"):
    prep_data = dataprep(ind, df, variable, horizon, univar=True, add_dummy=False)
    Xin = prep_data['Xin']
    yin = prep_data['yin']
    Xout = prep_data['Xout']
    dummy = prep_data['dummy']

    if type == "fixed":
        X = np.hstack((Xin, dummy.reshape(-1, 1)))
        X = add_constant(X)
        modelest = OLS(yin, X).fit()
        best = Xin.shape[1]
        
    elif type == "bic":
        bb = float('inf')
        best = 1
        best_model = None
        
        for i in range(1, Xin.shape[1] + 1):
            X = np.hstack((Xin[:, :i], dummy.reshape(-1, 1)))
            X = add_constant(X)
            model = OLS(yin, X).fit()
            crit = model.bic

            if crit < bb:
                bb = crit
                best_model = model
                best = i

        modelest = best_model

    coef = modelest.params
    coef = np.nan_to_num(coef)
    forecast = np.dot(np.hstack(([1], Xout[:, :best].flatten(), [0])), coef)
    outputs = []

    return {'forecast': forecast, 'outputs' : outputs}


def runrf(ind, df, variable, horizon):
    prep_data = dataprep(ind, df, variable, horizon)
    Xin = prep_data['Xin']
    yin = prep_data['yin']
    Xout = prep_data['Xout']
    
    print(f"Processing index range: {min(ind)} {max(ind)}...")
    
    # default python params
    #rf_model = RandomForestRegressor()
    #set as 
    n_features = Xin.shape[1]  
    max_feat = max(n_features // 3, 1)
    rf_model = RandomForestRegressor(n_estimators=500,
                                     min_samples_leaf=1,
                                     min_samples_split=5,
                                     max_depth=10,
                                     max_features = max_feat,
                                     bootstrap=True,
                                     n_jobs=-1,
                                     random_state=42
                                     )
    
    rf_model.fit(Xin, yin)
    forecast = rf_model.predict(Xout)
    
    # Get feature importance directly from the trained model
    importance = rf_model.feature_importances_
    
    outputs = {'importance': importance}
    
    return {'forecast': forecast, 'outputs': outputs}

def accumulate_model(forecasts):
    # Number of rows in forecasts
    n_rows = forecasts.shape[0]
    
    # Initialize accumulated arrays with NaNs
    acc3 = np.full(n_rows, np.nan)
    acc6 = np.full(n_rows, np.nan)
    acc12 = np.full(n_rows, np.nan)
    
    # Compute acc3
    for i in range(2, n_rows):
        acc3[i] = np.prod(1 + np.diag(forecasts[i-2:i+1, :3])) - 1
    
    # Compute acc6
    for i in range(5, n_rows):
        acc6[i] = np.prod(1 + np.diag(forecasts[i-5:i+1, :6])) - 1
    
    # Compute acc12
    for i in range(11, n_rows):
        acc12[i] = np.prod(1 + np.diag(forecasts[i-11:i+1, :12])) - 1
    
    # Combine forecasts with accumulated values
    forecasts = np.column_stack((forecasts, acc3, acc6, acc12))
    
    # Update column names (assuming the forecasts variable is a DataFrame)
    column_names = [f"t+{i+1}" for i in range(12)] + ["acc3", "acc6", "acc12"]
    
    # Return forecasts as DataFrame with updated column names
    return pd.DataFrame(forecasts, columns=column_names)

