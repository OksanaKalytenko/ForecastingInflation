#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:23:15 2024

@author: oksanakalytenko
"""

## This is a quick code for visualising the results

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_prep import dataprep

from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error

# Define the directory containing the forecast files
forecast_dir = "forecasts/"

nwindows = 156
rolling_window_size = 36  # Size of the rolling window

# Load the specific forecast files
yout = pd.read_csv(os.path.join(forecast_dir, f"yout_{nwindows}.csv"))
rw = pd.read_csv(os.path.join(forecast_dir, f"rw_{nwindows}.csv"))
date = pd.read_csv(os.path.join(forecast_dir, f"yout_index{nwindows}.csv"))
date['Dates'] = pd.to_datetime(date['Dates'])


# List all model files except for "rw.csv" and "yout.csv"

model_files = {f"AR_{nwindows}.csv", f"RF_{nwindows}.csv", f"rw_{nwindows}.csv"}
# Load each model file and store the data in a dictionary
models_list = {}
for model_file in model_files:
    model_path = os.path.join(forecast_dir, model_file)
    models_list[model_file] = pd.read_csv(model_path)
    
    

# Calculate errors for each model: RMSE between the first 12 columns of each model and the first column of yout
errors_rmse= {}
errors_mae = {}
errors_mad = {}
for model_file, model in models_list.items():
    error_rmse = np.array([np.sqrt(mean_squared_error(yout.iloc[:, 0], model.iloc[:, i])) for i in range(12)])
    errors_rmse[model_file] = error_rmse
    
    error_mae = np.array([mean_absolute_error(yout.iloc[:, 0], model.iloc[:, i]) for i in range(12)])
    errors_mae[model_file] = error_mae
    
    error_mad = np.array([np.median(np.abs((yout.iloc[:, 0] - model.iloc[:, i])) - np.median((yout.iloc[:, 0] - model.iloc[:, i]))) for i in range(12)])
    errors_mad[model_file] = error_mad

print("########## RMSE ##############")    
errors_rmse_df = pd.DataFrame(errors_rmse)
errors_rmse_df = errors_rmse_df.rename(columns={f"RF_{nwindows}.csv": "RF", f"AR_{nwindows}.csv": "AR", f"rw_{nwindows}.csv":"RW"})

errors_rmse_df.index = errors_rmse_df.index + 1  # Increment the existing index by 1
errors_rmse_df.index.name = 'horizon'

print(errors_rmse_df[["RW", "AR", "RF"]])

print("########## MAE ##############  ")
errors_mae_df = pd.DataFrame(errors_mae)
errors_mae_df = errors_mae_df.rename(columns={f"RF_{nwindows}.csv": "RF", f"AR_{nwindows}.csv": "AR", f"rw_{nwindows}.csv":"RW"})

errors_mae_df.index = errors_mae_df.index + 1  # Increment the existing index by 1
errors_mae_df.index.name = 'horizon'

print(errors_mae_df[["RW", "AR", "RF"]])

print("########## MAD ##############  ")
errors_mad_df = pd.DataFrame(errors_mad)
errors_mad_df = errors_mad_df.rename(columns={f"RF_{nwindows}.csv": "RF", f"AR_{nwindows}.csv": "AR", f"rw_{nwindows}.csv":"RW"})

errors_mad_df.index = errors_mad_df.index + 1  # Increment the existing index by 1
errors_mad_df.index.name = 'horizon'

print(errors_mad_df[["RW", "AR", "RF"]])



###################################################################################

# Function to calculate RMSE for rolling windows

def rolling_metrics(y_true, y_pred, window):
    rmse_list = []
    mae_list = []
    mad_list = []
    for end in range(window, len(y_true)):
        start = end - window
        
        rmse = np.sqrt(mean_squared_error(y_true[start:end], y_pred[start:end]))
        
        mae = mean_absolute_error(y_true[start:end], y_pred[start:end])
        
        errors = y_true[start:end] - y_pred[start:end]
        deviations = np.abs(errors - np.median(errors))
        mad = np.median(deviations)

        rmse_list.append(rmse)
        mae_list.append(mae)
        mad_list.append(mad)
        
    return rmse_list, mae_list, mad_list

# Calculate rolling metrics for each model
rolling_errors_rmse = {}
rolling_errors_mae = {}
rolling_errors_mad = {}

for model_file, model in models_list.items():
    model_rmse, model_mae, model_mad = zip(*[rolling_metrics(yout.iloc[:, 0], model.iloc[:, i], rolling_window_size) for i in range(12)])
    rolling_errors_rmse[model_file] = np.array(model_rmse)
    rolling_errors_mae[model_file] = np.array(model_mae)
    rolling_errors_mad[model_file] = np.array(model_mad)

    
    
    

#################### PLOT #######################

####rmse

# Define new plot styles
colors = {
    "RW": "#0C00EB",  # Blue
    "RF": "#00A35D",  # Green
    "AR": "#DC00EB",  # Purple
}

# Define the horizons to plot
horizons = [0, 2, 5, 8, 11]  # Horizons 1, 6, and 12 (Python index starts at 0)

# Loop over each horizon to create separate plots
for h in horizons:
    plt.figure(figsize=(18, 12))
    
    # Plot for RW model
    plt.plot(
        date["Dates"][rolling_window_size:], 
        rolling_errors_rmse['rw_156.csv'][h], 
        label="RW", 
        color=colors['RW'],  
        linestyle='-', 
        linewidth=2, 
        markersize=8
    )
    
    # Plot for RF model
    plt.plot(
        date["Dates"][rolling_window_size:], 
        rolling_errors_rmse['RF_156.csv'][h], 
        label="RF", 
        color=colors['RF'], 
        linestyle='--', 
        linewidth=2, 
        markersize=8
    )
    
    # Plot for AR model
    plt.plot(
        date["Dates"][rolling_window_size:], 
        rolling_errors_rmse['AR_156.csv'][h], 
        label="AR", 
        color=colors['AR'], 
        linestyle='-.', 
        linewidth=2, 
        markersize=8
    )
    
    # Add titles and labels
    plt.title(f'Rolling RMSE - h={h+1}', fontsize=24)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('RMSE', fontsize=20)
    
    # Customize ticks and grid
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.grid(True)
    
    # Add legend
    plt.legend(fontsize=16)
    
    # Display the plot
    plt.show()


###### mae

# Define new plot styles
colors = {
    "RW": "#0C00EB",  # Blue
    "RF": "#00A35D",  # Green
    "AR": "#DC00EB",  # Purple
}

# Define the horizons to plot
horizons = [0, 2, 5, 8, 11]  # Horizons 1, 6, and 12 (Python index starts at 0)

# Loop over each horizon to create separate plots
for h in horizons:
    plt.figure(figsize=(18, 12))
    
    # Plot for RW model
    plt.plot(
        date["Dates"][rolling_window_size:], 
        rolling_errors_mae['rw_156.csv'][h], 
        label="RW", 
        color=colors['RW'],  
        linestyle='-', 
        linewidth=2, 
        markersize=8
    )
    
    # Plot for RF model
    plt.plot(
        date["Dates"][rolling_window_size:], 
        rolling_errors_mae['RF_156.csv'][h], 
        label="RF", 
        color=colors['RF'], 
        linestyle='--', 
        linewidth=2, 
        markersize=8
    )
    
    # Plot for AR model
    plt.plot(
        date["Dates"][rolling_window_size:], 
        rolling_errors_mae['AR_156.csv'][h], 
        label="AR", 
        color=colors['AR'], 
        linestyle='-.', 
        linewidth=2, 
        markersize=8
    )
    
    # Add titles and labels
    plt.title(f'Rolling MAE - h={h+1}', fontsize=24)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('MAE', fontsize=20)
    
    # Customize ticks and grid
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.grid(True)
    
    # Add legend
    plt.legend(fontsize=16)
    
    # Display the plot
    plt.show()


#####mad

# Define new plot styles
colors = {
    "RW": "#0C00EB",  # Blue
    "RF": "#00A35D",  # Green
    "AR": "#DC00EB",  # Purple
}

# Define the horizons to plot
horizons = [0, 2, 5, 8, 11]  # Horizons 1, 6, and 12 (Python index starts at 0)

# Loop over each horizon to create separate plots
for h in horizons:
    plt.figure(figsize=(18, 12))
    
    # Plot for RW model
    plt.plot(
        date["Dates"][rolling_window_size:], 
        rolling_errors_mad['rw_156.csv'][h], 
        label="RW", 
        color=colors['RW'],  
        linestyle='-', 
        linewidth=2, 
        markersize=8
    )
    
    # Plot for RF model
    plt.plot(
        date["Dates"][rolling_window_size:], 
        rolling_errors_mad['RF_156.csv'][h], 
        label="RF", 
        color=colors['RF'], 
        linestyle='--', 
        linewidth=2, 
        markersize=8
    )
    
    # Plot for AR model
    plt.plot(
        date["Dates"][rolling_window_size:], 
        rolling_errors_mad['AR_156.csv'][h], 
        label="AR", 
        color=colors['AR'], 
        linestyle='-.', 
        linewidth=2, 
        markersize=8
    )
    
    # Add titles and labels
    plt.title(f'Rolling MAD - h={h+1}', fontsize=24)
    plt.xlabel('Date', fontsize=20)
    plt.ylabel('MAD', fontsize=20)
    
    # Customize ticks and grid
    plt.xticks(fontsize=18, rotation=45)
    plt.yticks(fontsize=18)
    plt.grid(True)
    
    # Add legend
    plt.legend(fontsize=16)
    
    # Display the plot
    plt.show()






##### rmse
#plot

plt.figure(figsize=(18, 12))
plt.plot(errors_rmse_df)
# Adding titles and labels 
plt.title('The RMSE over the same forecasting period', fontsize=24)
plt.xlabel('Horizon', fontsize=20)
plt.ylabel('RMSE', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# Adding a grid 
plt.legend(errors_rmse_df.columns, fontsize=16)
plt.grid(True)

errors_rmse_df.drop(columns=["RW"], inplace=True)
plt.figure(figsize=(18, 12))
plt.plot(errors_rmse_df)
# Adding titles and labels 
plt.title('The RMSE over the same forecasting period', fontsize=24)
plt.xlabel('Horizon', fontsize=20)
plt.ylabel('RMSE', fontsize=20)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# Adding a grid 
plt.legend(errors_rmse_df.columns, fontsize=16)
plt.grid(True)



#### mae
#plot

plt.figure(figsize=(18, 12))
plt.plot(errors_mae_df)
# Adding titles and labels 
plt.title('The MAE over the same forecasting period', fontsize=24)
plt.xlabel('Horizon', fontsize=20)
plt.ylabel('MAE', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# Adding a grid 
plt.legend(errors_mae_df.columns, fontsize=16)
plt.grid(True)

errors_mae_df.drop(columns=["RW"], inplace=True)
plt.figure(figsize=(18, 12))
plt.plot(errors_mae_df)
# Adding titles and labels 
plt.title('The MAE over the same forecasting period', fontsize=24)
plt.xlabel('Horizon', fontsize=20)
plt.ylabel('MAE', fontsize=20)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# Adding a grid 
plt.legend(errors_mae_df.columns, fontsize=16)
plt.grid(True)



##### mad
#plot

plt.figure(figsize=(18, 12))
plt.plot(errors_mad_df)
# Adding titles and labels
plt.title('The MAD over the same forecasting period', fontsize=24)
plt.xlabel('Horizon', fontsize=20)
plt.ylabel('MAD', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# Adding a grid for 
plt.legend(errors_mad_df.columns, fontsize=16)
plt.grid(True)

errors_mad_df.drop(columns=["RW"], inplace=True)
plt.figure(figsize=(18, 12))
plt.plot(errors_mad_df)
# Adding titles and labels 
plt.title('The MAD over the same forecasting period', fontsize=24)
plt.xlabel('Horizon', fontsize=20)
plt.ylabel('MAD', fontsize=20)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# Adding a grid 
plt.legend(errors_mad_df.columns, fontsize=16)
plt.grid(True)



# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(24, 14), sharex=True, sharey=True)
# Plot for AR model
axes[0].plot(date["Dates"], yout["y"].values, label="Actual data", color='#0C00EB')
axes[0].plot(date["Dates"], models_list["AR_156.csv"]["t+1"].values, label="Forecast with AR", color='#EB5500')
axes[0].set_title('Actual Data vs Forecast with AR, for horizon = 1', fontsize=20)
axes[0].set_xlabel('Date', fontsize=16)
axes[0].set_ylabel('Values', fontsize=16)
axes[0].legend(fontsize=16)
axes[0].grid(True)

# Plot for RF model
axes[1].plot(date["Dates"], yout["y"].values, label="Actual data", color='#0C00EB')
axes[1].plot(date["Dates"], models_list["RF_156.csv"]["t+1"].values, label="Forecast with RF", color='#EB5500')
axes[1].set_title('Actual Data vs Forecast with RF, for horizon = 1', fontsize=20)
axes[1].set_xlabel('Date', fontsize=16)
axes[1].set_ylabel('Values', fontsize=16)
axes[1].legend(fontsize=16)
axes[1].grid(True)


