#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Sun Jun 23 17:16:50 2024

@author: oksanakalytenko

"""
import numpy as np
import pandas as pd
from data_prep import dataprep
from rolling_window import *
import matplotlib.pyplot as plt
import pickle
import time

from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
from functools import partial

target_name = "CPIAUCSL"

model_function = runar
model_name = "AR"

# Run rolling window
nwindows = 156

nmodels = 12

def create_shared_memory_from_dataframe(df):
    """Converts a DataFrame to a shared memory array."""
    array = df.to_numpy()
    flat_array = array.flatten()
    shm = SharedMemory(create=True, size=flat_array.nbytes)
    shared_array = np.ndarray(flat_array.shape, dtype=flat_array.dtype, buffer=shm.buf)
    shared_array[:] = flat_array[:]
    return shm, array.shape, df.columns, df.index

def read_shared_memory_to_dataframe(shm_name, shape, columns, index):
    """Reads shared memory and converts it back to a DataFrame."""
    existing_shm = SharedMemory(name=shm_name)
    flat_array = np.ndarray(shape=np.prod(shape), dtype=np.float64, buffer=existing_shm.buf)
    array = flat_array.reshape(shape)
    df = pd.DataFrame(array, columns=columns, index=index)
    return df, existing_shm

def run_model_index(i, shm_name, shape, columns, index):
    print(f"Starting model {i}")
    """Worker function that reads shared memory and processes a row."""
    df, shm = read_shared_memory_to_dataframe(shm_name, shape, columns, index)
    
    start_time = time.time()

    result = rolling_window(model_function, df, nwindows + i - 1, i, "CPIAUCSL")
    shm.close()
    
    # Record the end time
    end_time = time.time()
    
    # Calculate the duration
    duration = end_time - start_time
    
    print(f"Training of one horizon model ({i}) took {duration} seconds.")
    return result

def main():
    dt = pd.read_csv("data/fredmd_data.csv", index_col=0)
    dt.index = pd.to_datetime(dt.index)
    dt.index.freq = 'MS'

    model_list = None
    
    shm, shape, columns, index = create_shared_memory_from_dataframe(dt)
    
    try:
        shm_name = shm.name
        with Pool(processes=6) as p:
            print("Starting processes...")
            start_time = time.time()
            
            partial_model_index = partial(run_model_index, shm_name=shm_name, shape=shape, columns=columns, index=index)
            model_list = p.map(partial_model_index, list(range(1, (nmodels + 1))))
            
            # Record the end time
            end_time = time.time()

            # Calculate the duration
            duration = end_time - start_time

            print(f"Training all models took {duration} seconds.")
    finally:
        # Clean up shared memory
        shm.close()
        shm.unlink()

    forecasts = np.column_stack([model['forecast'][:nwindows] for model in model_list])

    forecasts = accumulate_model(forecasts)

    # Save forecasts to file
    with open(f"forecasts/{model_name}_{nwindows}.pkl", "wb") as f:
        pickle.dump(forecasts, f)

    forecasts.to_csv(f"forecasts/{model_name}_{nwindows}.csv", index=False)

    # Plot the data
    plt.figure(figsize=(16, 10))
    plt.plot(dt["CPIAUCSL"].tail(nwindows).values, label="Actual CPIAUCSL", color='firebrick')
    plt.plot(forecasts["t+1"].values, label=f"Forecast with {model_name}", color='mediumblue')
    plt.legend()
    plt.savefig(f"forecasts/{model_name}_{nwindows}_figure.pdf")


    # Print summary of data
    print(dt.describe())
    
if __name__ == "__main__":
    main()
    
