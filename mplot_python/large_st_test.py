import os
import sys
import numpy as np
import pandas as pd
import pyscamp

import random


# Fresh seed for camera-ready reruns, fixed BEFORE the runs (submission number).
random.seed(158)
np.random.seed(158)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datasets_and_dataloaders.dataloader import load_large_st_data
from mplot_python.co_clustering_experiment_parallel import co_clustering_experiment_parallel
from mplot_python.co_clustering_experiment import co_clustering_experiment


"""
To run this test, follow the instructions in https://github.com/liuxu77/LargeST
and place the LargeST-main folder as a top-level subfolder in datasets_and_dataloaders.
ca_his_raw_2019.h5 should be under datasets_and_dataloaders/LargeST-main/data/ca/largest.
Then run this script.
"""

if __name__ == '__main__':

    print("initiating Large ST")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'datasets_and_dataloaders', 'LargeST-main', 'data', 'ca', 'largest')
    subsequenceLength = 95
    
    df = load_large_st_data(2019, data_path)
    print(df)
    
    col_list = df.columns.tolist()
    col_list = col_list
    print(len(col_list))
    exclude = []

    print("filtering sensors...")
    # Exclude invalid or NaN-filled sensors
    for idx, sensor_name in enumerate(col_list):
        series = df[sensor_name].to_numpy().astype(np.float32)
        if np.isnan(series).any():
            exclude.append(idx)
            print(sensor_name + " excluded due to NaNs in SEQUENCE")
            continue
        # Same pooled resolution as the experiment (NaN cells are pooling-dependent).
        mplot = pyscamp.abjoin_matrix(
            series, series, subsequenceLength,
            mheight=1107, mwidth=1107, threshold=-1
        )
        if np.isnan(mplot).any():
            print(sensor_name + " excluded due to NaNs in MATRIX PROFILE")
            exclude.append(idx)

    print("exclusion complere.")
    print(f"Excluding {len(exclude)} sensors out of {len(col_list)} total sensors.")
    print("Generating final column list...")


    col_list = [col for i, col in enumerate(col_list) if i not in exclude]

    # Remove known Mplot NaN sensors
    if ("1123069" in col_list):
        col_list.remove("1123069") 
    if ("1116593" in col_list):
        col_list.remove("1116593")
    if ("1116594" in col_list):
        col_list.remove("1116594")

   
    # mplot side 1107 = 3N/m (uniform rule: cell width = m/3)
    h1, h2, h3, h4, hlast, hall = co_clustering_experiment_parallel(
        df,
        col_list,
        subsequenceLength,
        1107,
        36,
        0.25,
        sensor_side_mat="C",
        s=6,
        dataframe_name="LargeST Full",
        num_trials=50,
        diff_sample = True,
        sample_size =100,
        seed = 158,
        max_workers = 12
    )

    print("FINALS")
    print(f"Hypothesis 1 Frequency: {h1}")
    print(f"Hypothesis 2 Frequency: {h2}")
    print(f"Hypothesis 3 Frequency: {h3}")
    print(f"Hypothesis 4 Frequency: {h4}")
    print(f"Last Three Hypotheses Frequency: {hlast}")
    print(f"All Hypotheses Frequency: {hall}")
    
 
   
    
    
    
    



