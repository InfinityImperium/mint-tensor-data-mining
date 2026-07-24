import os
import sys
import pyscamp

# ensure mplot_python package path and parent are in sys.path
current_dir = os.getcwd()
sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import numpy as np
import random
import pandas as pd
# Seed: submission number, fixed before the runs.
random.seed(158)
np.random.seed(158)

from mplot_python.co_clustering_experiment import co_clustering_experiment
from mplot_python.co_clustering_experiment_parallel import co_clustering_experiment_parallel
from datasets_and_dataloaders.dataloader import load_electricity_data


"""
To run this test, download the Electricity Load Diagrams 2011-2014 dataset folder from 
https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
and place the 'electricityloaddiagrams20112014' folder as a top-level subfolder in 
datasets_and_dataloaders. Then run this script.
"""

if __name__ == '__main__':
    print("initiating electricity")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'datasets_and_dataloaders', 'electricityloaddiagrams20112014', 'LD2011_2014.txt')
    # load_electricity_data performs the comma-decimal conversion and the /4
    # scaling itself; do NOT re-apply either here.
    df = load_electricity_data(data_path)

    print(df.head(10))

    subsequenceLength = 95

    col_list = [c for c in df.columns if c != "timestamps"]

    exclude = []

    print(len(df))


    print("filtering sensors...")
    for idx, sensor_name in enumerate(col_list):

        print("Processing sensor:", sensor_name)
        series = df[sensor_name].to_numpy().astype(np.float32)




        if np.isnan(series).any():
            exclude.append(idx)
            print(sensor_name + " excluded due to NaNs in SEQUENCE")
            continue

        # Filter at the same pooled resolution the experiment uses: NaN cells
        # are pooling-dependent (a cell's NaN can be overwritten by a valid
        # pair), so filtering at a different mheight/mwidth than the
        # experiment can under- or over-exclude.
        mplot = pyscamp.abjoin_matrix(
            series, series, subsequenceLength,
            mheight=1107, mwidth=1107, threshold=-1
        )

        if np.isnan(mplot).any():
            print(sensor_name + " excluded due to NaNs in MATRIX PROFILE")
            exclude.append(idx)

    old_len = len(col_list)
    col_list = [col for i, col in enumerate(col_list) if i not in exclude]
    print(f"Excluded {old_len - len(col_list)} of {old_len} clients; {len(col_list)} remain.")
    print(f"col_list: {col_list}")
    h1, h2, h3, h4, hlast, hall = co_clustering_experiment_parallel(
        df,
        col_list,
        subsequenceLength,
        1107,
        36,
        0.25,
        sensor_side_mat="C",
        s=6,
        dataframe_name="Electricity Load Diagrams 2011-2014",
        num_trials=50,
        diff_sample = True,
        sample_size = 100,
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


    

