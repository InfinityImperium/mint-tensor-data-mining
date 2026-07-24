import os
import sys
import time

# ensure mplot_python package path and parent are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datasets_and_dataloaders.dataloader import load_scada
from mplot_python.co_clustering_experiment_parallel import co_clustering_experiment_parallel
from mplot_python.co_clustering_experiment import co_clustering_experiment
import numpy as np
import pandas as pd
import pyscamp
import random
# Seed: submission number, fixed before the runs.
random.seed(158)
np.random.seed(158)

"""
Wind-turbine SCADA co-clustering test (Wind Farm A, comma_0.csv) from
https://www.kaggle.com/datasets/azizkasimov/wind-turbine-scada-data-for-early-fault-detection
The dataloader downloads via kagglehub on first run (cached afterwards).
"""

if __name__ == '__main__':
    print("initiating scada")

    allsensordata, listOfSensorsRaw = load_scada()

    print(len(allsensordata))
    print(f"{len(listOfSensorsRaw)} sensor channels enter the filter")

    subsequenceLength = 140

    # All aggregate sensor channels enter the NaN filter; exclusions are
    # decided at runtime, at the experiment's pooled resolution (NaN cells
    # are pooling-dependent).
    listOfSensorsFinal = listOfSensorsRaw

    exclude = []
    print("filtering sensors...")
    for idx, sensor_name in enumerate(listOfSensorsFinal):
        series = np.array(allsensordata[sensor_name].values, dtype=np.float32).reshape(-1,)
        if np.isnan(series).any():
            exclude.append(idx)
            print(sensor_name + " excluded due to NaNs in SEQUENCE")
            continue
        mplot = pyscamp.abjoin_matrix(
            series, series, subsequenceLength,
            mheight=1179, mwidth=1179, threshold=-1
        )
        if np.isnan(mplot).any():
            print(sensor_name + " excluded due to NaNs in MATRIX PROFILE")
            exclude.append(idx)

    old_len = len(listOfSensorsFinal)
    listOfSensorsFinal = [col for i, col in enumerate(listOfSensorsFinal) if i not in exclude]
    print(f"Excluded {old_len - len(listOfSensorsFinal)} of {old_len} sensors; {len(listOfSensorsFinal)} remain.")

    t_start = time.time()
    h1, h2, h3, h4, hlast, hall = co_clustering_experiment_parallel(
        allsensordata,
        listOfSensorsFinal,
        subsequenceLength,
        1179,
        39,
        0.25,
        sensor_side_mat="C",
        s=6,
        dataframe_name="Wind Farm A SCADA",
        num_trials=50,
        seed = 158,
        max_workers = 12
    ) 

    t_end = time.time()
    elapsed = t_end - t_start
    print(f"Experiment duration: {elapsed:.2f}s ({elapsed/60:.2f} min)")

    print("FINALS")
    print(f"Hypothesis 1 Frequency: {h1}")
    print(f"Hypothesis 2 Frequency: {h2}")
    print(f"Hypothesis 3 Frequency: {h3}")
    print(f"Hypothesis 4 Frequency: {h4}")
    print(f"Last Three Hypotheses Frequency: {hlast}")
    print(f"All Hypotheses Frequency: {hall}")
