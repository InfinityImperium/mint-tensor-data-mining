import os
import sys
import time

# ensure mplot_python package path and parent are in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datasets_and_dataloaders.dataloader import load_taipeiMRT
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
To run this test, download the Taipei MRT dataset folder from 
https://sites.google.com/view/gbatch
and place the 'mrt_data' folder as a top-level subfolder in 
datasets_and_dataloaders. Then run this script.
"""

if __name__ == '__main__':
    print("initiating taipei")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'datasets_and_dataloaders', 'mrt_data')

    allstationdata = load_taipeiMRT(data_path=data_path)

    print(len(allstationdata))
     
    timestamps = allstationdata["timestamps"]
    subsequenceLength = 146

    listOfStationsRaw= []

    with open(os.path.join(data_path, 'station_name_en.txt'), 'r') as listFile:
        lines = listFile.readlines()
        for line in lines:
            line = line.strip()
            listOfStationsRaw.append(line + " enter")

    listOfStationsRaw[0] = "Songshan Airport enter"

    # All "enter" stations enter the NaN filter; exclusions are decided at
    # runtime, at the experiment's pooled resolution (NaN cells are
    # pooling-dependent).
    listOfStationsFinal = listOfStationsRaw

    print(len(listOfStationsFinal))
    print(len(listOfStationsRaw))

    exclude = []
    print("filtering sensors...")
    for idx, sensor_name in enumerate(listOfStationsFinal):
        series = np.array(allstationdata[sensor_name].values, dtype=np.float32).reshape(-1,)
        if np.isnan(series).any():
            exclude.append(idx)
            print(sensor_name + " excluded due to NaNs in SEQUENCE")
            continue
        mplot = pyscamp.abjoin_matrix(
            series, series, subsequenceLength,
            mheight=224, mwidth=224, threshold=-1
        )
        if np.isnan(mplot).any():
            print(sensor_name + " excluded due to NaNs in MATRIX PROFILE")
            exclude.append(idx)

    old_len = len(listOfStationsFinal)
    listOfStationsFinal = [col for i, col in enumerate(listOfStationsFinal) if i not in exclude]
    print(f"Excluded {old_len - len(listOfStationsFinal)} of {old_len} stations; {len(listOfStationsFinal)} remain.")

    ts = pd.to_datetime(timestamps)
    x = np.where(ts.diff().dt.total_seconds() < 0)[0]
    for i in x:
        print(i, str(timestamps[i - 2]),
               "->", str(timestamps[i - 1]),
               "->", str(timestamps[i]) ,
               "->", str(timestamps[i + 1]),
               "->", str(timestamps[i + 2]))


    t_start = time.time()
    h1, h2, h3, h4, hlast, hall = co_clustering_experiment_parallel(
        allstationdata,
        listOfStationsFinal,
        subsequenceLength,
        224,
        7,
        0.25,
        sensor_side_mat="C",
        s=6,
        dataframe_name="All Stations Taipei",
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
