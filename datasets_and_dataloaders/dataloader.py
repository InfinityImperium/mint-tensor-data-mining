import pandas as pd
from pandas import read_csv, DataFrame
import numpy as np
from scipy.io import loadmat
import datetime
from datetime import datetime
import calendar
import os

def load_taipeiMRT(data_path="./mrt_data"):
    """
    Loads the Taipei MRT data from: https://sites.google.com/view/gbatch
    """
    # The following is specific to how the file system hierarchy
    stationnames_path = os.path.join(data_path, "station_name_en.txt")
    stationnames = np.squeeze(pd.read_csv(stationnames_path, dtype=str, encoding='utf-8-sig', header=None).to_numpy())
    stations_enter = [station+" enter" for station in stationnames] 
    stations_exit = [station+" exit" for station in stationnames] 
    columns_all = np.concatenate((np.array(['yyyymm','day','hour']),np.array(stations_enter),np.array(stations_exit)))
    all_data_path = os.path.join(data_path, "all_data.mat")
    d = loadmat(all_data_path)
    d = d['data']
    # Parse time columns
    yyyymm = d[:, 0].astype(int)
    year = yyyymm // 100
    month = yyyymm % 100
    day = d[:, 1].astype(int)
    hour = d[:, 2].astype(int)

    # Convert to datetime
    timestamps = pd.to_datetime({
        'year': year,
        'month': month,
        'day': day,
        'hour': hour
    })

    timestamps += pd.to_timedelta((hour < 5).astype(int), unit="D")

    #anomalies
    timestamps.loc[1284] = timestamps.loc[1284].replace(minute = 30, hour = 4, day = 1, month = 1, year = 2016)
    for i in range(8971, 8975):
        timestamps.loc[i] = timestamps.loc[i].replace(day = 1, month = 1, year = 2017)

    # Save as Series
    timestamps = pd.Series(timestamps, name='timestamps')
    pd.set_option('display.max_rows', 100)
    print(timestamps.head(100))
    d = d.transpose()
    allstationdata = pd.DataFrame(d,columns_all) # load all station data into dataframes
    allstationdata = allstationdata.T
    allstationdata['timestamps'] = timestamps


    return allstationdata

def load_scada(data_path=None, farm="Wind Farm A", file_name="comma_0.csv", local_cache_dir=None):
    """
    Loads wind-turbine SCADA data from the Kaggle dataset
    azizkasimov/wind-turbine-scada-data-for-early-fault-detection
    (CARE benchmark farms).

    Priority order:
    1. If data_path is provided, load from that path directly
    2. If the requested file is in the local cache, load from there
    3. Otherwise, download via kagglehub and copy ONLY the requested file
       into the local cache (the full extracted dataset is ~37 GB; one
       farm/turbine CSV is ~35 MB)

    local_cache_dir defaults to a 'scada' folder next to this module, so it
    is independent of the caller's working directory.

    Returns (df, col_list): df is the raw frame with NaNs filled with 0;
    col_list is the aggregate sensor channels only (drops the max/min/std
    sub-channels and the time_stamp/id/train metadata columns).
    """
    if local_cache_dir is None:
        local_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scada")
    cached_csv = os.path.join(local_cache_dir, farm, "datasets", file_name)

    # Check if explicit data_path is provided
    if data_path is not None:
        csv_path = os.path.join(data_path, farm, "datasets", file_name)
        if os.path.exists(csv_path):
            print(f"Loading from provided path: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"File not found at provided path: {csv_path}")

    # Check if the requested file is already cached locally
    elif os.path.exists(cached_csv):
        print(f"Loading from local cache: {cached_csv}")
        df = pd.read_csv(cached_csv)

    # Download via kagglehub and cache only the requested file
    else:
        print(f"Downloading dataset via kagglehub...")
        import kagglehub
        download_path = kagglehub.dataset_download(
            "azizkasimov/wind-turbine-scada-data-for-early-fault-detection")
        print(f"Downloaded to: {download_path}")

        src_csv = os.path.join(download_path, farm, "datasets", file_name)
        if not os.path.exists(src_csv):
            raise FileNotFoundError(f"File not found in downloaded dataset: {src_csv}")

        import shutil
        print(f"Caching {file_name} to: {cached_csv}")
        os.makedirs(os.path.dirname(cached_csv), exist_ok=True)
        shutil.copy2(src_csv, cached_csv)

        df = pd.read_csv(cached_csv)

    col_list = [column for column in df.columns
                if "max" not in column and "min" not in column
                and "std" not in column and "time_stamp" not in column
                and "id" not in column and "train" not in column]

    df = df.fillna(0)

    return df, col_list


def load_opsd_data(data_path= ".opsd-time_series-2020-10-06"):
    dataset = None
    print("Looking for files in:", data_path)
    for (root,dirs,files) in os.walk(data_path):
        print(files)
        for f in files:
            if "60min_singleindex.csv" in f:
                dataset = read_csv(os.path.join(data_path, f))
                break
    print("Loaded OPSD dataset with shape:", dataset.shape)

    fullcountrylist = []
    for k in dataset.keys():
        if "load_actual_entsoe_transparency" in k:
            if "50hertz" not in k:
                fullcountrylist.append(k)


    l = ['Austria', 'Cyprus', 'Germany', 'Denmark', 'Estonia', 'Spain', 'Great Britain', 'United Kingdom', 'Greece', "Croatia",  'Hungary', 'Italy','Lithuania','Latvia', 'Norway', 'Portugal', 'Sweden', 'Slovakia']

    col_names = ['AT_load_actual_entsoe_transparency','CY_load_actual_entsoe_transparency',  'DE_load_actual_entsoe_transparency', 'DK_load_actual_entsoe_transparency', 'EE_load_actual_entsoe_transparency', 'ES_load_actual_entsoe_transparency', 'GB_GBN_load_actual_entsoe_transparency', 'GB_UKM_load_actual_entsoe_transparency', 'GR_load_actual_entsoe_transparency','HR_load_actual_entsoe_transparency', 'HU_load_actual_entsoe_transparency', 'IT_load_actual_entsoe_transparency', 'LT_load_actual_entsoe_transparency', 'LV_load_actual_entsoe_transparency',  'NO_load_actual_entsoe_transparency', 'PT_load_actual_entsoe_transparency', 'SE_load_actual_entsoe_transparency', 'SK_load_actual_entsoe_transparency']

    eighteen_countries = DataFrame()
    eighteen_countries['utc_timestamp'] = dataset['utc_timestamp']
    
    
    for lk_idx in range(len(col_names)):
        lk = col_names[lk_idx]
        ts = dataset[lk]
        ts = np.ma.masked_where(ts>200000,ts)
        eighteen_countries[l[lk_idx]] = ts
    
    return eighteen_countries


def load_large_st_data(year=2019, data_path='./LargeST-main/data/ca/largest/'):
    """
    Loads the LargeST dataset (California traffic flow data) and forces 32-bit types.
    If the processed HDF file already exists, loads it directly.
    """
    year = str(year)
    
    processed_file = os.path.join(data_path, 'ca_his_' + year + '.h5')
    
    # Check if processed file already exists
    if os.path.exists(processed_file):
        print(f"Loading existing processed file: {processed_file}")
        df = pd.read_hdf(processed_file)
        return df
    
    # If not, process the raw file
    print(f"Processing raw data from {data_path}")
    
    ca_his = pd.read_hdf(os.path.join(data_path, f'ca_his_raw_{year}.h5'))
    
    ca_his = ca_his.astype(np.float32)

    ca_his = (
    ca_his
    .groupby(pd.Grouper(freq="15min"))
    .mean()
    .astype(np.float32)
    )
    
    ca_his = ca_his.fillna(0)
    
    ca_his = ca_his.astype(np.float32)

    ca_his.to_hdf(processed_file, key='t', mode='w')

    df = pd.read_hdf(processed_file)
    
    return df


def load_electricity_data(data_path = "./electricityloaddiagrams20112014/LD2011_2014.txt"):
    df = pd.read_csv(data_path, sep = ";")
    df.rename(columns = {"Unnamed: 0": "Index"}, inplace = True)
    
    # Parse the index as datetime to filter by year
    df['timestamps'] = pd.to_datetime(df['Index'], format='%Y-%m-%d %H:%M:%S')
    df.set_index("timestamps", inplace=True)
    
    # Restrict to calendar year 2013: the full 2011-2014 record makes the
    # pooled matrix infeasible at the uniform resolution rule (C = 3N/m), and
    # a late single year maximizes clients with complete records.
    df = df[(df.index.year == 2013)]
    
    # Reset index but keep timestamps as a column
    df = df.reset_index()
    df = df.drop(columns=['Index'])
    
    df = df.assign(
    **{
        col: pd.to_numeric(
            df[col].astype(str).str.replace(",", ".", regex=False),
            errors="coerce"
        ).div(4)
        for col in df.columns if col != "timestamps"
    }
    )   
    
    return df

    
    
