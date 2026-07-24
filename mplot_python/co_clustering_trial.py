import numpy as np
import pandas as pd

import random
import MINT

# Theta transformations: the planted ground truth for the co-clustering trials
def theta_1(series):
    #completely random
    mu = np.mean(series)
    sigma = np.std(series)

    completely_random_series = np.random.normal(mu, sigma, len(series))

    return completely_random_series


def theta_2(series, starts, window_size):
    #mostly random
    mu = np.mean(series)
    sigma= np.std(series)

    mostly_rand_series = np.random.normal(mu, sigma, len(series))
    for start in starts:
        for index in range(start, start + window_size):
            mostly_rand_series[index] = series[index]
    return mostly_rand_series
        

def theta_3(series, starts, window_size):
    #mostly normal
    mu = np.mean(series)
    sigma = np.std(series) 
    
    mostly_normal_series = series.copy()
    rand_background = np.random.normal(mu, sigma, len(series))

    for start in starts:
        for index in range(start, start + window_size):
            mostly_normal_series[index] = rand_background[index]
    return mostly_normal_series



def create_processed_dataframe(dataframe, list_of_sensors, b_prop, num_of_windows, window_size):
    length = len(dataframe)

    b = int(np.ceil(b_prop * len(list_of_sensors)))

    # Windows are drawn from non-overlapping window_size-aligned slots.
    list_of_possible_indices = []
    count = 0
    while(count < length - window_size + 1):
        list_of_possible_indices.append(count)
        count += window_size

    starts = np.random.choice(list_of_possible_indices, size = num_of_windows, replace = False)  

    df_dict = {}

    shuffled_sensors = list_of_sensors.copy()
    np.random.shuffle(shuffled_sensors)

    mostly_random = shuffled_sensors[:b]
    completely_random = shuffled_sensors[b:2*b]
    mostly_normal = shuffled_sensors[2*b:]

    for sensor in completely_random:
        series = dataframe[sensor].values
        series = theta_1(series)
        if (np.isnan(series).all()):
            print(f"WARNING: {sensor} (in completely random) is NAN!")
        df_dict[sensor] = series
        
    for sensor in mostly_random:
        series = dataframe[sensor].values
        series = theta_2(series, starts, window_size)
        df_dict[sensor] = series

    for sensor in mostly_normal:
        series = dataframe[sensor].values
        series = theta_3(series, starts, window_size)
        df_dict[sensor] = series
  
    processed_dataframe = pd.DataFrame(df_dict)

    chosen_intervals = [[start, start + window_size - 1] for start in starts]

    return processed_dataframe, chosen_intervals, completely_random, mostly_random, mostly_normal


def distinctiveness_pipeline(mint_result, s, sensor_side_mat, list_of_sensors, dataframe_length, mplot_side_length,  window_size):
    A,B,C  = mint_result.low_rank_factors
    
    if (sensor_side_mat == "B"):
        mat = B
    else:
        mat = C
    
    component_indices = range(A.shape[1])
    
    top_s_sensors = {i: [] for i in component_indices}
    top_s_intervals = {i: [] for i in component_indices}
    
    
    for i in component_indices:
        
        #top-s sensors
        component = A[:, i]
        component_temp = np.abs(component)
        ind = np.argpartition(component_temp, -s)[-s: ]
        top_s_indices = ind[np.argsort(component_temp[ind])]
        top_s_indices = np.flip(top_s_indices)
        for top_index in top_s_indices:
            top_s_sensors[i].append(list_of_sensors[top_index])


        #top-s subsequences
        component = mat[:, i]
        component_temp = np.abs(component)
        ind = np.argpartition(component_temp, -s)[-s: ]
        top_s_indices = ind[np.argsort(component_temp[ind])]
        top_s_indices = np.flip(top_s_indices)
        
        for index in top_s_indices:
            start = np.max([0,int(np.floor(((index - 1)/mplot_side_length) * dataframe_length))])
            end = np.min([dataframe_length - 1, int(np.ceil(((index + 1)/mplot_side_length) * dataframe_length + window_size))])
            top_s_intervals[i].append([start, end])
            
    return top_s_sensors, top_s_intervals
       

def co_clustering_trial(dataframe, list_of_sensors, subsequence_length, mplot_side_length, num_of_windows, b_prop, sensor_side_mat = "C", s = 3, dataframe_name = "Dataset", trial_num = 0):
    window_size = subsequence_length

    processed_dataframe, chosen_intervals, completely_random, mostly_random, mostly_normal = create_processed_dataframe(dataframe, list_of_sensors, b_prop, num_of_windows,  window_size)
    from MINT import processAll
    result = processAll(list_of_sensors, 
                        processed_dataframe, 
                        subsequence_length, 
                        Mheight=mplot_side_length, 
                        Mwidth=mplot_side_length,
                        name = f"{dataframe_name} All Sensors Co-Clustering {str(subsequence_length)} Trial {trial_num}")   

    dataframe_length = len(dataframe)
    top_s_sensors, top_s_intervals = distinctiveness_pipeline(result, s, sensor_side_mat, list_of_sensors, dataframe_length, mplot_side_length, window_size)
    print(top_s_sensors)
    statistics = {key: None for key in top_s_sensors}
    for component_index in top_s_sensors:
        completely_random_count = 0
        mostly_random_count = 0
        mostly_normal_count = 0
        for sensor in top_s_sensors[component_index]:
            if sensor in completely_random:
                completely_random_count +=1
            elif sensor in mostly_random:
                mostly_random_count += 1
            elif sensor in mostly_normal:
                mostly_normal_count +=1
        statistics[component_index] ={
            "completely_random": completely_random_count,
            "mostly_random": mostly_random_count,
            "mostly_normal": mostly_normal_count
        }
        
        print(f"Component {component_index} statistics: {statistics[component_index]}")
    return top_s_sensors, statistics, chosen_intervals, top_s_intervals, completely_random, mostly_random, mostly_normal


def overlap(interval1, interval2):
    min1 = np.min(interval1)
    min2 = np.min(interval2)
    max1 = np.max(interval1)
    max2 = np.max(interval2)
    
    return (min1 <= max2 and min2 <= max1)


# Existential criterion: passes iff SOME component's top-s intervals overlap
# planted windows at >= hyp_1_threshold.
def hypothesis_1_met(chosen_intervals, top_s_intervals, hyp_1_threshold = 0.5):
    met_hyp_1 = False
    for c in top_s_intervals:
        overlap_count = 0
        for interval1 in top_s_intervals[c]:
            for interval2 in chosen_intervals:
                if (overlap(interval1, interval2)):
                    overlap_count += 1
                    break
        print(c, overlap_count)
        if (overlap_count / len(top_s_intervals[c]) >= hyp_1_threshold):
            met_hyp_1 = True
    
    return met_hyp_1

def hypothesis_2_met(top_s_sensors, statistics, completely_random, s , hyp_2_threshold = 0.5):
    for component_index in statistics:
        completely_random_count = statistics[component_index]["completely_random"]
        mostly_random_count = statistics[component_index]["mostly_random"]
        mostly_normal_count = statistics[component_index]["mostly_normal"]
        total_count =  completely_random_count + mostly_random_count + mostly_normal_count
        print(f"Completely Random Count: {completely_random_count}")
        print(f"Mostly Random Count: {mostly_random_count}")
        if (completely_random_count / total_count <= hyp_2_threshold and mostly_random_count / total_count <= hyp_2_threshold and mostly_normal_count / total_count <= hyp_2_threshold):
            return False
    return True


def hypothesis_3_met(statistics):
    violating_components = 0
    for component_index in statistics:
        completely_random_count = statistics[component_index]["completely_random"]
        if (completely_random_count >= 1):
            violating_components +=1
    return violating_components <= 1


def hypothesis_4_met(statistics, hyp_4_threshold_in = 0.5, hyp_4_threshold_out = 0.70):
   

    mostly_normal_component_count = 0
    for component_index in statistics:
        completely_random_count = statistics[component_index]["completely_random"]
        mostly_random_count = statistics[component_index]["mostly_random"]
        mostly_normal_count = statistics[component_index]["mostly_normal"]
        total_count =  completely_random_count + mostly_random_count + mostly_normal_count
        if (mostly_normal_count / total_count > hyp_4_threshold_in):
            mostly_normal_component_count += 1
            
    return mostly_normal_component_count >= np.floor(hyp_4_threshold_out * len(statistics))

