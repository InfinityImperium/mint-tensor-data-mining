import sys
import os
from typing import Tuple
import random

# ensure mplot_python package path is added
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from co_clustering_trial import (co_clustering_trial, hypothesis_1_met, 
                         hypothesis_2_met, hypothesis_3_met, hypothesis_4_met)
from MINT import processAll  # required by co_clustering_experiment

def co_clustering_experiment(dataframe, list_of_sensors, subsequence_length, mplot_side_length, num_of_windows, b_prop,
               sensor_side_mat: str = "C", s: int = 3, dataframe_name: str = "Dataset",
               num_trials: int = 1,  hyp_1_threshold: float = 1.0,
               hyp_2_threshold: float = 0.5, hyp_4_threshold_in: float = 0.5, hyp_4_threshold_out: float = 0.70,
               diff_sample = False, sample_size = 100) -> Tuple[float, float, float, float, float, float]:
    """Run repeated co-clustering trials and report the frequency each hypothesis is met.

    Returns proportions across num_trials:
        (h1, h2, h3, h4, hLast3, hAll)
    """
    hyp_1_count = 0
    hyp_2_count = 0
    hyp_3_count = 0
    hyp_4_count = 0
    all_hyp_count = 0
    last_three_hyp_count = 0


    trial_count = 0
    iterator = 0

    while trial_count < num_trials and iterator < num_trials * 10:
        print(f"Trial {trial_count}")
        try:
            if (diff_sample):
                list_of_sensors_sample = random.sample(list_of_sensors, sample_size)
                top_s_sensors, statistics, chosen_intervals, top_s_intervals, completely_random, mostly_random, mostly_normal = co_clustering_trial(
                    dataframe,
                    list_of_sensors_sample,
                    subsequence_length,
                    mplot_side_length,
                    num_of_windows,
                    b_prop,
                    sensor_side_mat,
                    s,
                    dataframe_name=f"{dataframe_name}",
                    trial_num = trial_count
                )

            else:
                top_s_sensors, statistics, chosen_intervals, top_s_intervals, completely_random, mostly_random, mostly_normal = co_clustering_trial(
                    dataframe,
                    list_of_sensors,
                    subsequence_length,
                    mplot_side_length,
                    num_of_windows,
                    b_prop,
                    sensor_side_mat,
                    s,
                    dataframe_name=f"{dataframe_name}",
                    trial_num = trial_count
                )

            hyp_1 = hypothesis_1_met(chosen_intervals, top_s_intervals, hyp_1_threshold)
            hyp_2 = hypothesis_2_met(top_s_sensors, statistics, completely_random, s, hyp_2_threshold)
            hyp_3 = hypothesis_3_met(statistics)
            hyp_4 = hypothesis_4_met(statistics, hyp_4_threshold_in, hyp_4_threshold_out)

            print(f"Hypothesis 1: {hyp_1}")
            print(f"Hypothesis 2: {hyp_2}")
            print(f"Hypothesis 3: {hyp_3}")
            print(f"Hypothesis 4: {hyp_4}")

            if hyp_1:
                hyp_1_count += 1
            if hyp_2:
                hyp_2_count += 1
            if hyp_3:
                hyp_3_count += 1
            if hyp_4:
                hyp_4_count += 1
            if hyp_2 and hyp_3 and hyp_4:
                last_three_hyp_count += 1
            if hyp_1 and hyp_2 and hyp_3 and hyp_4:
                all_hyp_count += 1

            # Cumulative counts: progress is recoverable from the log if a
            # run dies mid-way.
            print(f"Cumulative after trial {trial_count}: "
                  f"H1={hyp_1_count} H2={hyp_2_count} H3={hyp_3_count} H4={hyp_4_count} "
                  f"last3={last_three_hyp_count} all={all_hyp_count}")

            trial_count +=1
        except Exception as e:
            print(f"Trial {trial_count} failed with exception: {e}")
            continue
        finally:
            iterator += 1
    
    if iterator >= num_trials * 10:
        print(f"Experiment terminated after {iterator} iterations due to excessive failures.")
    
         


       




        

    return (
        hyp_1_count / num_trials,
        hyp_2_count / num_trials,
        hyp_3_count / num_trials,
        hyp_4_count / num_trials,
        last_three_hyp_count / num_trials,
        all_hyp_count / num_trials,
    )


__all__ = ["co_clustering_experiment"]
