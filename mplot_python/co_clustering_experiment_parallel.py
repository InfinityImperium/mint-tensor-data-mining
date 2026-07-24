"""Trial-parallel version of co_clustering_experiment.

Each worker process runs one full, unmodified co_clustering_trial (single
threaded: MINT pins OMP_NUM_THREADS=1) plus the hypothesis evaluations, so
throughput scales with worker count while per-trial results stay identical to
what a serial process would produce under the same seed.

Differences from the serial co_clustering_experiment, both deliberate:
  * Seeding: trial t is seeded with (seed + t) for both `random` and
    `np.random`, instead of all trials sharing one sequential stream. This
    makes every trial reproducible in isolation (re-running trial 7 alone
    gives the same planted ground truth) and independent of worker
    scheduling. It does mean trial draws differ from a serial run of the
    old harness -- statistically equivalent, but not draw-for-draw.
  * Output: each trial's prints are captured under its own folder,
    trial_logs/<name>/trial_<t>/attempt_seed<seed>.log, instead of
    interleaving on stdout. Retry attempts get separate files (named by
    their seed) so failed-attempt logs are preserved, and per-trial
    artifacts have a natural home next to their log.

Attempt k (k=0 is the first try) of trial t is seeded seed + num_trials*k + t,
which is collision-free across trials and attempts (t < num_trials) and keeps
retry seeding deterministic regardless of completion order. Total attempts are
capped at 10 * num_trials, mirroring the serial harness's retry loop.

Choose max_workers by RAM: each C~1100 trial peaks around 2-2.5 GB, so
~16-20 workers on a 64 GB VM; Taipei-sized trials are ~tens of MB, so
worker count can equal core count.
"""
import os
import re
import contextlib
import random as _random_mod
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import numpy as np

# Worker globals, populated once per worker by _init_worker.
_DF = None
_SENSORS = None


def _init_worker(df, sensors, log_dir):
    global _DF, _SENSORS
    _DF = df
    _SENSORS = sensors
    os.makedirs(log_dir, exist_ok=True)


def _run_one_trial(args):
    (trial_num, seed, subsequence_length, mplot_side_length, num_of_windows,
     b_prop, sensor_side_mat, s, dataframe_name, hyp_1_threshold,
     hyp_2_threshold, hyp_4_threshold_in, hyp_4_threshold_out,
     diff_sample, sample_size, log_dir) = args

    import random
    from co_clustering_trial import (co_clustering_trial, hypothesis_1_met,
                                     hypothesis_2_met, hypothesis_3_met,
                                     hypothesis_4_met)

    random.seed(seed)
    np.random.seed(seed)

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", dataframe_name)
    trial_dir = os.path.join(log_dir, safe_name, f"trial_{trial_num:02d}")
    os.makedirs(trial_dir, exist_ok=True)
    log_path = os.path.join(trial_dir, f"attempt_seed{seed}.log")
    try:
        with open(log_path, "w") as lf, contextlib.redirect_stdout(lf):
            print(f"trial={trial_num} seed={seed}")
            sensors = _SENSORS
            if diff_sample:
                sensors = random.sample(_SENSORS, sample_size)

            (top_s_sensors, statistics, chosen_intervals, top_s_intervals,
             completely_random, mostly_random, mostly_normal) = co_clustering_trial(
                _DF, sensors, subsequence_length, mplot_side_length,
                num_of_windows, b_prop, sensor_side_mat, s,
                dataframe_name=dataframe_name, trial_num=trial_num)

            h1 = hypothesis_1_met(chosen_intervals, top_s_intervals, hyp_1_threshold)
            h2 = hypothesis_2_met(top_s_sensors, statistics, completely_random,
                                  s, hyp_2_threshold)
            h3 = hypothesis_3_met(statistics)
            h4 = hypothesis_4_met(statistics, hyp_4_threshold_in, hyp_4_threshold_out)
            print(f"RESULT h1={h1} h2={h2} h3={h3} h4={h4}")
        return (trial_num, seed, True,
                (bool(h1), bool(h2), bool(h3), bool(h4)),
                None)
    except Exception as e:
        return (trial_num, seed, False, None, f"{type(e).__name__}: {e}")


def co_clustering_experiment_parallel(
        dataframe, list_of_sensors, subsequence_length, mplot_side_length,
        num_of_windows, b_prop, sensor_side_mat: str = "C", s: int = 3,
        dataframe_name: str = "Dataset", num_trials: int = 1,
        hyp_1_threshold: float = 1.0, hyp_2_threshold: float = 0.5,
        hyp_4_threshold_in: float = 0.5,
        hyp_4_threshold_out: float = 0.70,
        diff_sample=False, sample_size=100,
        seed: int = 158, max_workers: int = None, log_dir: str = "trial_logs",
) -> Tuple[float, float, float, float, float, float]:
    """Drop-in parallel replacement for co_clustering_experiment.

    Same return tuple:
        (h1, h2, h3, h4, hLast3, hAll)
    """
    if max_workers is None:
        max_workers = max(2, (os.cpu_count() or 4) - 2)

    def make_args(trial_num, trial_seed):
        return (trial_num, trial_seed, subsequence_length, mplot_side_length,
                num_of_windows, b_prop, sensor_side_mat, s, dataframe_name,
                hyp_1_threshold, hyp_2_threshold, hyp_4_threshold_in,
                hyp_4_threshold_out, diff_sample, sample_size, log_dir)

    results = {}          # trial_num -> (h1, h2, h3, h4)
    attempts = 0
    max_attempts = num_trials * 10
    attempt_of = {t: 0 for t in range(num_trials)}  # attempts issued per trial

    def seed_for(trial_num, k):
        # Attempt k of trial t: collision-free since t < num_trials.
        return seed + num_trials * k + trial_num

    with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(dataframe, list_of_sensors, log_dir)) as ex:

        pending = {ex.submit(_run_one_trial, make_args(t, seed_for(t, 0))): t
                   for t in range(num_trials)}
        attempts += num_trials

        while pending:
            for fut in as_completed(list(pending)):
                trial_num = pending.pop(fut)
                tnum, tseed, ok, verdicts, err = fut.result()
                if ok:
                    results[tnum] = verdicts
                    h1, h2, h3, h4 = verdicts
                    print(f"[trial {tnum} seed {tseed}] "
                          f"H1={h1} H2={h2} H3={h3} H4={h4}  "
                          f"({len(results)}/{num_trials} done)", flush=True)
                elif attempts < max_attempts:
                    attempt_of[trial_num] += 1
                    retry_seed = seed_for(trial_num, attempt_of[trial_num])
                    print(f"[trial {trial_num} seed {tseed}] FAILED: {err} "
                          f"-- retrying with seed {retry_seed}", flush=True)
                    pending[ex.submit(_run_one_trial,
                                      make_args(trial_num, retry_seed))] = trial_num
                    attempts += 1
                else:
                    print(f"[trial {trial_num}] FAILED: {err} -- retry budget "
                          f"exhausted ({attempts} attempts)", flush=True)

    n_ok = len(results)
    if n_ok < num_trials:
        print(f"WARNING: only {n_ok}/{num_trials} trials completed; "
              f"frequencies below divide by num_trials as in the serial harness.")

    c = {"h1": 0, "h2": 0, "h3": 0, "h4": 0, "last3": 0, "all": 0}
    for h1, h2, h3, h4 in results.values():
        c["h1"] += h1; c["h2"] += h2; c["h3"] += h3; c["h4"] += h4
        last3 = h2 and h3 and h4
        c["last3"] += last3
        c["all"] += (h1 and last3)

    return (c["h1"] / num_trials, c["h2"] / num_trials,
            c["h3"] / num_trials, c["h4"] / num_trials,
            c["last3"] / num_trials, c["all"] / num_trials)


__all__ = ["co_clustering_experiment_parallel"]
