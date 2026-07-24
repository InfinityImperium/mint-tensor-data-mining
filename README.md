# MINT: Tensor Decomposition on Stacked Recurrence Plots for Time Series Data Mining
**Kaamil Kaka, Audrey Der, Evangelos Papalexakis, Zachary Zimmerman, Vikram Jayaram**

Code for the MINT paper: matrix-summary recurrence plots (Mplots via
[pyscamp](https://github.com/zpzim/SCAMP)), per-series nonnegative robust PCA
(Stable PCP objective), and CPD/PARAFAC over the stacked low-rank tensor,
together with the planted-corruption validation experiments reported in the
paper.

## Repository layout

- `mplot_python/MINT.py` — core pipeline: Mplots, RPCA solver, CPD rank
  selection, `processAll`.
- `mplot_python/co_clustering_trial.py` — planted ground truth (theta
  transformations) and the four hypothesis criteria.
- `mplot_python/co_clustering_experiment.py` /
  `co_clustering_experiment_parallel.py` — repeated-trial drivers (serial and
  trial-parallel; the parallel driver seeds each trial independently and logs
  every trial to `trial_logs/`).
- `mplot_python/{taipei,electricity_load_diagrams,large_st,scada}_test.py` —
  per-dataset experiment harnesses (seed 158).
- `mplot_python/combined_simple_experiment.py` — synthetic MINT-vs-NMF
  comparison figures.
- `run_experiment.py` — runs a harness and tees all output to
  `logs/<experiment>_<timestamp>.txt`.

## Environment

Docker (environment only; the repo is volume-mounted):

```bash
docker build -t mint-env .
docker run --rm -it -v "$PWD":/work mint-env
```

Or a plain virtual environment:

```bash
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Datasets

Place under `datasets_and_dataloaders/` (all are gitignored):

| dataset | source | expected path |
|---|---|---|
| Taipei MRT | https://sites.google.com/view/gbatch | `mrt_data/` |
| Electricity Load Diagrams 2011--2014 | https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014 | `electricityloaddiagrams20112014/LD2011_2014.txt` |
| LargeST | https://github.com/liuxu77/LargeST | `LargeST-main/data/ca/largest/ca_his_raw_2019.h5` |
| Wind Farm A (CARE to Compare) | https://www.kaggle.com/datasets/azizkasimov/wind-turbine-scada-data-for-early-fault-detection | `scada/Wind Farm A/datasets/comma_0.csv` (auto-downloaded via `kagglehub` on first run) |
| OPSD (case-study notebook only) | https://data.open-power-system-data.org/time_series/ | `opsd-time_series-2020-10-06/` |

## Running the experiments

```bash
python run_experiment.py taipei
python run_experiment.py electricity
python run_experiment.py large_st
python run_experiment.py scada
```

Each harness runs 50 independent trials (seed 158; trial *t* is seeded
`158 + t`, so any single trial is reproducible in isolation). Aggregate
hypothesis frequencies are printed at the end of the tee log in `logs/`;
per-trial logs are written to `trial_logs/`. Trials whose corrupted series
produce undefined (NaN) Mplot cells abort deterministically and are retried
on a fresh deterministic seed.

Run logs and per-trial logs from the experiments reported in the paper are
available from the authors on request.

## Citation

Citation information will be added upon publication.
