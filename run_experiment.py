"""Experiment harness: runs a driver and tees ALL of its output to a log file.

    python run_experiment.py taipei
    python run_experiment.py electricity
    python run_experiment.py large_st
    python run_experiment.py scada

Everything the driver prints (stdout + stderr, unbuffered) is echoed to the
console AND written to logs/<experiment>_<timestamp>.txt, with a provenance
header (start time, python version, OMP setting) and a footer (exit code,
wall time). Per-trial hypothesis lines are part of the driver output, so a
killed run's progress is recoverable from its log.
"""

import datetime
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = {
    "taipei": "taipei_test.py",
    "electricity": "electricity_load_diagrams_test.py",
    "large_st": "large_st_test.py",
    "scada": "scada_test.py"
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in EXPERIMENTS:
        print(f"usage: python run_experiment.py [{'|'.join(EXPERIMENTS)}]")
        return 2

    name = sys.argv[1]
    script = os.path.join(REPO_ROOT, "mplot_python", EXPERIMENTS[name])
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(REPO_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}_{ts}.txt")

    t0 = time.time()
    # cwd must be the repo root: the electricity driver builds sys.path from
    # os.getcwd(). -u keeps output unbuffered so the log is live.
    proc = subprocess.Popen(
        [sys.executable, "-u", script],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    with open(log_path, "w", encoding="utf-8") as f:
        header = (
            f"# experiment: {name}\n"
            f"# script: {script}\n"
            f"# started: {datetime.datetime.now().isoformat()}\n"
            f"# python: {sys.version.split()[0]}\n"
            f"# OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', '(unset)')}\n"
            + "#" * 60 + "\n"
        )
        print(header, end="")
        f.write(header)
        f.flush()
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        code = proc.wait()
        footer = (
            "#" * 60 + "\n"
            f"# exit code: {code}\n"
            f"# wall time: {(time.time() - t0)/60:.1f} min\n"
            f"# finished: {datetime.datetime.now().isoformat()}\n"
        )
        print(footer, end="")
        f.write(footer)

    print(f"\nlog written: {log_path}")
    return code


if __name__ == "__main__":
    sys.exit(main())
