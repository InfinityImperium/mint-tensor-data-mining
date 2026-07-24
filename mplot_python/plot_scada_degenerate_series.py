"""Sparkline-style plot of the four kept (non-excluded) SCADA channels whose
traces go degenerate (flat) after the labeled failure event ends (row 54448
onward): sensor_48 flatlines immediately, sensor_5 (pitch angle) pins at ~90
degrees, sensor_15/16 (stator temperatures) decay to ambient and freeze.

Produces two versions: with and without time stamps.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.dirname(current_dir))

import matplotlib.pyplot as plt
from datasets_and_dataloaders.dataloader import load_scada

SENSORS = ["sensor_48", "sensor_5_avg", "sensor_15_avg", "sensor_16_avg"]
COLORS = ["#1f77b4", "#d62728", "#ff7f0e", "#e69fd8"]
START = 54000  # plot the final stretch; the shutdown flat regions begin at 54448

df, _ = load_scada()
df = df.iloc[START:]

for with_timestamps in (True, False):
    fig, axes = plt.subplots(len(SENSORS), 1, figsize=(8, 3), sharex=True)
    for ax, name, color in zip(axes, SENSORS, COLORS):
        ax.plot(df[name].values, color=color, linewidth=0.5)
        # reserve headroom above the trace so the label never intersects it
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + 0.25 * (y1 - y0))
        label_size = 8 if not with_timestamps else 4
        ax.text(0.005, 0.97, name, transform=ax.transAxes,
                color=color, fontsize=label_size, va="top")
        ax.set_axis_off()
    if with_timestamps:
        ax = axes[-1]
        ax.set_axis_on()
        ax.get_yaxis().set_visible(False)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ticks = range(0, len(df), 144)  # every day
        ax.set_xticks(list(ticks))
        ax.set_xticklabels([df["time_stamp"].iloc[t][:10] for t in ticks],
                           fontsize=4)
    fig.tight_layout(h_pad=0.3)
    suffix = "timestamps" if with_timestamps else "plain"
    out = os.path.join(current_dir, "figures", f"scada_degenerate_{suffix}")
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}.png|pdf")
