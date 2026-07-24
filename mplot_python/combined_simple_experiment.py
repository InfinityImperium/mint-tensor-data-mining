"""
Combined MINT + NMF simple experiment pipeline.

Generates synthetic time series data once and applies both:
1. MINT: MPlot + RPCA + CPD (3-way tensor factorization)
2. NMF: Direct 2-way matrix factorization

Preserves original implementations as closely as possible.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from scipy import signal
import random
import torch
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.decomposition import NMF
from kneed import KneeLocator

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pyscamp
from tqdm import tqdm
from MINT import nnrobustpca_stable_pcp, find_optimal_components_parafac, plotMatrixRaw

# ══════════════════════════════════════════════════════════════════════════════
# SEEDS — identical to both original experiments
# ══════════════════════════════════════════════════════════════════════════════
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("GENERATING SYNTHETIC DATA")
print("=" * 80)

amplitude = 10
x = np.linspace(0, 8760, 8761)
sines = amplitude * (1.1 + np.sin((2 * np.pi / 24) * x))

y = np.linspace(0, 24, 25)
sawtooth = amplitude * (1.1 + signal.sawtooth((2 * np.pi / 24) * y))
square   = amplitude * (1.1 + signal.square((2 * np.pi / 24) * y))

multi_timeseries = {}
group_labels = {}       # series_name -> group index (0=T1, 1=T2, 2=T3)
first_imputation = {}   # series_name -> sample index of first placed sawtooth

# Generate 60 series for MINT (but use first 50 for NMF to match original)
for i in range(0, 60):
    series = sines.copy()
    wave_to_use = sawtooth

    if i < 20:
        idx_counter = 0
        group = 0          # T1: phase offset  0 days
    elif i < 40:
        idx_counter = 24 * 10
        group = 1          # T2: phase offset 10 days
    else:
        idx_counter = 24 * 20
        group = 2          # T3: phase offset 20 days

    first_placed = None
    while idx_counter < len(sines):
        this_time = random.random()
        if this_time < 0.9:
            to_place = min(
                max(0, idx_counter + random.randint(-24, 24)),
                len(sines) - 25
            )
            series[to_place : to_place + 25] = wave_to_use
            if first_placed is None:
                first_placed = to_place
        idx_counter += 24 * 30

    series += np.random.normal(0, amplitude / 15, size=len(sines))
    multi_timeseries[f'series_{i}'] = np.maximum(series, 0)
    group_labels[f'series_{i}'] = group
    first_imputation[f'series_{i}'] = first_placed

keys = list(multi_timeseries.keys())
random.shuffle(keys)
multi_timeseries = {k: multi_timeseries[k] for k in keys}

multi_timeseries_df = pd.DataFrame(multi_timeseries)
series_names = list(multi_timeseries_df.columns)

print(f"Generated {len(series_names)} time series")
print(f"  T1 (0d offset): {sum(1 for s in series_names if group_labels[s] == 0)} series")
print(f"  T2 (10d offset): {sum(1 for s in series_names if group_labels[s] == 1)} series")
print(f"  T3 (20d offset): {sum(1 for s in series_names if group_labels[s] == 2)} series")

# Save sample series plot
n_plot = 24 * 30 * 7
sampled_cols = random.sample(list(multi_timeseries_df.columns), 3)
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
for ax, col in zip(axes, sampled_cols):
    ax.plot(multi_timeseries_df[col].iloc[:n_plot].values)
    ax.set_title(col)
    ax.set_ylabel("Amplitude")
axes[-1].set_xlabel("Sample")
plt.tight_layout()
os.makedirs('figures/combined_synthetic', exist_ok=True)
plt.savefig("figures/combined_synthetic/sample_series.png", dpi=150, bbox_inches='tight')
plt.close()
print("Sample series plot saved")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1: MINT (MPlot + RPCA + CPD)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("METHOD 1: MINT (MPLOT + RPCA + CPD)")
print("=" * 80)

SUBSEQ_LEN  = 23
MHEIGHT     = 365
MWIDTH      = 365
MINT_MPLOTS_FILE = "combined_mint_mplots.npz"
MINT_FACTORS_FILE = "combined_mint_factors.npz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Stage 1: MPlot + RPCA per series ──────────────────────────────────────────
print("\n=== MINT Stage 1: MPlot + RPCA ===")

if os.path.exists(MINT_MPLOTS_FILE):
    saved = np.load(MINT_MPLOTS_FILE)
    low_rank_mplots = {k: saved[k] for k in saved.files}
    print(f"  Loaded {len(low_rank_mplots)}/60 cached MPlots from {MINT_MPLOTS_FILE}")
else:
    low_rank_mplots = {}

pbar = tqdm(series_names, desc="MINT: MPlot + RPCA", unit="series")
for series_name in pbar:
    if series_name in low_rank_mplots:
        pbar.set_postfix_str(f"{series_name}: cached")
        continue

    pbar.set_postfix_str(f"{series_name}: computing...")
    series = np.ascontiguousarray(
        multi_timeseries_df[series_name].values,
        dtype=np.float64
    )

    try:
        mplot = pyscamp.abjoin_matrix(
            np.ascontiguousarray(series),
            np.ascontiguousarray(series),
            SUBSEQ_LEN,
            mheight=MHEIGHT,
            mwidth=MWIDTH,
            threshold=-1
        )
    except ValueError as e:
        tqdm.write(f"  Error computing MPlot for {series_name}: {e}")
        continue

    L, S, num_iter = nnrobustpca_stable_pcp(mplot, max_iter=5000, verbose=False)
    
    # Per-slice mean-centering
    L_np = L.detach().cpu().numpy()
    L_np = L_np - np.mean(L_np)
    
    low_rank_mplots[series_name] = L_np.astype(np.float32)
    np.savez(MINT_MPLOTS_FILE, **low_rank_mplots)
    pbar.set_postfix_str(f"{series_name}: done")

# ── Stage 2: Build tensor ─────────────────────────────────────────────────────
print("\n=== MINT Stage 2: Build tensor ===")

missing = [n for n in series_names if n not in low_rank_mplots]
if missing:
    raise RuntimeError(f"MINT Stage 1 failed for {len(missing)}/60 series")

ordered_mplots = [low_rank_mplots[n] for n in series_names]
tensor_np = np.stack(ordered_mplots, axis=0).astype(np.float32)
tensor_np = tensor_np / (np.linalg.norm(tensor_np) + 1e-12)
print(f"  Tensor shape: {tensor_np.shape}")

# ── Stage 3: PARAFAC ──────────────────────────────────────────────────────────
print("\n=== MINT Stage 3: PARAFAC ===")

if os.path.exists(MINT_FACTORS_FILE):
    saved = np.load(MINT_FACTORS_FILE, allow_pickle=True)
    n_factors = sum(1 for k in saved.files if k.startswith('factor_'))
    mint_factors = [saved[f'factor_{i}'] for i in range(n_factors)]
    print(f"  Loaded {n_factors} factors from {MINT_FACTORS_FILE}")
else:
    tl.set_backend('pytorch')
    tensor_t = torch.from_numpy(tensor_np).to(dtype=torch.float32, device=DEVICE)

    R_mint = find_optimal_components_parafac(tensor_t)
    _, factors_raw = parafac(tensor_t, rank=R_mint, normalize_factors=True)
    mint_factors = [f.detach().cpu().numpy() for f in factors_raw]

    np.savez(MINT_FACTORS_FILE, **{f'factor_{i}': mint_factors[i] for i in range(len(mint_factors))})
    print(f"  Saved factors to {MINT_FACTORS_FILE}")

for i, f in enumerate(mint_factors):
    print(f"    MINT Factor {i}: {f.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2: NMF (Direct matrix factorization)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("METHOD 2: NMF (DIRECT MATRIX FACTORIZATION)")
print("=" * 80)

NMF_FILE = "combined_nmf_factors.npz"

print(f"\nUsing all {len(series_names)} series for NMF")

def nmf_pipeline(df_matrix, n_components=15):
    """NMF with automatic component selection via elbow method"""
    n_components_range = range(1, min(n_components + 1, min(df_matrix.shape)) + 1)
    reconstruction_errors = []

    print("Finding optimal number of components...")
    for i in n_components_range:
        model = NMF(n_components=i, max_iter=1000, init='random', random_state=0)
        W = model.fit_transform(df_matrix)
        H = model.components_
        reconstructed = W @ H
        error = np.linalg.norm(df_matrix - reconstructed, 'fro')
        reconstruction_errors.append(error)
    
    knee_locator = KneeLocator(
        list(n_components_range), 
        reconstruction_errors, 
        curve='convex', 
        direction='decreasing'
    )
    
    optimal_components = knee_locator.elbow
    if optimal_components is None:
        optimal_components = 2
    
    print(f"  Optimal number of components: {optimal_components}")

    model = NMF(n_components=optimal_components, max_iter=1000, init='random', random_state=0)
    W = model.fit_transform(df_matrix)
    H = model.components_

    reconstructed = W @ H
    error = np.linalg.norm(df_matrix - reconstructed, 'fro')
    df_matrix_norm = np.linalg.norm(df_matrix)
    print(f"  Normalized reconstruction error: {error / df_matrix_norm:.6f}")
    
    return W, H

if os.path.exists(NMF_FILE):
    saved = np.load(NMF_FILE)
    W_nmf, H_nmf = saved['W'], saved['H']
    print(f"Loaded W {W_nmf.shape} and H {H_nmf.shape} from {NMF_FILE}")
else:
    W_nmf, H_nmf = nmf_pipeline(multi_timeseries_df.to_numpy(), n_components=15)
    np.savez(NMF_FILE, W=W_nmf, H=H_nmf)
    print(f"Saved W {W_nmf.shape} and H {H_nmf.shape} to {NMF_FILE}")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION: MINT Figure
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("GENERATING MINT FIGURE")
print("=" * 80)

R_mint = mint_factors[0].shape[1]
GROUP_COLORS = ['#d62728', '#2ca02c', '#1f77b4']   # T1=red, T2=green, T3=blue

# Sort rows of A by ground-truth phase group
group_order = [group_labels[s] for s in series_names]
sorted_indices = sorted(range(len(series_names)), key=lambda i: group_order[i])
A_sorted = mint_factors[0][sorted_indices, :]

n_T1 = sum(1 for g in group_order if g == 0)
n_T2 = sum(1 for g in group_order if g == 1)
n_T3 = sum(1 for g in group_order if g == 2)

# Compute phase-offset positions in MPlot index space
n_valid   = len(x) - SUBSEQ_LEN
scale     = MHEIGHT / n_valid
period_mp = 30 * 24 * scale

phase_offsets_mp = {
    '$T_1$': 0  * 24 * scale,
    '$T_2$': 10 * 24 * scale,
    '$T_3$': 20 * 24 * scale,
}

def phase_lines(ax, color, label, offset_mp):
    """Draw dashed horizontal lines at every recurrence of this phase offset"""
    pos = offset_mp
    first = True
    while pos < MHEIGHT:
        ax.axhline(
            pos,
            color=color, linewidth=0.8, linestyle='--', alpha=0.7,
            label=label if first else None
        )
        first = False
        pos += period_mp

# Create MINT figure
fig_mint = plt.figure(figsize=(8, 11), constrained_layout=True)
outer = gridspec.GridSpec(2, 1, figure=fig_mint, height_ratios=[1.5, 5])

# Panel A — top row
ax_A = fig_mint.add_subplot(outer[0])
im_A = ax_A.imshow(A_sorted, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(im_A, ax=ax_A, fraction=0.046, pad=0.04)
ax_A.set_title('$A$', fontsize=13)
ax_A.set_xlabel('Component', fontsize=9)
ax_A.set_ylabel('Series (sorted by phase group)', fontsize=9)
ax_A.set_xticks(range(R_mint))
ax_A.set_xticklabels([str(i + 1) for i in range(R_mint)], fontsize=8)

for boundary, color, label, mid in [
    (n_T1 - 0.5,          GROUP_COLORS[0], '$T_1$', n_T1 / 2),
    (n_T1 + n_T2 - 0.5,   GROUP_COLORS[1], '$T_2$', n_T1 + n_T2 / 2),
    (None,                 GROUP_COLORS[2], '$T_3$', n_T1 + n_T2 + n_T3 / 2),
]:
    if boundary is not None:
        ax_A.axhline(boundary, color='white', linewidth=1.2, linestyle='-')
    ax_A.text(
        0.98, mid, label,
        color='white', fontsize=9, fontweight='bold', va='center', ha='right',
        transform=ax_A.get_yaxis_transform()
    )

# Panels B and C — bottom row
inner_bc = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1])

for ax, matrix, title, show_ylabel in [
    (fig_mint.add_subplot(inner_bc[0]), mint_factors[1], '$B$', True),
    (fig_mint.add_subplot(inner_bc[1]), mint_factors[2], '$C$', False),
]:
    im = ax.imshow(matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Component', fontsize=9)
    if show_ylabel:
        ax.set_ylabel('MPlot Index', fontsize=9)
    ax.set_xticks(range(R_mint))
    ax.set_xticklabels([str(i + 1) for i in range(R_mint)], fontsize=8)

    for (label, offset_mp), color in zip(phase_offsets_mp.items(), GROUP_COLORS):
        phase_lines(ax, color, label, offset_mp)

    ax.legend(fontsize=7, loc='lower right', framealpha=0.6)

plt.savefig('figures/combined_synthetic/mint_figure.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/combined_synthetic/mint_figure.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/combined_synthetic/mint_figure.svg', bbox_inches='tight')
plt.close()
print("MINT figure saved: figures/combined_synthetic/mint_figure.{pdf,png,svg}")

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION: NMF Figure
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("GENERATING NMF FIGURE")
print("=" * 80)

GROUP_LABELS = ['$T_1$', '$T_2$', '$T_3$']
n_plot_combined = 24 * 30   # 30 days
R_nmf = W_nmf.shape[1]

# Pick representatives for each group
def best_rep(group_range):
    candidates = [f'series_{i}' for i in group_range if f'series_{i}' in series_names]
    in_window = [(s, first_imputation[s]) for s in candidates
                 if first_imputation[s] is not None and first_imputation[s] < n_plot_combined]
    if in_window:
        return min(in_window, key=lambda x: abs(x[1]))[0]
    return candidates[0] if candidates else None

REPRESENTATIVES = [best_rep(range(0, 20)), best_rep(range(20, 40)), best_rep(range(40, 60))]
REPRESENTATIVES = [r for r in REPRESENTATIVES if r is not None]

# Sort H columns by phase group
nmf_sorted_indices = sorted(range(len(series_names)),
                            key=lambda i: group_labels[series_names[i]])
H_sorted = H_nmf[:, nmf_sorted_indices]

fig_nmf = plt.figure(figsize=(9, 5.5), constrained_layout=True)
outer_nmf = gridspec.GridSpec(2, 1, figure=fig_nmf, height_ratios=[1, 1.5])

# Row 1 — sample series
inner_series = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_nmf[0], hspace=0.0)
for g in range(min(3, len(REPRESENTATIVES))):
    ax = fig_nmf.add_subplot(inner_series[g])
    vals = multi_timeseries_df[REPRESENTATIVES[g]].values[:n_plot_combined]
    days = np.arange(len(vals)) / 24
    ax.plot(days, vals, color=GROUP_COLORS[g], linewidth=0.5)

    fp = first_imputation[REPRESENTATIVES[g]]
    if fp is not None and fp < n_plot_combined:
        ax.axvspan(fp / 24, (fp + 25) / 24, color='gray', alpha=0.25, zorder=0)

    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_ylabel(GROUP_LABELS[g], fontsize=11, rotation=0, labelpad=14, va='center')
    ax.set_xlim(0, n_plot_combined / 24)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xticklabels(['0d', '10d', '20d', '30d'], fontsize=8)
    if g < 2:
        ax.set_xticklabels([])
        ax.spines['bottom'].set_visible(False)
    else:
        ax.set_xlabel('Days', fontsize=8)
    if g == 0:
        ax.set_title('Sample Series', fontsize=10, pad=6)

# Row 2 — W^T and H^T
inner_mats = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_nmf[1],
                                              width_ratios=[3, 1])

ax_W = fig_nmf.add_subplot(inner_mats[0])
im_W = ax_W.imshow(W_nmf.T, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(im_W, ax=ax_W, fraction=0.03, pad=0.02)
ax_W.set_title('$W^\\top$', fontsize=13)
ax_W.set_xlabel('Time sample', fontsize=9)
ax_W.set_ylabel('Component', fontsize=9)
ax_W.set_yticks(range(R_nmf))
ax_W.set_yticklabels([str(i + 1) for i in range(R_nmf)], fontsize=8)

ax_H = fig_nmf.add_subplot(inner_mats[1])
im_H = ax_H.imshow(H_sorted.T, aspect='auto', cmap='viridis', interpolation='nearest')
plt.colorbar(im_H, ax=ax_H, fraction=0.1, pad=0.04)
ax_H.set_title('$H^\\top$', fontsize=13)
ax_H.set_xlabel('Component', fontsize=9)
ax_H.set_ylabel('Series (sorted by phase group)', fontsize=9)
ax_H.set_xticks(range(R_nmf))
ax_H.set_xticklabels([str(i + 1) for i in range(R_nmf)], fontsize=8)

plt.savefig('figures/combined_synthetic/nmf_figure.pdf', dpi=300, bbox_inches='tight')
plt.savefig('figures/combined_synthetic/nmf_figure.png', dpi=150, bbox_inches='tight')
plt.savefig('figures/combined_synthetic/nmf_figure.svg', bbox_inches='tight')
plt.close()
print("NMF figure saved: figures/combined_synthetic/nmf_figure.{pdf,png,svg}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)
print(f"\nMINT Results:")
print(f"  Series used: 60")
print(f"  Components found: {R_mint}")
print(f"  Factor shapes: A={mint_factors[0].shape}, B={mint_factors[1].shape}, C={mint_factors[2].shape}")

print(f"\nNMF Results:")
print(f"  Series used: 60")
print(f"  Components found: {R_nmf}")
print(f"  Factor shapes: W={W_nmf.shape}, H={H_nmf.shape}")

print(f"\nFigures saved to: figures/combined_synthetic/")
print("=" * 80)
