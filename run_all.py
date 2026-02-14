#!/usr/bin/env python3
"""
PaW Toy Model — Full simulation pipeline
==========================================

Runs Version A and Version B of the Page-Wootters demonstrator,
generates publication-quality plots, computes all metrics from
Sections 4-6, and exports CSV tables for the paper.

Usage:
    python run_all.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for PNG export
import matplotlib.pyplot as plt
import csv
import os

from paw_core import run_version_a, run_version_b, clock_back_action


# ── Configuration ─────────────────────────────────────────────────

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paper reference parameters
N      = 30
DT     = 0.2
OMEGA  = 1.0
G      = 0.1
N_ENVS = [2, 4, 6, 8]

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


# ══════════════════════════════════════════════════════════════════
#  VERSION A — No Environment
# ══════════════════════════════════════════════════════════════════

print("=" * 60)
print("VERSION A — No Environment")
print("=" * 60)

res_a = run_version_a(N, DT, OMEGA, build_full_history=True)

print(f"  Parameters: N={N}, dt={DT}, ω={OMEGA}")
print(f"  Theoretical period: {2*np.pi/(OMEGA*DT):.1f} steps")
print(f"  Max |⟨σ_z⟩|: {np.max(np.abs(res_a.sz_values)):.6f}")
print(f"  Max deviation from cos(ωkdt): "
      f"{np.max(np.abs(res_a.sz_values - res_a.sz_theory)):.2e}")

# ── Plot A ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(res_a.k_values, res_a.sz_values, 'o-', color='#2196F3',
        markersize=5, linewidth=1.2,
        label=r'$\langle\sigma_z\rangle_k$ (PaW conditioned)')
ax.plot(res_a.k_values, res_a.sz_theory, '--', color='gray', alpha=0.7,
        linewidth=1.5, label=r'$\cos(\omega\, k\, dt)$ (theory)')
ax.set_xlabel('k (clock tick)')
ax.set_ylabel(r'$\langle\sigma_z\rangle$')
ax.set_title('Version A: Coherent oscillations (no environment)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-1.15, 1.15)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/version_A_oscillation.png")
plt.close(fig)
print(f"  → {OUTPUT_DIR}/version_A_oscillation.png")


# ══════════════════════════════════════════════════════════════════
#  VERSION B — With Environment (single n_env=4)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("VERSION B — With Environment (n_env = 4)")
print("=" * 60)

res_b4 = run_version_b(N, DT, OMEGA, G, n_env=4)

print(f"  Parameters: N={N}, dt={DT}, ω={OMEGA}, g={G}, n_env=4")
print(f"  S_eff initial: {res_b4.s_eff_values[0]:.4f}")
print(f"  S_eff final:   {res_b4.s_eff_values[-1]:.4f}")
print(f"  S_eff max:     {np.max(res_b4.s_eff_values):.4f}")
print(f"  Fidelity final: {res_b4.fidelity_values[-1]:.4f}")

# ── Plot B (dual panel) ──────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(res_b4.k_values, res_b4.sz_values, 'o-', color='#E91E63',
         markersize=4, linewidth=1, label=r'$\langle\sigma_z\rangle_k$')
ax1.plot(res_b4.k_values, res_b4.sz_theory, '--', color='gray', alpha=0.5,
         label=r'$\cos(\omega\, k\, dt)$ (ideal)')
ax1.set_xlabel('k')
ax1.set_ylabel(r'$\langle\sigma_z\rangle$')
ax1.set_title('Damped oscillations')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(res_b4.k_values, res_b4.s_eff_values, 's-', color='#FF5722',
         markersize=4, linewidth=1, label=r'$S_{\mathrm{eff}}(k)$')
ax2.set_xlabel('k')
ax2.set_ylabel(r'$S_{\mathrm{eff}}$')
ax2.set_title('Effective entropy growth')
ax2.legend()
ax2.grid(True, alpha=0.3)

fig.suptitle(f'Version B: n_env = 4, g = {G}', fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/version_B_n4.png", bbox_inches='tight')
plt.close(fig)
print(f"  → {OUTPUT_DIR}/version_B_n4.png")


# ══════════════════════════════════════════════════════════════════
#  MULTI-ENVIRONMENT COMPARISON
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("MULTI-ENVIRONMENT COMPARISON")
print("=" * 60)

colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
results_b = {}

for ne in N_ENVS:
    print(f"  Running n_env={ne} ...", end=" ", flush=True)
    res = run_version_b(N, DT, OMEGA, G, n_env=ne)
    results_b[ne] = res
    print(f"done  (S_eff final = {res.s_eff_values[-1]:.4f}, "
          f"fidelity final = {res.fidelity_values[-1]:.4f})")

# ── 2×2 grid: σ_z + S_eff per n_env ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, ne in enumerate(N_ENVS):
    r = results_b[ne]
    row, col = divmod(i, 2)
    ax = axes[row, col]

    ax.plot(r.k_values, r.sz_values, 'o-', color=colors[i],
            markersize=3, alpha=0.8, label=r'$\langle\sigma_z\rangle$')
    ax.plot(r.k_values, r.sz_theory, '--', color='gray', alpha=0.4)
    ax.set_ylabel(r'$\langle\sigma_z\rangle$')

    ax_tw = ax.twinx()
    ax_tw.plot(r.k_values, r.s_eff_values, 's-', color='red',
               markersize=3, alpha=0.6, label=r'$S_{\mathrm{eff}}$')
    ax_tw.set_ylabel(r'$S_{\mathrm{eff}}$', color='red')
    ax_tw.tick_params(axis='y', labelcolor='red')

    ax.set_title(f'$n_{{env}} = {ne}$')
    ax.set_xlabel('k')
    ax.grid(True, alpha=0.2)

fig.suptitle(f'Environment-size comparison  (N={N}, g={G})', fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/multi_nenv_grid.png")
plt.close(fig)
print(f"  → {OUTPUT_DIR}/multi_nenv_grid.png")

# ── S_eff overlay ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for i, ne in enumerate(N_ENVS):
    ax.plot(results_b[ne].k_values, results_b[ne].s_eff_values,
            'o-', color=colors[i], markersize=3, label=f'$n_{{env}} = {ne}$')
ax.set_xlabel('k (clock tick)')
ax.set_ylabel(r'$S_{\mathrm{eff}}(k)$')
ax.set_title('Informational arrow: entropy growth vs environment size')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/entropy_comparison.png")
plt.close(fig)
print(f"  → {OUTPUT_DIR}/entropy_comparison.png")

# ── Fidelity overlay (Sec. 4.3 metric) ──────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for i, ne in enumerate(N_ENVS):
    ax.plot(results_b[ne].k_values, results_b[ne].fidelity_values,
            'o-', color=colors[i], markersize=3, label=f'$n_{{env}} = {ne}$')
ax.set_xlabel('k (clock tick)')
ax.set_ylabel(r'$F^2(k)$')
ax.set_title('Fidelity vs ideal Schrödinger dynamics (Sec. 4.3)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/fidelity_comparison.png")
plt.close(fig)
print(f"  → {OUTPUT_DIR}/fidelity_comparison.png")


# ══════════════════════════════════════════════════════════════════
#  CLOCK BACK-ACTION METRIC (Sec. 4.2)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CLOCK BACK-ACTION METRIC (Sec. 4.2)")
print("=" * 60)

ks_ba, delta_E = clock_back_action(N, DT)
print(f"  ΔE_C(0) = {delta_E[0]:+.4f}")
print(f"  ΔE_C({N-1}) = {delta_E[-1]:+.4f}")
print("  (Linear for orthogonal clock readouts — non-trivial only")
print("   for overlapping/quasi-ideal clock POVMs)")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(ks_ba, delta_E, 'o-', color='#607D8B', markersize=4)
ax.set_xlabel('k (clock tick)')
ax.set_ylabel(r'$\Delta E_C(k)$')
ax.set_title('Clock back-action metric (Sec. 4.2)')
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUTPUT_DIR}/back_action.png")
plt.close(fig)
print(f"  → {OUTPUT_DIR}/back_action.png")


# ══════════════════════════════════════════════════════════════════
#  CSV EXPORT
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EXPORTING CSV TABLES")
print("=" * 60)

# ── Table 1: Version A ───────────────────────────────────────────
path = f"{OUTPUT_DIR}/table_version_A.csv"
with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['k', 'sz_conditioned', 'sz_theory', 'deviation'])
    for k in range(N):
        w.writerow([
            k,
            f"{res_a.sz_values[k]:.6f}",
            f"{res_a.sz_theory[k]:.6f}",
            f"{abs(res_a.sz_values[k] - res_a.sz_theory[k]):.2e}",
        ])
print(f"  → {path}")

# ── Table 2: Version B (n_env=4) ────────────────────────────────
path = f"{OUTPUT_DIR}/table_version_B_n4.csv"
with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['k', 'sz', 'S_eff', 'fidelity_sq', 'sz_theory'])
    for k in range(N):
        w.writerow([
            k,
            f"{res_b4.sz_values[k]:.6f}",
            f"{res_b4.s_eff_values[k]:.6f}",
            f"{res_b4.fidelity_values[k]:.6f}",
            f"{res_b4.sz_theory[k]:.6f}",
        ])
print(f"  → {path}")

# ── Table 3: Multi n_env summary ────────────────────────────────
path = f"{OUTPUT_DIR}/table_multi_nenv_summary.csv"
with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow([
        'n_env', 'd_E', 'S_eff_initial', 'S_eff_final', 'S_eff_max',
        'sz_amplitude_last10', 'fidelity_final',
    ])
    for ne in N_ENVS:
        r = results_b[ne]
        amp = np.max(np.abs(r.sz_values[-10:]))
        w.writerow([
            ne,
            2 ** ne,
            f"{r.s_eff_values[0]:.6f}",
            f"{r.s_eff_values[-1]:.6f}",
            f"{np.max(r.s_eff_values):.6f}",
            f"{amp:.6f}",
            f"{r.fidelity_values[-1]:.6f}",
        ])
print(f"  → {path}")

# ── Table 4: Full multi-n_env detail ────────────────────────────
path = f"{OUTPUT_DIR}/table_version_B_all.csv"
with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    header = ['k']
    for ne in N_ENVS:
        header += [f'sz_n{ne}', f'Seff_n{ne}', f'fid_n{ne}']
    w.writerow(header)
    for k in range(N):
        row = [k]
        for ne in N_ENVS:
            r = results_b[ne]
            row += [
                f"{r.sz_values[k]:.6f}",
                f"{r.s_eff_values[k]:.6f}",
                f"{r.fidelity_values[k]:.6f}",
            ]
        w.writerow(row)
print(f"  → {path}")

# ══════════════════════════════════════════════════════════════════
#  CLOCK REVERSAL VALIDATION (Gap 1)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CLOCK REVERSAL — Validation across all three pillars")
print("=" * 60)

import subprocess, sys
subprocess.run([sys.executable, "generate_clock_reversal.py"], check=True)

# ══════════════════════════════════════════════════════════════════
#  CLOCK ORIENTATION COVARIANCE THEOREM (Gap 2)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("COVARIANCE THEOREM — Permutation invariance & T-symmetry distinction")
print("=" * 60)

subprocess.run([sys.executable, "generate_covariance_theorem.py"], check=True)

# ══════════════════════════════════════════════════════════════════
#  ANGULAR INTERPOLATION OF CLOCK ORIENTATION (Gap 3)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ANGULAR INTERPOLATION — Continuous clock orientation & temporal interference")
print("=" * 60)

subprocess.run([sys.executable, "generate_angular_interpolation.py"], check=True)

# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("CONTINUUM LIMIT — N→∞ convergence, clock transformations & group structure")
print("=" * 60)

subprocess.run([sys.executable, "generate_continuum_limit.py"], check=True)

print("\n" + "=" * 60)
print("✓  All simulations complete.")
print(f"   Figures and tables in: {OUTPUT_DIR}/")
print("=" * 60)
