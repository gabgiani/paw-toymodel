"""
Generate all figures for the God Observer analysis.
Tests the PaW formula under total access (omniscient observer).
"""
import sys
sys.path.insert(0, '/Users/gianig/paw-toymodel')

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from validate_formula import build_paw_history, get_conditioned_observables
from validate_formula import N, dt, omega, initial_S, g, sigma_z

# ══════════════════════════════════════════════════════════════
# Figure 1: God (n_env=0) vs Limited observer (n_env=4)
# Side-by-side: dynamics + entropy for both
# ══════════════════════════════════════════════════════════════
print("Building Version A (God: no environment)...")
psi_god = build_paw_history(N, dt, omega, initial_S, n_env=0)
sz_god, S_god = get_conditioned_observables(psi_god, N, 0)

print("Building Version B (limited: n_env=4)...")
psi_lim = build_paw_history(N, dt, omega, initial_S, n_env=4, g=g)
sz_lim, S_lim = get_conditioned_observables(psi_lim, N, 4)

analytic = np.cos(omega * np.arange(N) * dt)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Top-left: God dynamics
axes[0, 0].plot(range(N), sz_god, 'o-', color='royalblue', markersize=5, label=r'$\langle\sigma_z\rangle_k$ (God)')
axes[0, 0].plot(range(N), analytic, 'k--', alpha=0.5, label=r'$\cos(\omega k\,dt)$')
axes[0, 0].set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=12)
axes[0, 0].set_title('God observer: dynamics (reversible)', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)
axes[0, 0].set_ylim(-1.15, 1.15)

# Top-right: God entropy
axes[0, 1].plot(range(N), S_god, 's-', color='royalblue', markersize=5, label=r'$S_{\rm eff}(k)$ (God)')
axes[0, 1].axhline(np.log(2), color='gray', linestyle=':', alpha=0.5, label=r'$\ln 2$')
axes[0, 1].set_ylabel(r'$S_{\rm eff}$', fontsize=12)
axes[0, 1].set_title(r'God observer: $S_{\rm eff} = 0$ always (no arrow)', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_ylim(-0.05, 0.8)
axes[0, 1].text(15, 0.35, 'NO ARROW\n(nothing hidden)', fontsize=14, ha='center',
                color='royalblue', fontweight='bold', alpha=0.7)

# Bottom-left: Limited dynamics
axes[1, 0].plot(range(N), sz_lim, 'o-', color='green', markersize=5, label=r'$\langle\sigma_z\rangle_k$ (limited)')
axes[1, 0].plot(range(N), analytic, 'k--', alpha=0.3, label=r'$\cos(\omega k\,dt)$ ideal')
axes[1, 0].set_xlabel('Clock reading $k$', fontsize=12)
axes[1, 0].set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=12)
axes[1, 0].set_title('Limited observer: damped dynamics', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(alpha=0.3)
axes[1, 0].set_ylim(-1.15, 1.15)

# Bottom-right: Limited entropy
axes[1, 1].plot(range(N), S_lim, 's-', color='red', markersize=5, label=r'$S_{\rm eff}(k)$ (limited)')
axes[1, 1].axhline(np.log(2), color='gray', linestyle=':', alpha=0.5, label=r'$\ln 2 \approx 0.693$')
axes[1, 1].set_xlabel('Clock reading $k$', fontsize=12)
axes[1, 1].set_ylabel(r'$S_{\rm eff}$', fontsize=12)
axes[1, 1].set_title(r'Limited observer: $S_{\rm eff} \to \ln 2$ (arrow emerges)', fontsize=13, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_ylim(-0.05, 0.8)

plt.suptitle('Same formula, same universe — different access', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('output/god_vs_limited.png', dpi=150, bbox_inches='tight')
print("Saved: output/god_vs_limited.png")

# ══════════════════════════════════════════════════════════════
# Figure 2: Progressive blindness — S_eff(final) vs n_env
# ══════════════════════════════════════════════════════════════
print("\nProgressive blindness sweep...")
n_env_list = [0, 1, 2, 3, 4, 5, 6]
S_final_list = []
S_curves = {}

for ne in n_env_list:
    print("  n_env = {}...".format(ne))
    if ne == 0:
        psi = build_paw_history(N, dt, omega, initial_S, n_env=0)
        _, S = get_conditioned_observables(psi, N, 0)
    else:
        psi = build_paw_history(N, dt, omega, initial_S, n_env=ne, g=g)
        _, S = get_conditioned_observables(psi, N, ne)
    S_final_list.append(S[-1])
    S_curves[ne] = S

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: S_eff curves for each n_env
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(n_env_list)))
for i, ne in enumerate(n_env_list):
    label = 'God ($n_{{env}}=0$)' if ne == 0 else '$n_{{env}}={}$'.format(ne)
    lw = 3 if ne == 0 else 1.5
    axes[0].plot(range(N), S_curves[ne], '-', color=colors[i], linewidth=lw,
                 label=label, alpha=0.9)
axes[0].axhline(np.log(2), color='gray', linestyle=':', alpha=0.5)
axes[0].set_xlabel('Clock reading $k$', fontsize=12)
axes[0].set_ylabel(r'$S_{\rm eff}(k)$', fontsize=12)
axes[0].set_title('Progressive blindness: entropy growth per environment size', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=9, ncol=2)
axes[0].grid(alpha=0.3)

# Right: Final S_eff as bar chart
bar_colors = ['royalblue'] + ['green'] * (len(n_env_list) - 1)
bars = axes[1].bar(n_env_list, S_final_list, color=bar_colors, edgecolor='black', alpha=0.8)
axes[1].axhline(np.log(2), color='red', linestyle='--', linewidth=1.5, label=r'$\ln 2$')
axes[1].set_xlabel('$n_{env}$ (hidden qubits)', fontsize=12)
axes[1].set_ylabel(r'$S_{\rm eff}^{\rm final}$', fontsize=12)
axes[1].set_title('More hidden qubits = stronger arrow', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3, axis='y')
# Annotate God bar
axes[1].annotate('GOD\n(no arrow)', xy=(0, S_final_list[0] + 0.02), fontsize=10,
                 ha='center', color='royalblue', fontweight='bold')

plt.tight_layout()
plt.savefig('output/god_progressive_blindness.png', dpi=150, bbox_inches='tight')
print("Saved: output/god_progressive_blindness.png")

# ══════════════════════════════════════════════════════════════
# Figure 3: Level 2 — God sees |Ψ⟩ directly (no clock conditioning)
# Show: global sigma_z is a single frozen number
# ══════════════════════════════════════════════════════════════
print("\nLevel 2: Global state analysis...")
psi_global = build_paw_history(N, dt, omega, initial_S, n_env=4, g=g)

# Trace out everything except S
rho_S_global = psi_global.ptrace(1)
sz_global = qt.expect(qt.sigmaz(), rho_S_global)
sx_global = qt.expect(qt.sigmax(), rho_S_global)
sy_global = qt.expect(qt.sigmay(), rho_S_global)
S_global = qt.entropy_vn(rho_S_global)

# Compare: conditioned values (time sequence) vs global (frozen)
sz_cond, S_cond = get_conditioned_observables(psi_global, N, 4)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Conditioned sigma_z (has dynamics) vs global sigma_z (frozen)
axes[0].plot(range(N), sz_cond, 'o-', color='green', markersize=4, label=r'$\langle\sigma_z\rangle_k$ conditioned on clock')
axes[0].axhline(sz_global, color='red', linewidth=3, linestyle='-', alpha=0.8,
                label=r'$\langle\sigma_z\rangle_{{global}} = {:.4f}$ (frozen)'.format(sz_global))
axes[0].fill_between(range(N), sz_global - 0.02, sz_global + 0.02, color='red', alpha=0.15)
axes[0].set_xlabel('Clock reading $k$', fontsize=12)
axes[0].set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=12)
axes[0].set_title('Conditioned (time exists) vs Global (time absent)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Right: Conditioned entropy (grows) vs global entropy (fixed)
axes[1].plot(range(N), S_cond, 's-', color='red', markersize=4, label=r'$S_{\rm eff}(k)$ conditioned')
axes[1].axhline(S_global, color='purple', linewidth=3, linestyle='-', alpha=0.8,
                label=r'$S_{{\rm vn}}(\rho_S^{{\rm global}}) = {:.4f}$ (frozen)'.format(S_global))
axes[1].axhline(np.log(2), color='gray', linestyle=':', alpha=0.5, label=r'$\ln 2$')
axes[1].set_xlabel('Clock reading $k$', fontsize=12)
axes[1].set_ylabel(r'$S_{\rm eff}$', fontsize=12)
axes[1].set_title('Conditioned entropy (dynamic) vs Global entropy (static)', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

plt.suptitle(r'Level 2: Without clock conditioning, $\langle\sigma_z\rangle$ is a single frozen number — no time',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('output/god_level2_frozen.png', dpi=150, bbox_inches='tight')
print("Saved: output/god_level2_frozen.png")

# ══════════════════════════════════════════════════════════════
# Figure 4: Summary diagram — 3 levels of omniscience
# ══════════════════════════════════════════════════════════════
print("\nSummary figure...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

# Level 1: God with clock
axes[0].plot(range(N), sz_god, 'o-', color='royalblue', markersize=4)
axes[0].plot(range(N), analytic, 'k--', alpha=0.3)
ax0b = axes[0].twinx()
ax0b.plot(range(N), S_god, 's-', color='orange', markersize=3, alpha=0.7)
ax0b.set_ylim(-0.05, 0.8)
ax0b.set_ylabel(r'$S_{\rm eff}$ (orange)', fontsize=10, color='orange')
axes[0].set_xlabel('$k$', fontsize=12)
axes[0].set_ylabel(r'$\langle\sigma_z\rangle$ (blue)', fontsize=10, color='royalblue')
axes[0].set_title('Level 1: God + clock\n(dynamics, no arrow)', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].text(15, -0.7, r'$S_{\rm eff} \equiv 0$', fontsize=14, ha='center',
             color='darkorange', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='orange'))

# Level 2: God sees |Ψ⟩
axes[1].axhline(sz_global, color='red', linewidth=4, alpha=0.8)
axes[1].fill_between(range(N), sz_global - 0.03, sz_global + 0.03, color='red', alpha=0.15)
axes[1].set_xlabel('$k$ (meaningless — no clock)', fontsize=12)
axes[1].set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=12)
axes[1].set_title('Level 2: God sees $|\\Psi\\rangle$ directly\n(no dynamics, no time)', fontsize=12, fontweight='bold')
axes[1].set_ylim(-1.15, 1.15)
axes[1].grid(alpha=0.3)
axes[1].text(15, 0.5, r'$\langle\sigma_z\rangle = {:.4f}$'.format(sz_global) + '\n(one frozen number)',
             fontsize=13, ha='center', color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='mistyrose', edgecolor='red'))

# Level 3: Universe IS
axes[2].text(0.5, 0.65, r'$\hat{\mathcal{C}}\,|\Psi\rangle = 0$', fontsize=28, ha='center', va='center',
             transform=axes[2].transAxes, fontweight='bold', color='darkblue')
axes[2].text(0.5, 0.40, r'$S_{\rm vn}(|\Psi\rangle\langle\Psi|) = 0$', fontsize=20, ha='center', va='center',
             transform=axes[2].transAxes, color='purple')
axes[2].text(0.5, 0.18, 'No time. No arrow.\nNo history. No experience.\nThe universe simply IS.',
             fontsize=13, ha='center', va='center', transform=axes[2].transAxes,
             style='italic', color='gray')
axes[2].set_title('Level 3: Global atemporal object\n(pure state, zero entropy)', fontsize=12, fontweight='bold')
axes[2].set_xticks([])
axes[2].set_yticks([])
for spine in axes[2].spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig('output/god_three_levels.png', dpi=150, bbox_inches='tight')
print("Saved: output/god_three_levels.png")

# ══════════════════════════════════════════════════════════════
# Export data
# ══════════════════════════════════════════════════════════════
import csv
with open('output/table_god_progressive_blindness.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['n_env', 'd_E', 'S_eff_final', 'has_arrow'])
    for ne, sf in zip(n_env_list, S_final_list):
        d_e = 2**ne if ne > 0 else 1
        arrow = 'NO' if sf < 0.01 else 'YES'
        writer.writerow([ne, d_e, '{:.6f}'.format(sf), arrow])

print("Saved: output/table_god_progressive_blindness.csv")
print("\nAll God observer figures generated successfully.")
