"""
Geometric interpretation of the unified relational formula.

Produces:
  output/geometric_interpretation.png  — 3-panel conceptual figure
  output/bloch_trajectory.png          — Bloch disk: Version A vs B

Three-panel figure:
  Left  : Timeless global state |Ψ⟩ on constraint surface Ĉ|Ψ⟩=0
  Center: Relational bundle — clock base, ρ_S(k) fibers
  Right : Bloch disk trajectory — pure circle vs inward spiral

The Bloch trajectory shows how the thermodynamic arrow is geometrically
encoded: Version A stays on the Bloch sphere surface (pure, reversible),
while Version B spirals toward the center (mixed, irreversible).
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.patches import (Ellipse, FancyArrowPatch, Circle,
                                Arc, Wedge, PathPatch)
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
import os

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Parameters (consistent with validate_formula.py) ──────────
N = 30
dt = 0.2
omega = 1.0
g = 0.1
n_env = 4

initial_S = qt.basis(2, 0)
sigma_x = qt.sigmax()
sigma_y = qt.sigmay()
sigma_z = qt.sigmaz()


# ══════════════════════════════════════════════════════════════
#  Compute Bloch trajectories from the PaW formula
# ══════════════════════════════════════════════════════════════

def compute_bloch_trajectories():
    """
    Compute Bloch vector components for Version A and Version B.

    Version A (no environment): trajectory stays on Bloch sphere surface.
    Version B (n_env=4, Tr_E): trajectory spirals inward.

    Returns
    -------
    dict with keys:  bx_a, by_a, bz_a  (Version A)
                     bx_b, by_b, bz_b  (Version B)
                     s_eff_b            (entropy, Version B)
                     bloch_radius_b     (|r| for Version B)
    """
    # ── Version A: pure dynamics ──
    H_S = (omega / 2) * sigma_x
    bx_a, by_a, bz_a = [], [], []

    for k in range(N):
        U = (-1j * H_S * k * dt).expm()
        psi_k = U * initial_S
        rho_k = psi_k * psi_k.dag()
        bx_a.append(qt.expect(sigma_x, rho_k))
        by_a.append(qt.expect(sigma_y, rho_k))
        bz_a.append(qt.expect(sigma_z, rho_k))

    # ── Version B: with environment ──
    dim_env = 2**n_env
    id_list = [qt.qeye(2) for _ in range(n_env)]
    H_S_full = (omega / 2) * qt.tensor(sigma_x, *id_list)

    H_SE = qt.Qobj(np.zeros((2 * dim_env, 2 * dim_env)),
                    dims=[[2] + [2]*n_env, [2] + [2]*n_env])
    for j in range(n_env):
        ops = [qt.qeye(2) for _ in range(n_env)]
        ops[j] = sigma_x
        H_SE += g * qt.tensor(sigma_x, *ops)

    H_tot = H_S_full + H_SE
    env0 = qt.tensor([qt.basis(2, 0) for _ in range(n_env)])
    initial_SE = qt.tensor(initial_S, env0)

    bx_b, by_b, bz_b, s_eff_b = [], [], [], []

    for k in range(N):
        U = (-1j * H_tot * k * dt).expm()
        psi_SE = U * initial_SE
        rho_S = psi_SE.ptrace(0)
        bx_b.append(qt.expect(sigma_x, rho_S))
        by_b.append(qt.expect(sigma_y, rho_S))
        bz_b.append(qt.expect(sigma_z, rho_S))
        s_eff_b.append(qt.entropy_vn(rho_S))

    result = {
        'bx_a': np.array(bx_a), 'by_a': np.array(by_a), 'bz_a': np.array(bz_a),
        'bx_b': np.array(bx_b), 'by_b': np.array(by_b), 'bz_b': np.array(bz_b),
        's_eff_b': np.array(s_eff_b),
    }
    result['bloch_radius_a'] = np.sqrt(result['bx_a']**2 + result['by_a']**2 + result['bz_a']**2)
    result['bloch_radius_b'] = np.sqrt(result['bx_b']**2 + result['by_b']**2 + result['bz_b']**2)
    return result


# ══════════════════════════════════════════════════════════════
#  Drawing helpers
# ══════════════════════════════════════════════════════════════

def draw_constraint_surface(ax):
    """
    Panel 1: Timeless global state |Ψ⟩ on constraint hypersurface.
    Schematic of CP(H) with Ĉ=0 submanifold.
    """
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')

    # Projective Hilbert space (outer ellipse)
    ellipse = Ellipse((0, 0), 2.2, 1.8, facecolor='#e8f0fe',
                       edgecolor='#4a86c8', linewidth=2, alpha=0.5)
    ax.add_patch(ellipse)
    ax.text(0, 0.98, r'$\mathcal{CP}(\mathcal{H})$',
            ha='center', va='center', fontsize=13, color='#4a86c8',
            fontweight='bold')

    # Constraint surface Ĉ=0 (smooth curve inside)
    t_curve = np.linspace(-0.85, 0.85, 200)
    y_curve = 0.4 * np.sin(2.5 * t_curve) - 0.1
    ax.plot(t_curve, y_curve, color='#d4380d', linewidth=2.5, zorder=3)
    ax.text(0.75, -0.45, r'$\hat{C}=0$', fontsize=12, color='#d4380d',
            fontweight='bold')

    # The state |Ψ⟩ (point on the curve)
    psi_x, psi_y = 0.15, 0.4 * np.sin(2.5 * 0.15) - 0.1
    ax.plot(psi_x, psi_y, 'o', color='#722ed1', markersize=14, zorder=5,
            markeredgecolor='white', markeredgewidth=2)
    ax.annotate(r'$|\Psi\rangle$', xy=(psi_x, psi_y),
                xytext=(psi_x + 0.35, psi_y + 0.35),
                fontsize=15, fontweight='bold', color='#722ed1',
                arrowprops=dict(arrowstyle='->', color='#722ed1', lw=1.5))

    # Annotations
    ax.text(0, -0.75, 'Timeless\nstationary state',
            ha='center', va='center', fontsize=10, color='#555',
            style='italic')
    ax.text(0, -1.05, r'$S_{\rm eff} = 0$ · no $t$ · no arrow',
            ha='center', va='center', fontsize=9, color='#888',
            family='monospace')

    ax.set_title('Global object', fontsize=13, fontweight='bold', pad=12)


def draw_relational_bundle(ax, data):
    """
    Panel 2: Relational bundle over the clock subsystem C.
    Base = discrete clock readings k. Fibers = state at each k.
    """
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.3, 1.35)
    ax.axis('off')

    # Select representative k values
    k_show = [0, 3, 7, 12, 18, 25, 29]
    x_pos = np.linspace(0, 6, len(k_show))

    # Draw base line (clock readings)
    ax.plot([-0.3, 6.3], [0, 0], '-', color='#333', linewidth=2.5, zorder=2)
    ax.text(3.0, -0.18, r'Base: clock readings $\{|k\rangle_C\}$',
            ha='center', va='center', fontsize=9, color='#555')

    # Draw fibers and state points
    for i, (k, x) in enumerate(zip(k_show, x_pos)):
        # Fiber (vertical line)
        ax.plot([x, x], [0, 1.1], '-', color='#ccc', linewidth=1.2, zorder=1)

        # State point on fiber (Bloch radius = purity)
        r = data['bloch_radius_b'][k]
        s = data['s_eff_b'][k]
        y_state = 0.1 + 0.95 * r  # map radius to height

        # Color by entropy (blue=pure → red=mixed)
        frac = s / np.log(2) if np.log(2) > 0 else 0
        color = plt.cm.coolwarm(frac)

        ax.plot(x, y_state, 'o', color=color, markersize=10, zorder=4,
                markeredgecolor='white', markeredgewidth=1.5)

        # k label
        ax.text(x, -0.08, f'k={k}', ha='center', va='top', fontsize=7,
                color='#666')

    # Connect the state points (section of the bundle)
    y_section = [0.1 + 0.95 * data['bloch_radius_b'][k] for k in k_show]
    ax.plot(x_pos, y_section, '--', color='#722ed1', linewidth=1.5,
            alpha=0.7, zorder=3)

    # Label: fiber
    ax.annotate('fiber:\n' + r'$\mathcal{D}(\mathcal{H}_S)$',
                xy=(6, 0.55), fontsize=9, color='#888', ha='center',
                va='center')

    # Label: section
    ax.text(3.5, 1.22, r'Section: $k \mapsto \rho_S(k)$',
            ha='center', va='center', fontsize=10, color='#722ed1',
            fontweight='bold')

    # Arrow showing projection operation
    ax.annotate('', xy=(0, 1.1), xytext=(0, 1.3),
                arrowprops=dict(arrowstyle='<-', color='#d4380d', lw=1.5))
    ax.text(1.5, 1.32, r'$\langle k|_C \otimes \mathrm{Tr}_E$',
            ha='center', va='center', fontsize=10, color='#d4380d')

    ax.set_title('Relational bundle', fontsize=13, fontweight='bold', pad=12)


def draw_bloch_disk(ax, data):
    """
    Panel 3: Bloch disk (y-z plane) showing trajectories.
    Version A: circle on boundary (pure, reversible).
    Version B: spiral toward center (mixed, irreversible).
    """
    # Draw Bloch sphere boundary (unit circle)
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), '-', color='#ddd', linewidth=1.5,
            zorder=1)
    ax.fill(np.cos(theta), np.sin(theta), color='#f9f9f9', alpha=0.3)

    # Axes
    ax.axhline(0, color='#eee', linewidth=0.8, zorder=0)
    ax.axvline(0, color='#eee', linewidth=0.8, zorder=0)
    ax.text(1.12, 0.0, r'$\langle\sigma_y\rangle$', fontsize=9, color='#999',
            va='center')
    ax.text(0.0, 1.15, r'$\langle\sigma_z\rangle$', fontsize=9, color='#999',
            ha='center')

    # Version A trajectory (y-z plane, circle on surface)
    by_a, bz_a = data['by_a'], data['bz_a']
    ax.plot(by_a, bz_a, '-', color='#4a86c8', linewidth=1.5, alpha=0.4,
            zorder=2, label='Version A (no Tr$_E$)')
    ax.plot(by_a[0], bz_a[0], 'o', color='#4a86c8', markersize=8, zorder=5,
            markeredgecolor='white', markeredgewidth=1.5)

    # Version B trajectory (spiral inward) — color by S_eff
    by_b, bz_b = data['by_b'], data['bz_b']
    s_eff = data['s_eff_b']
    s_norm = s_eff / np.log(2)  # normalize to [0, 1]

    # Draw as colored line segments
    points = np.array([by_b, bz_b]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='coolwarm', linewidth=2.5, zorder=3)
    lc.set_array(s_norm[:-1])
    ax.add_collection(lc)

    # Start and end markers
    ax.plot(by_b[0], bz_b[0], 'D', color='#1a7f37', markersize=9, zorder=6,
            markeredgecolor='white', markeredgewidth=1.5)
    ax.plot(by_b[-1], bz_b[-1], 's', color='#d4380d', markersize=9, zorder=6,
            markeredgecolor='white', markeredgewidth=1.5)

    # Center point (maximally mixed)
    ax.plot(0, 0, '+', color='#d4380d', markersize=14, markeredgewidth=2,
            zorder=4, alpha=0.5)
    ax.text(0.12, -0.12, r'$\frac{I}{2}$', fontsize=10, color='#d4380d',
            alpha=0.7)

    # Arrow toward center (geometric arrow of time)
    mid_k = N // 2
    ax.annotate('', xy=(by_b[-1]*0.5, bz_b[-1]*0.5),
                xytext=(by_b[mid_k], bz_b[mid_k]),
                arrowprops=dict(arrowstyle='->', color='#d4380d',
                                lw=2, connectionstyle='arc3,rad=0.2'))
    ax.text(-0.55, -0.7, r'Arrow: $S_{\rm eff}\!\uparrow$',
            fontsize=10, color='#d4380d', fontweight='bold')

    # Legend markers
    ax.plot([], [], 'D', color='#1a7f37', markersize=6, label='k=0 (pure)')
    ax.plot([], [], 's', color='#d4380d', markersize=6,
            label=f'k={N-1} (mixed)')

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax.set_title(r'Trajectory in $\mathcal{D}(\mathcal{H}_S)$',
                 fontsize=13, fontweight='bold', pad=12)
    ax.axis('off')


# ══════════════════════════════════════════════════════════════
#  Figure 1: Three-panel geometric interpretation
# ══════════════════════════════════════════════════════════════

def generate_main_figure(data):
    """Generate the 3-panel geometric interpretation figure."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        'Geometric Structure: from timeless state to emergent temporal curve',
        fontsize=15, fontweight='bold', y=0.98
    )

    draw_constraint_surface(axes[0])
    draw_relational_bundle(axes[1], data)
    draw_bloch_disk(axes[2], data)

    # Connecting arrows between panels
    for i in range(2):
        fig.text(0.345 + i * 0.3, 0.5, r'$\Longrightarrow$',
                 fontsize=22, ha='center', va='center', color='#999')

    # Operation labels between panels
    fig.text(0.345, 0.43,
             r'$\langle t|_C \otimes I_{SE}$',
             fontsize=10, ha='center', va='center', color='#555')
    fig.text(0.645, 0.43,
             r'$\mathrm{Tr}_E$',
             fontsize=10, ha='center', va='center', color='#555')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    outpath = os.path.join(OUTPUT_DIR, 'geometric_interpretation.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {outpath}")


# ══════════════════════════════════════════════════════════════
#  Figure 2: Detailed Bloch trajectory comparison
# ══════════════════════════════════════════════════════════════

def generate_bloch_figure(data):
    """
    Detailed Bloch trajectory: Version A (circle) vs Version B (spiral).
    Two panels: trajectory + Bloch radius decay.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    # ── Bloch disk (y-z plane) ──
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), '-', color='#ddd', linewidth=1.5)
    ax.fill(np.cos(theta), np.sin(theta), color='#f5f5f5', alpha=0.3)
    ax.axhline(0, color='#eee', linewidth=0.8)
    ax.axvline(0, color='#eee', linewidth=0.8)

    # Version A
    ax.plot(data['by_a'], data['bz_a'], '-', color='#4a86c8', linewidth=2,
            alpha=0.5, label=r'Version A: $|\vec{r}|=1$ (pure)')

    # Version B — colored by entropy
    by_b, bz_b = data['by_b'], data['bz_b']
    s_norm = data['s_eff_b'] / np.log(2)

    points = np.array([by_b, bz_b]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='coolwarm', linewidth=3)
    lc.set_array(s_norm[:-1])
    line = ax.add_collection(lc)

    # Colorbar
    cbar = plt.colorbar(line, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(r'$S_{\rm eff} / \ln 2$', fontsize=11)

    # Markers
    ax.plot(by_b[0], bz_b[0], 'D', color='#1a7f37', markersize=11, zorder=6,
            markeredgecolor='white', markeredgewidth=2, label='k = 0')
    ax.plot(by_b[-1], bz_b[-1], 's', color='#d4380d', markersize=11, zorder=6,
            markeredgecolor='white', markeredgewidth=2, label=f'k = {N-1}')
    ax.plot(0, 0, '+', color='black', markersize=15, markeredgewidth=2.5,
            zorder=5, alpha=0.4)
    ax.text(0.08, -0.13, r'$I/2$', fontsize=11, color='#666')

    ax.set_xlabel(r'$\langle\sigma_y\rangle$', fontsize=12)
    ax.set_ylabel(r'$\langle\sigma_z\rangle$', fontsize=12)
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect('equal')
    ax.legend(fontsize=10, loc='lower left', framealpha=0.9)
    ax.set_title('Bloch disk (y–z plane)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.15)

    # ── Panel 2: Bloch radius + entropy vs k ──
    ax2 = axes[1]
    ks = np.arange(N)

    color_r = '#4a86c8'
    color_s = '#d4380d'

    ax2.plot(ks, data['bloch_radius_a'], '--', color=color_r, alpha=0.4,
             linewidth=1.5, label=r'$|\vec{r}|$ Version A')
    ax2.plot(ks, data['bloch_radius_b'], 'o-', color=color_r, markersize=4,
             linewidth=2, label=r'$|\vec{r}|$ Version B')
    ax2.set_xlabel('Clock reading k', fontsize=12)
    ax2.set_ylabel(r'Bloch radius $|\vec{r}|$', fontsize=12, color=color_r)
    ax2.tick_params(axis='y', labelcolor=color_r)
    ax2.set_ylim(-0.05, 1.15)

    ax2b = ax2.twinx()
    ax2b.plot(ks, data['s_eff_b'], 's-', color=color_s, markersize=4,
              linewidth=2, label=r'$S_{\rm eff}$')
    ax2b.axhline(np.log(2), color=color_s, linestyle=':', alpha=0.4)
    ax2b.set_ylabel(r'$S_{\rm eff}$', fontsize=12, color=color_s)
    ax2b.tick_params(axis='y', labelcolor=color_s)
    ax2b.set_ylim(-0.05, 0.85)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
               loc='center right', framealpha=0.9)

    ax2.set_title(r'Purity decay $\leftrightarrow$ entropy growth',
                  fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.2)

    # Annotations
    ax2.annotate(r'Pure ($|\vec{r}|=1$)', xy=(0, 1.0),
                 xytext=(5, 1.08), fontsize=9, color=color_r,
                 arrowprops=dict(arrowstyle='->', color=color_r, lw=1))
    ax2.annotate(r'Mixed ($|\vec{r}|\to 0$)', xy=(29, data['bloch_radius_b'][-1]),
                 xytext=(20, 0.15), fontsize=9, color=color_r,
                 arrowprops=dict(arrowstyle='->', color=color_r, lw=1))

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'bloch_trajectory.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {outpath}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Computing Bloch trajectories (Version A & B)...")
    data = compute_bloch_trajectories()

    print(f"  Version A: |r| = {data['bloch_radius_a'][0]:.4f} → "
          f"{data['bloch_radius_a'][-1]:.4f}  (stays on surface)")
    print(f"  Version B: |r| = {data['bloch_radius_b'][0]:.4f} → "
          f"{data['bloch_radius_b'][-1]:.4f}  (spirals inward)")
    print(f"  S_eff:     0.000 → {data['s_eff_b'][-1]:.4f}  (ln 2 = {np.log(2):.4f})")

    print("\nGenerating geometric interpretation figure...")
    generate_main_figure(data)

    print("Generating Bloch trajectory figure...")
    generate_bloch_figure(data)

    print("\nGeometric interpretation:")
    print("  • |Ψ⟩ is a fixed point on the constraint surface Ĉ=0 in CP(H)")
    print("  • The projection ⟨k|_C creates a bundle over clock readings")
    print("  • Tr_E maps the section into D(H_S) — the arrow emerges")
    print("  • Version A: |r|=1 → circle on Bloch surface (reversible)")
    print(f"  • Version B: |r|→{data['bloch_radius_b'][-1]:.3f} "
          f"→ spiral toward I/2 (irreversible)")
    print("\nDone.")
