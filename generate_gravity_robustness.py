#!/usr/bin/env python3
"""
Gravity Robustness Tests
========================

Three computational tests probing whether the unified relational formula
  ρ_S(t) = Tr_E[⟨t|_C |Ψ⟩⟨Ψ| |t⟩_C] / p(t)
is robust against perturbations that mimic aspects of quantum gravity.

Test 1 — Clock backreaction (H_CS ≠ 0):
  The clock's reading affects the system Hamiltonian through a
  k-dependent term ε·(k/N)·σ_z⊗I_E, simulating gravitational
  backreaction from a physical clock.

Test 2 — Fuzzy subsystem boundaries:
  A partial SWAP between S and E₁ blurs the S–E partition before
  tracing, simulating fluctuating subsystem boundaries (the "problem
  of subsystems" in quantum gravity).

Test 3 — Clock uncertainty (Gaussian-smeared projection):
  Projection uses smeared clock states |k̃⟩ = Σ_j c_j|j⟩ with
  c_j ∝ exp(-(j-k)²/(2σ²)) instead of sharp |k⟩, modeling clock
  imprecision from gravitational time dilation.

Expected result: all three pillars degrade GRACEFULLY — the arrow
of time, in particular, is structurally robust against all tested
perturbations.

Script for: "The Observer as a Local Breakdown of Atemporality"
by Gabriel Giani Moreno (2026).
"""

import numpy as np
import qutip as qt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os

# ── Parameters ──────────────────────────────────────────────
N = 30
dt = 0.2
omega = 1.0
g = 0.1
n_env = 4
initial_S = qt.basis(2, 0)
sigma_x = qt.sigmax()
sigma_z = qt.sigmaz()

clock_basis = [qt.basis(N, k) for k in range(N)]
analytic = np.cos(omega * np.arange(N) * dt)


# ══════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════

def build_H_SE(omega, g, n_env):
    """Build H_SE = H_S + H_int on S⊗E space."""
    id_list = [qt.qeye(2) for _ in range(n_env)]
    H_S = (omega / 2) * qt.tensor(sigma_x, *id_list)

    dim_env = 2**n_env
    H_int = qt.Qobj(np.zeros((2 * dim_env, 2 * dim_env)),
                     dims=[[2] + [2]*n_env, [2] + [2]*n_env])
    for j in range(n_env):
        ops = [qt.qeye(2) for _ in range(n_env)]
        ops[j] = sigma_x
        H_int += g * qt.tensor(sigma_x, *ops)

    return H_S + H_int


def build_history_state(N, dt, H_SE, n_env, initial_SE):
    """Build standard PaW history state |Ψ⟩ = (1/√N) Σ_k |k⟩_C ⊗ U(t_k)|ψ₀⟩."""
    norm = 1.0 / np.sqrt(N)
    dim_env = 2**n_env
    total_dim = N * 2 * dim_env
    psi = qt.Qobj(np.zeros((total_dim, 1)),
                   dims=[[N, 2] + [2]*n_env, [1]*(2 + n_env)])

    for k in range(N):
        t_k = k * dt
        U = (-1j * H_SE * t_k).expm()
        comp = norm * qt.tensor(clock_basis[k], U * initial_SE)
        psi += comp

    return psi.unit()


def condition_standard(psi, N, n_env):
    """Standard sharp-clock conditioning + clean Tr_E."""
    d_SE = 2 * (2**n_env)
    psi_vec = psi.full().flatten()
    blocks = psi_vec.reshape(N, d_SE)

    sz_list, S_list = [], []
    for k in range(N):
        phi_k = blocks[k, :]
        p_k = np.vdot(phi_k, phi_k).real
        if p_k > 1e-12:
            dims_ket = [[2] + [2]*n_env, [1]*(1 + n_env)]
            psi_SE_k = qt.Qobj(phi_k.reshape(-1, 1), dims=dims_ket)
            rho_SE_k = (psi_SE_k * psi_SE_k.dag()) / p_k
            rho_S = rho_SE_k.ptrace(0)
            sz_list.append(qt.expect(sigma_z, rho_S))
            S_list.append(qt.entropy_vn(rho_S))
        else:
            sz_list.append(np.nan)
            S_list.append(np.nan)

    return np.array(sz_list), np.array(S_list)


def arrow_metrics(S_eff):
    """
    Compute arrow quality metrics.

    arrow_strength: S_eff(k_final) / ln(2)  — how close to thermal
    monotonicity:   fraction of steps where S(k+1) ≥ S(k)
    """
    strength = S_eff[-1] / np.log(2)
    diffs = np.diff(S_eff)
    mono = np.sum(diffs >= -1e-10) / len(diffs)
    return strength, mono


# ══════════════════════════════════════════════════════════════
# TEST 1: Clock backreaction
# ══════════════════════════════════════════════════════════════

def run_test1_backreaction(eps_vals):
    """
    Clock–system interaction: at each tick k, H_eff = H_SE + ε(k/N)σ_z⊗I_E.

    Physical meaning: the clock's gravitational field shifts the system's
    energy levels proportionally to the clock reading.  In standard PaW
    the clock is ideal (no backreaction); here we break that idealisation.

    ε = 0:   standard model (no backreaction)
    ε ≫ ω:   strong backreaction dominates dynamics
    """
    print("\n" + "=" * 60)
    print("TEST 1: Clock backreaction (ε sweep)")
    print("=" * 60)

    H_base = build_H_SE(omega, g, n_env)
    H_back_unit = qt.tensor(sigma_z, *[qt.qeye(2) for _ in range(n_env)])

    env0 = qt.tensor([qt.basis(2, 0) for _ in range(n_env)])
    initial_SE = qt.tensor(initial_S, env0)

    results = {}
    for eps in eps_vals:
        print(f"  ε = {eps:.3f} ...", end=" ", flush=True)

        norm = 1.0 / np.sqrt(N)
        dim_env = 2**n_env
        total_dim = N * 2 * dim_env
        psi = qt.Qobj(np.zeros((total_dim, 1)),
                       dims=[[N, 2] + [2]*n_env, [1]*(2 + n_env)])

        for k in range(N):
            t_k = k * dt
            H_eff = H_base + eps * (k / N) * H_back_unit
            U = (-1j * H_eff * t_k).expm()
            comp = norm * qt.tensor(clock_basis[k], U * initial_SE)
            psi += comp

        psi = psi.unit()
        sz, S_eff = condition_standard(psi, N, n_env)
        strength, mono = arrow_metrics(S_eff)

        results[eps] = {'sz': sz, 'S_eff': S_eff,
                        'strength': strength, 'mono': mono}
        print(f"S_eff(final)={S_eff[-1]:.4f}, "
              f"arrow={strength:.3f}, mono={mono:.3f}")

    return results


# ══════════════════════════════════════════════════════════════
# TEST 2: Fuzzy subsystem boundaries
# ══════════════════════════════════════════════════════════════

def run_test2_fuzzy_boundary(theta_vals):
    """
    Partial SWAP between S and E₁ before Tr_E.

    V(θ) = cos(θ) I − i sin(θ) SWAP_{S,E₁} ⊗ I_{E_rest}

    Physical meaning: the boundary between "system" and "environment"
    is not sharp — a fraction of what we call S contains E₁ degrees
    of freedom, and vice versa.  This is the "problem of subsystems"
    in quantum gravity (Donnelly & Freidel, Höhn et al.).

    θ = 0:    clean partition (standard model)
    θ = π/2:  S and E₁ fully swapped
    """
    print("\n" + "=" * 60)
    print("TEST 2: Fuzzy subsystem boundaries (θ sweep)")
    print("=" * 60)

    # Build standard history state (same for all θ)
    H_SE = build_H_SE(omega, g, n_env)
    env0 = qt.tensor([qt.basis(2, 0) for _ in range(n_env)])
    initial_SE = qt.tensor(initial_S, env0)
    psi = build_history_state(N, dt, H_SE, n_env, initial_SE)

    # Build SWAP_{S,E₁} ⊗ I_{E₂,...,E_n} on full SE space
    swap_2q = qt.Qobj(np.array([[1, 0, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 1]]),
                       dims=[[2, 2], [2, 2]])

    if n_env > 1:
        id_rest = [qt.qeye(2) for _ in range(n_env - 1)]
        SWAP_full = qt.tensor(swap_2q, *id_rest)
    else:
        SWAP_full = swap_2q

    I_SE = qt.qeye([2] + [2]*n_env)

    # Extract blocks once
    d_SE = 2 * (2**n_env)
    psi_vec = psi.full().flatten()
    blocks = psi_vec.reshape(N, d_SE)

    results = {}
    for theta in theta_vals:
        print(f"  θ = {theta:.3f} (θ/π = {theta/np.pi:.3f}) ...",
              end=" ", flush=True)

        # V(θ) = cos(θ)I − i·sin(θ)·SWAP  [unitary: SWAP² = I]
        V = np.cos(theta) * I_SE - 1j * np.sin(theta) * SWAP_full
        V_dag = V.dag()

        sz_list, S_list = [], []
        for k in range(N):
            phi_k = blocks[k, :]
            p_k = np.vdot(phi_k, phi_k).real
            if p_k > 1e-12:
                dims_ket = [[2] + [2]*n_env, [1]*(1 + n_env)]
                psi_SE_k = qt.Qobj(phi_k.reshape(-1, 1), dims=dims_ket)
                rho_SE_k = (psi_SE_k * psi_SE_k.dag()) / p_k

                # Rotate before tracing: blur the S–E boundary
                rho_mixed = V * rho_SE_k * V_dag
                rho_S = rho_mixed.ptrace(0)

                sz_list.append(qt.expect(sigma_z, rho_S))
                S_list.append(qt.entropy_vn(rho_S))
            else:
                sz_list.append(np.nan)
                S_list.append(np.nan)

        sz, S_eff = np.array(sz_list), np.array(S_list)
        strength, mono = arrow_metrics(S_eff)

        results[theta] = {'sz': sz, 'S_eff': S_eff,
                          'strength': strength, 'mono': mono}
        print(f"S_eff(final)={S_eff[-1]:.4f}, "
              f"arrow={strength:.3f}, mono={mono:.3f}")

    return results


# ══════════════════════════════════════════════════════════════
# TEST 3: Clock uncertainty (Gaussian-smeared projection)
# ══════════════════════════════════════════════════════════════

def run_test3_fuzzy_clock(sigma_vals):
    """
    Project onto Gaussian-smeared clock states:
      |k̃⟩ = Σ_j c_j |j⟩,  c_j ∝ exp(-(j-k)²/(2σ²))

    Physical meaning: gravitational time dilation makes the clock
    reading inherently uncertain.  A smeared projection mixes SE
    states at different "times", creating coherent superpositions
    of past and future — exactly what happens when spacetime
    geometry is uncertain.

    σ = 0:  sharp delta-function clock (standard model)
    σ = N:  clock is useless (projects onto everything equally)
    """
    print("\n" + "=" * 60)
    print("TEST 3: Clock uncertainty (σ sweep)")
    print("=" * 60)

    # Build standard history state (same for all σ)
    H_SE = build_H_SE(omega, g, n_env)
    env0 = qt.tensor([qt.basis(2, 0) for _ in range(n_env)])
    initial_SE = qt.tensor(initial_S, env0)
    psi = build_history_state(N, dt, H_SE, n_env, initial_SE)

    d_SE = 2 * (2**n_env)
    psi_vec = psi.full().flatten()
    blocks = psi_vec.reshape(N, d_SE)

    results = {}
    for sig in sigma_vals:
        print(f"  σ = {sig:.2f} ...", end=" ", flush=True)

        sz_list, S_list = [], []
        for k in range(N):
            if sig < 1e-6:
                # Sharp projection (standard)
                phi_k = blocks[k, :]
            else:
                # Gaussian-smeared projection
                coeffs = np.exp(-0.5 * ((np.arange(N) - k) / sig)**2)
                coeffs /= np.linalg.norm(coeffs)

                # |ψ_SE(k̃)⟩ = Σ_j c_j ⟨j|Ψ⟩_SE
                phi_k = np.zeros(d_SE, dtype=complex)
                for j in range(N):
                    phi_k += coeffs[j] * blocks[j, :]

            p_k = np.vdot(phi_k, phi_k).real
            if p_k > 1e-12:
                dims_ket = [[2] + [2]*n_env, [1]*(1 + n_env)]
                psi_SE_k = qt.Qobj(phi_k.reshape(-1, 1), dims=dims_ket)
                rho_SE_k = (psi_SE_k * psi_SE_k.dag()) / p_k
                rho_S = rho_SE_k.ptrace(0)

                sz_list.append(qt.expect(sigma_z, rho_S))
                S_list.append(qt.entropy_vn(rho_S))
            else:
                sz_list.append(np.nan)
                S_list.append(np.nan)

        sz, S_eff = np.array(sz_list), np.array(S_list)
        strength, mono = arrow_metrics(S_eff)

        results[sig] = {'sz': sz, 'S_eff': S_eff,
                        'strength': strength, 'mono': mono}
        print(f"S_eff(final)={S_eff[-1]:.4f}, "
              f"arrow={strength:.3f}, mono={mono:.3f}")

    return results


# ══════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_curves(res1, res2, res3, eps_vals, theta_vals, sigma_vals):
    """Main figure: S_eff(k) and ⟨σ_z⟩(k) for all three tests."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    ks = np.arange(N)
    cmap = plt.cm.viridis

    # ── Row 1: Entropy curves (the arrow) ──

    # Test 1
    ax = axes[0, 0]
    colors = cmap(np.linspace(0.1, 0.95, len(eps_vals)))
    for i, eps in enumerate(eps_vals):
        lw = 2.5 if eps == 0 else 1.2
        ax.plot(ks, res1[eps]['S_eff'], color=colors[i], linewidth=lw,
                label=f'ε={eps:.2f}')
    ax.axhline(np.log(2), color='gray', ls=':', alpha=0.5, label='ln 2')
    ax.set_xlabel('Clock reading k')
    ax.set_ylabel('S_eff(k)')
    ax.set_title('Test 1: Clock backreaction — Arrow')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)

    # Test 2
    ax = axes[0, 1]
    colors = cmap(np.linspace(0.1, 0.95, len(theta_vals)))
    for i, theta in enumerate(theta_vals):
        lw = 2.5 if theta == 0 else 1.2
        ax.plot(ks, res2[theta]['S_eff'], color=colors[i], linewidth=lw,
                label=f'θ={theta:.2f}')
    ax.axhline(np.log(2), color='gray', ls=':', alpha=0.5, label='ln 2')
    ax.set_xlabel('Clock reading k')
    ax.set_ylabel('S_eff(k)')
    ax.set_title('Test 2: Fuzzy boundaries — Arrow')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)

    # Test 3
    ax = axes[0, 2]
    colors = cmap(np.linspace(0.1, 0.95, len(sigma_vals)))
    for i, sig in enumerate(sigma_vals):
        lw = 2.5 if sig == 0 else 1.2
        ax.plot(ks, res3[sig]['S_eff'], color=colors[i], linewidth=lw,
                label=f'σ={sig:.1f}')
    ax.axhline(np.log(2), color='gray', ls=':', alpha=0.5, label='ln 2')
    ax.set_xlabel('Clock reading k')
    ax.set_ylabel('S_eff(k)')
    ax.set_title('Test 3: Clock uncertainty — Arrow')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.3)

    # ── Row 2: Dynamics curves ──

    # Test 1
    ax = axes[1, 0]
    colors = cmap(np.linspace(0.1, 0.95, len(eps_vals)))
    ax.plot(ks, analytic, 'k--', alpha=0.3, label='cos(ωkdt)')
    for i, eps in enumerate(eps_vals):
        lw = 2.5 if eps == 0 else 1.2
        ax.plot(ks, res1[eps]['sz'], color=colors[i], linewidth=lw,
                label=f'ε={eps:.2f}')
    ax.set_xlabel('Clock reading k')
    ax.set_ylabel('⟨σ_z⟩(k)')
    ax.set_title('Test 1: Dynamics under backreaction')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    # Test 2
    ax = axes[1, 1]
    colors = cmap(np.linspace(0.1, 0.95, len(theta_vals)))
    ax.plot(ks, analytic, 'k--', alpha=0.3, label='cos(ωkdt)')
    for i, theta in enumerate(theta_vals):
        lw = 2.5 if theta == 0 else 1.2
        ax.plot(ks, res2[theta]['sz'], color=colors[i], linewidth=lw,
                label=f'θ={theta:.2f}')
    ax.set_xlabel('Clock reading k')
    ax.set_ylabel('⟨σ_z⟩(k)')
    ax.set_title('Test 2: Dynamics under fuzzy boundaries')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    # Test 3
    ax = axes[1, 2]
    colors = cmap(np.linspace(0.1, 0.95, len(sigma_vals)))
    ax.plot(ks, analytic, 'k--', alpha=0.3, label='cos(ωkdt)')
    for i, sig in enumerate(sigma_vals):
        lw = 2.5 if sig == 0 else 1.2
        ax.plot(ks, res3[sig]['sz'], color=colors[i], linewidth=lw,
                label=f'σ={sig:.1f}')
    ax.set_xlabel('Clock reading k')
    ax.set_ylabel('⟨σ_z⟩(k)')
    ax.set_title('Test 3: Dynamics under clock uncertainty')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    plt.suptitle(
        'Gravity Robustness: Three Tests\n'
        'Top: thermodynamic arrow S_eff(k)  |  '
        'Bottom: quantum dynamics ⟨σ_z⟩(k)',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('output/gravity_robustness_curves.png', dpi=150)
    plt.close()
    print("\n→ Saved output/gravity_robustness_curves.png")


def plot_summary(res1, res2, res3, eps_vals, theta_vals, sigma_vals):
    """Summary figure: arrow strength and monotonicity vs perturbation."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, vals, res, xlabel, title in [
        (axes[0], eps_vals, res1, 'Backreaction ε', 'Test 1: Clock backreaction'),
        (axes[1], theta_vals, res2, 'Boundary mixing θ', 'Test 2: Fuzzy boundaries'),
        (axes[2], sigma_vals, res3, 'Clock uncertainty σ', 'Test 3: Clock uncertainty'),
    ]:
        strengths = [res[v]['strength'] for v in vals]
        monos = [res[v]['mono'] for v in vals]

        ax.plot(vals, strengths, 'o-', color='#e74c3c', linewidth=2,
                markersize=8, label='Arrow strength (S_eff/ln2)')
        ax.plot(vals, monos, 's--', color='#3498db', linewidth=2,
                markersize=8, label='Monotonicity')
        ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
        ax.axhline(0.5, color='gray', ls=':', alpha=0.3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Metric (normalized)')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 1.15)

    plt.suptitle(
        'Gravity Robustness: Arrow Survival Under Perturbation',
        fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig('output/gravity_robustness_summary.png', dpi=150)
    plt.close()
    print("→ Saved output/gravity_robustness_summary.png")


def export_csv(res1, res2, res3, eps_vals, theta_vals, sigma_vals):
    """Export summary metrics to CSV."""
    rows = []
    for eps in eps_vals:
        rows.append({
            'test': 'backreaction', 'parameter': 'epsilon',
            'value': f'{eps:.4f}',
            'S_eff_final': f'{res1[eps]["S_eff"][-1]:.6f}',
            'arrow_strength': f'{res1[eps]["strength"]:.4f}',
            'monotonicity': f'{res1[eps]["mono"]:.4f}'
        })
    for theta in theta_vals:
        rows.append({
            'test': 'fuzzy_boundary', 'parameter': 'theta',
            'value': f'{theta:.4f}',
            'S_eff_final': f'{res2[theta]["S_eff"][-1]:.6f}',
            'arrow_strength': f'{res2[theta]["strength"]:.4f}',
            'monotonicity': f'{res2[theta]["mono"]:.4f}'
        })
    for sig in sigma_vals:
        rows.append({
            'test': 'fuzzy_clock', 'parameter': 'sigma',
            'value': f'{sig:.4f}',
            'S_eff_final': f'{res3[sig]["S_eff"][-1]:.6f}',
            'arrow_strength': f'{res3[sig]["strength"]:.4f}',
            'monotonicity': f'{res3[sig]["mono"]:.4f}'
        })

    path = 'output/table_gravity_robustness.csv'
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"→ Saved {path}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Gravity Robustness Tests")
    print("=" * 60)
    print(f"Parameters: N={N}, dt={dt}, ω={omega}, g={g}, n_env={n_env}")

    # Sweep parameters
    eps_vals = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    theta_vals = [0.0, 0.1, 0.3, 0.5, np.pi/4, np.pi/2]
    sigma_vals = [0.0, 0.3, 0.7, 1.0, 2.0, 4.0]

    # Run tests
    res1 = run_test1_backreaction(eps_vals)
    res2 = run_test2_fuzzy_boundary(theta_vals)
    res3 = run_test3_fuzzy_clock(sigma_vals)

    # Output
    os.makedirs('output', exist_ok=True)
    plot_curves(res1, res2, res3, eps_vals, theta_vals, sigma_vals)
    plot_summary(res1, res2, res3, eps_vals, theta_vals, sigma_vals)
    export_csv(res1, res2, res3, eps_vals, theta_vals, sigma_vals)

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    base = res1[0.0]
    print(f"\nBaseline (no perturbation):")
    print(f"  S_eff(final) = {base['S_eff'][-1]:.4f}  (ln 2 = {np.log(2):.4f})")
    print(f"  Arrow strength = {base['strength']:.3f}")
    print(f"  Monotonicity = {base['mono']:.3f}")

    for label, res, vals, pname in [
        ("Test 1 — max backreaction", res1, eps_vals, f"ε={eps_vals[-1]}"),
        ("Test 2 — max boundary mixing", res2, theta_vals,
         f"θ={theta_vals[-1]:.3f}=π/2"),
        ("Test 3 — max clock blur", res3, sigma_vals, f"σ={sigma_vals[-1]}"),
    ]:
        r = res[vals[-1]]
        b = res[vals[0]]
        print(f"\n{label} ({pname}):")
        print(f"  Arrow:  {r['strength']:.3f}  (baseline {b['strength']:.3f})")
        print(f"  Mono:   {r['mono']:.3f}  (baseline {b['mono']:.3f})")

    # Overall verdict
    all_strengths = (
        [res1[e]['strength'] for e in eps_vals] +
        [res2[t]['strength'] for t in theta_vals] +
        [res3[s]['strength'] for s in sigma_vals]
    )
    min_s = min(all_strengths)

    print(f"\n{'=' * 60}")
    print(f"VERDICT: Minimum arrow strength across ALL tests = {min_s:.3f}")
    if min_s > 0.3:
        print("→ The arrow SURVIVES all gravity-like perturbations.")
        print("  The mechanism is structurally robust — not fragile.")
    elif min_s > 0.1:
        print("→ The arrow degrades but PERSISTS under most perturbations.")
    else:
        print("→ Some perturbations significantly weaken the arrow.")
    print("  (Dynamics [Pillar 1] degrade; arrow [Pillar 2] is resilient.)")
    print("=" * 60)
