"""
IBM Quantum Validation of Stability & Uniqueness Theorems
=========================================================

Validates key predictions from stability_uniqueness.tex on IBM Quantum
simulator (and optionally real hardware).

Model: 2-qubit Heisenberg system-environment
  H = B₁σ_z⊗I + B₂I⊗σ_z + λ(σ_x⊗σ_x + σ_y⊗σ_y + σ_z⊗σ_z)
  |ψ₀⟩ = |1⟩_S ⊗ |0⟩_E  (pure product state, NOT an eigenstate)

First-order Trotterization:
  U(dt) ≈ Rz(2B₁dt)₀ · Rz(2B₂dt)₁ · RXX(2λdt)₀₁ · RYY(2λdt)₀₁ · RZZ(2λdt)₀₁

Theorems validated:
  - Thm 3.1: Quadratic MI bound  →  I(S:E) ∝ η²
  - Thm 3.1: Convergence to K    →  I(S:E)/(t²‖H_int‖²) → const
  - Thm 8.1: Almost-unitary      →  purity controlled by η

Measurements: ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ on qubit 0
For pure |ψ(t)⟩: I(S:E) = 2·S(ρ_S) and Purity = (1+|r|²)/2

Usage:
  python run_ibm_stability.py --mode simulator    # Local test (free)
  python run_ibm_stability.py --mode hardware     # Real IBM QPU
  python run_ibm_stability.py --mode both         # Full comparison

Estimated QPU time: ~20 seconds (well within 10 min/month free tier).
"""

import json
import os
import sys
import csv
import argparse
import numpy as np
from scipy.linalg import expm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════
# PHYSICAL PARAMETERS
# ═══════════════════════════════════════════════════════════

B1 = 1.0          # Local field on system
B2 = 0.7          # Local field on environment
DT = 0.1          # Trotter step size
K_MAX = 30        # Trotter steps → t_max = 3.0
N_QUBITS = 2      # 1 system + 1 environment

# λ values for parameter scans (matching verify_corrected.py)
LAMBDAS_TIME = [0.05, 0.1, 0.2, 0.4]
LAMBDAS_SCAN = [0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05,
                0.08, 0.1, 0.15, 0.2, 0.3, 0.4]
T_FIXED = 0.3     # Fixed time for η² scaling test

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'output')

# Pauli matrices
I2 = np.eye(2, dtype=complex)
_sx = np.array([[0, 1], [1, 0]], dtype=complex)
_sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
_sz = np.array([[1, 0], [0, -1]], dtype=complex)


# ═══════════════════════════════════════════════════════════
# API KEY
# ═══════════════════════════════════════════════════════════

def load_api_key():
    """Load IBM Quantum API key from ../apikey.json."""
    key_path = os.path.join(SCRIPT_DIR, '..', 'apikey.json')
    if not os.path.exists(key_path):
        print(f"ERROR: API key not found at {key_path}")
        print("Create apikey.json with: {\"apikey\": \"YOUR_KEY\"}")
        sys.exit(1)
    with open(key_path) as f:
        return json.load(f)['apikey']


# ═══════════════════════════════════════════════════════════
# EXACT NUMPY REFERENCE (no Trotter, no noise)
# ═══════════════════════════════════════════════════════════

def tensor(A, B):
    return np.kron(A, B)


def partial_trace_env(rho_4x4):
    """Trace out environment (qubit 1) from 4×4 density matrix."""
    r = rho_4x4.reshape(2, 2, 2, 2)
    return np.trace(r, axis1=1, axis2=3)


def build_H(b1, b2, lam):
    """Build Heisenberg Hamiltonian: H_loc + H_int."""
    H_loc = b1 * tensor(_sz, I2) + b2 * tensor(I2, _sz)
    H_int = lam * (tensor(_sx, _sx) + tensor(_sy, _sy) + tensor(_sz, _sz))
    return H_loc + H_int, H_loc, H_int


def hs_norm(M):
    """Hilbert-Schmidt norm ‖M‖₂ = √Tr(M†M)."""
    return np.sqrt(np.real(np.trace(M.conj().T @ M)))


def vn_entropy(rho):
    """Von Neumann entropy S(ρ) = -Tr(ρ ln ρ)."""
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-15]
    return -np.sum(evals * np.log(evals))


def compute_exact_bloch(b1, b2, lam, k_max, dt):
    """
    Compute exact Bloch vector on system qubit for |00⟩ initial state.

    Returns (sx, sy, sz, S_eff, purity) arrays of length k_max+1.
    """
    H, _, _ = build_H(b1, b2, lam)
    psi0 = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩

    sx_arr = np.zeros(k_max + 1)
    sy_arr = np.zeros(k_max + 1)
    sz_arr = np.zeros(k_max + 1)
    S_arr = np.zeros(k_max + 1)
    pur_arr = np.zeros(k_max + 1)

    for k in range(k_max + 1):
        t = k * dt
        U = expm(-1j * H * t)
        psi_t = U @ psi0
        rho = np.outer(psi_t, psi_t.conj())
        rho_S = partial_trace_env(rho)

        sx_arr[k] = np.real(np.trace(_sx @ rho_S))
        sy_arr[k] = np.real(np.trace(_sy @ rho_S))
        sz_arr[k] = np.real(np.trace(_sz @ rho_S))
        S_arr[k] = vn_entropy(rho_S)
        pur_arr[k] = np.real(np.trace(rho_S @ rho_S))

    return sx_arr, sy_arr, sz_arr, S_arr, pur_arr


def compute_exact_MI_at_t(b1, b2, lam, t):
    """Compute mutual information at fixed time t for pure |10⟩."""
    H, _, _ = build_H(b1, b2, lam)
    psi0 = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
    U = expm(-1j * H * t)
    psi_t = U @ psi0
    rho = np.outer(psi_t, psi_t.conj())
    rho_S = partial_trace_env(rho)
    S = vn_entropy(rho_S)
    return 2 * S  # I(S:E) = 2·S(ρ_S) for pure global state


# ═══════════════════════════════════════════════════════════
# CIRCUIT CONSTRUCTION (Trotterized Heisenberg)
# ═══════════════════════════════════════════════════════════

def build_trotter_circuit(b1, b2, lam, k, dt):
    """
    Build k-step first-order Trotter circuit for Heisenberg H.

    Each step applies:
      1. Rz(2·B₁·dt) on q0    — local field on system
      2. Rz(2·B₂·dt) on q1    — local field on environment
      3. RXX(2·λ·dt) on q0,q1 — Heisenberg XX coupling
      4. RYY(2·λ·dt) on q0,q1 — Heisenberg YY coupling
      5. RZZ(2·λ·dt) on q0,q1 — Heisenberg ZZ coupling

    Gate count per step: 2 single-qubit + 3 two-qubit.
    For k=30: ~90 CX after decomposition — within coherence limits.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(N_QUBITS)
    qc.x(0)  # Prepare |10⟩ = |1⟩_S ⊗ |0⟩_E

    for _ in range(k):
        # Local terms
        qc.rz(2 * b1 * dt, 0)
        qc.rz(2 * b2 * dt, 1)
        # Heisenberg interaction: σ_x⊗σ_x + σ_y⊗σ_y + σ_z⊗σ_z
        qc.rxx(2 * lam * dt, 0, 1)
        qc.ryy(2 * lam * dt, 0, 1)
        qc.rzz(2 * lam * dt, 0, 1)

    return qc


def build_observables():
    """
    Build Pauli observables for partial tomography of qubit 0.

    Qiskit convention: rightmost character = qubit 0.
    So 'IX' means σ_x on q0, identity on q1.
    """
    from qiskit.quantum_info import SparsePauliOp

    return [
        SparsePauliOp('IX'),   # ⟨σ_x⟩ on system qubit
        SparsePauliOp('IY'),   # ⟨σ_y⟩ on system qubit
        SparsePauliOp('IZ'),   # ⟨σ_z⟩ on system qubit
    ]


# ═══════════════════════════════════════════════════════════
# ENTROPY / PURITY FROM BLOCH VECTOR
# ═══════════════════════════════════════════════════════════

def entropy_from_bloch(sx, sy, sz):
    """
    Von Neumann entropy from Bloch vector.

    ρ = (I + r·σ)/2, eigenvalues λ± = (1 ± |r|)/2.
    S = -λ₊ ln λ₊ - λ₋ ln λ₋
    """
    r = np.sqrt(np.array(sx)**2 + np.array(sy)**2 + np.array(sz)**2)
    r = np.clip(r, 0, 1)
    lp = (1 + r) / 2
    lm = (1 - r) / 2
    S = np.zeros_like(r)
    mask_p = lp > 1e-15
    mask_m = lm > 1e-15
    S[mask_p] -= lp[mask_p] * np.log(lp[mask_p])
    S[mask_m] -= lm[mask_m] * np.log(lm[mask_m])
    return S


def purity_from_bloch(sx, sy, sz):
    """Purity Tr(ρ²) = (1 + |r|²)/2 from Bloch vector."""
    r2 = np.array(sx)**2 + np.array(sy)**2 + np.array(sz)**2
    return (1 + r2) / 2


# ═══════════════════════════════════════════════════════════
# LOCAL SIMULATOR
# ═══════════════════════════════════════════════════════════

def run_on_simulator(circuits, obs_list):
    """
    Run on Qiskit's local StatevectorEstimator (exact Trotter, no noise).

    This verifies the Trotter circuits before using QPU minutes.
    """
    from qiskit.primitives import StatevectorEstimator

    estimator = StatevectorEstimator()
    pubs = [(circ, obs_list) for circ in circuits]
    job = estimator.run(pubs)
    result = job.result()

    sx, sy, sz = [], [], []
    for pub_result in result:
        evs = pub_result.data.evs
        sx.append(float(evs[0]))
        sy.append(float(evs[1]))
        sz.append(float(evs[2]))

    return np.array(sx), np.array(sy), np.array(sz)


# ═══════════════════════════════════════════════════════════
# IBM QUANTUM HARDWARE
# ═══════════════════════════════════════════════════════════

def run_on_hardware(circuits, obs_list, shots=4096):
    """
    Run on IBM Quantum hardware via EstimatorV2.

    Uses the least busy available backend with ≥2 qubits.
    Transpiles circuits to ISA, maps observables to physical layout.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
    from qiskit.transpiler import generate_preset_pass_manager

    api_key = load_api_key()

    print("    Connecting to IBM Quantum...")
    service = QiskitRuntimeService(
        channel='ibm_quantum_platform',
        token=api_key
    )

    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=N_QUBITS
    )
    print(f"    Backend: {backend.name} ({backend.num_qubits} qubits)")

    # Transpile all circuits to ISA
    print(f"    Transpiling {len(circuits)} circuits...")
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    isa_circuits = pm.run(circuits)

    # Map observables to physical qubit layout
    pubs = []
    for isa_circ in isa_circuits:
        isa_obs = [ob.apply_layout(isa_circ.layout) for ob in obs_list]
        pubs.append((isa_circ, isa_obs))

    # Report circuit stats
    last_circ = isa_circuits[-1]
    ops = last_circ.count_ops()
    print(f"    Deepest circuit: {ops}")

    # Submit job
    estimator = EstimatorV2(mode=backend)
    print(f"    Submitting {len(pubs)} PUBs (3 observables each)...")
    job = estimator.run(pubs)
    print(f"    Job ID: {job.job_id()}")
    print("    Waiting for results (queue + execution)...")
    print("    (QPU budget: ~20s of the 600s monthly quota)")

    result = job.result()
    print("    Results received!")

    sx, sy, sz = [], [], []
    for pub_result in result:
        evs = pub_result.data.evs
        sx.append(float(evs[0]))
        sy.append(float(evs[1]))
        sz.append(float(evs[2]))

    return np.array(sx), np.array(sy), np.array(sz), backend.name


# ═══════════════════════════════════════════════════════════
# VALIDATION: TIME EVOLUTION (Panels A, B, D, F)
# ═══════════════════════════════════════════════════════════

def validate_time_evolution(mode, shots):
    """
    Run time evolution for multiple λ values.

    Returns dict: {lam: {eta, exact: {sx,sy,sz,S,pur,MI}, sim: {...}, hw: {...}}}
    """
    obs_list = build_observables()
    results = {}
    hw_backend = None

    for lam in LAMBDAS_TIME:
        H, _, H_int = build_H(B1, B2, lam)
        eta = hs_norm(H_int) / hs_norm(H)
        print(f"\n  λ = {lam}, η = {eta:.4f}")

        # Exact numpy reference
        sx_ex, sy_ex, sz_ex, S_ex, pur_ex = compute_exact_bloch(
            B1, B2, lam, K_MAX, DT)
        MI_ex = 2 * S_ex  # Pure state: I(S:E) = 2·S(ρ_S)

        entry = {
            'eta': eta,
            'exact': {
                'sx': sx_ex, 'sy': sy_ex, 'sz': sz_ex,
                'S': S_ex, 'pur': pur_ex, 'MI': MI_ex
            }
        }

        # Build Trotter circuits for k = 0 .. K_MAX
        circuits = [build_trotter_circuit(B1, B2, lam, k, DT)
                     for k in range(K_MAX + 1)]

        if mode in ('simulator', 'both'):
            print(f"    Simulator ({len(circuits)} circuits)...")
            sim_sx, sim_sy, sim_sz = run_on_simulator(circuits, obs_list)
            S_sim = entropy_from_bloch(sim_sx, sim_sy, sim_sz)
            pur_sim = purity_from_bloch(sim_sx, sim_sy, sim_sz)
            entry['sim'] = {
                'sx': sim_sx, 'sy': sim_sy, 'sz': sim_sz,
                'S': S_sim, 'pur': pur_sim, 'MI': 2 * S_sim
            }

        if mode in ('hardware', 'both'):
            print(f"    Hardware ({len(circuits)} circuits)...")
            hw_sx, hw_sy, hw_sz, hw_backend = run_on_hardware(
                circuits, obs_list, shots)
            S_hw = entropy_from_bloch(hw_sx, hw_sy, hw_sz)
            pur_hw = purity_from_bloch(hw_sx, hw_sy, hw_sz)
            entry['hw'] = {
                'sx': hw_sx, 'sy': hw_sy, 'sz': hw_sz,
                'S': S_hw, 'pur': pur_hw, 'MI': 2 * S_hw
            }
            entry['hw_backend'] = hw_backend

        results[lam] = entry

    return results, hw_backend


# ═══════════════════════════════════════════════════════════
# VALIDATION: η² SCALING (Panel C)
# ═══════════════════════════════════════════════════════════

def validate_eta_scaling(mode, shots):
    """
    Validate I(S:E) ∝ η² at fixed time t = T_FIXED.

    Returns dict with etas, MIs_exact, purity_deficits_exact, MIs_sim, etc.
    """
    obs_list = build_observables()
    k_fixed = int(round(T_FIXED / DT))

    etas = []
    MIs_exact = []
    pds_exact = []   # purity deficit = 1 - Tr(ρ²)
    MIs_sim = []
    pds_sim = []
    MIs_hw = []
    pds_hw = []
    hw_backend = None

    # Build all circuits at once for batching
    circuits = []
    for lam in LAMBDAS_SCAN:
        circuits.append(build_trotter_circuit(B1, B2, lam, k_fixed, DT))

    # Exact reference
    for lam in LAMBDAS_SCAN:
        H, _, H_int = build_H(B1, B2, lam)
        eta = hs_norm(H_int) / hs_norm(H)
        etas.append(eta)
        MIs_exact.append(compute_exact_MI_at_t(B1, B2, lam, T_FIXED))
        # Exact purity deficit via single-step compute_exact_bloch
        sx_a, sy_a, sz_a, _, pur_a = compute_exact_bloch(
            B1, B2, lam, k_fixed, DT)
        pds_exact.append(1.0 - pur_a[k_fixed])

    # Simulator
    if mode in ('simulator', 'both'):
        print(f"    Simulator ({len(circuits)} circuits)...")
        sim_sx, sim_sy, sim_sz = run_on_simulator(circuits, obs_list)
        S_sim = entropy_from_bloch(sim_sx, sim_sy, sim_sz)
        MIs_sim = list(2 * S_sim)
        pur_sim = purity_from_bloch(sim_sx, sim_sy, sim_sz)
        pds_sim = list(1.0 - pur_sim)

    # Hardware
    if mode in ('hardware', 'both'):
        print(f"    Hardware ({len(circuits)} circuits)...")
        hw_sx, hw_sy, hw_sz, hw_backend = run_on_hardware(
            circuits, obs_list, shots)
        S_hw = entropy_from_bloch(hw_sx, hw_sy, hw_sz)
        MIs_hw = list(2 * S_hw)
        pur_hw = purity_from_bloch(hw_sx, hw_sy, hw_sz)
        pds_hw = list(1.0 - pur_hw)

    return {
        'etas': np.array(etas),
        'MIs_exact': np.array(MIs_exact),
        'pds_exact': np.array(pds_exact),
        'MIs_sim': np.array(MIs_sim) if MIs_sim else None,
        'pds_sim': np.array(pds_sim) if pds_sim else None,
        'MIs_hw': np.array(MIs_hw) if MIs_hw else None,
        'pds_hw': np.array(pds_hw) if pds_hw else None,
        'hw_backend': hw_backend
    }


# ═══════════════════════════════════════════════════════════
# PLOTTING (6-panel publication figure)
# ═══════════════════════════════════════════════════════════

def plot_results(time_results, scaling_results, mode):
    """Generate 6-panel figure validating stability theorems."""

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    fig.suptitle(
        'IBM Quantum Validation of Stability & Uniqueness Theorems\n'
        r'$H = B_1\sigma_z\otimes I + B_2 I\otimes\sigma_z + '
        r'\lambda(\vec{\sigma}\cdot\vec{\sigma})$, '
        r'$|\psi_0\rangle = |10\rangle$',
        fontsize=13, fontweight='bold', y=0.99
    )

    times = np.arange(K_MAX + 1) * DT
    colors = ['#2166ac', '#4393c3', '#d6604d', '#b2182b']

    # ── Panel A: Mutual information growth (Thm 3.1) ──
    ax = fig.add_subplot(gs[0, 0])
    for i, lam in enumerate(LAMBDAS_TIME):
        r = time_results[lam]
        eta = r['eta']
        ax.plot(times, r['exact']['MI'], color=colors[i], linewidth=2,
                label=rf'$\lambda={lam}$, $\eta={eta:.3f}$')
        if 'sim' in r:
            ax.plot(times, r['sim']['MI'], color=colors[i], linestyle='--',
                    linewidth=1.5, alpha=0.7)
        if 'hw' in r:
            ax.plot(times, r['hw']['MI'], 'o', color=colors[i],
                    markersize=4, alpha=0.7)
    ax.set_xlabel('$t$', fontsize=11)
    ax.set_ylabel(r'$I(S:E) = 2\,S(\rho_S)$', fontsize=11)
    ax.set_title('(A) Mutual information growth — Thm 3.1',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ── Panel B: I/(t²‖H_int‖²) → K (Thm 3.1 bound) ──
    ax = fig.add_subplot(gs[0, 1])
    t_arr = times[1:]  # skip t=0
    for i, lam in enumerate(LAMBDAS_TIME):
        r = time_results[lam]
        H, _, H_int = build_H(B1, B2, lam)
        h2 = hs_norm(H_int)**2
        MI = r['exact']['MI'][1:]
        ratio = MI / (t_arr**2 * h2)
        ax.plot(t_arr, ratio, color=colors[i], linewidth=2,
                label=rf'$\lambda={lam}$')
        if 'sim' in r:
            MI_sim = r['sim']['MI'][1:]
            ratio_sim = MI_sim / (t_arr**2 * h2)
            ax.plot(t_arr, ratio_sim, color=colors[i], linestyle='--',
                    linewidth=1.5, alpha=0.7)
    ax.set_xlabel('$t$', fontsize=11)
    ax.set_ylabel(r'$I(S:E)\,/\,(t^2\|H_{\mathrm{int}}\|_2^2)$', fontsize=11)
    ax.set_title(r'(B) Convergence to constant $K$ — Thm 3.1',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel C: η² scaling (key result) ──
    # Use purity deficit Δ = 1 - Tr(ρ²) which scales as clean η²
    # (no log correction unlike von Neumann entropy near pure state)
    ax = fig.add_subplot(gs[0, 2])
    etas = scaling_results['etas']
    MIs_ex = scaling_results['MIs_exact']
    pds_ex = scaling_results['pds_exact']

    # Fit purity deficit (clean η² scaling)
    mask_pd = pds_ex > 1e-15
    if np.sum(mask_pd) > 2:
        c_pd = np.polyfit(np.log(etas[mask_pd]),
                          np.log(pds_ex[mask_pd]), 1)
    else:
        c_pd = [2.0, 0.0]

    # Fit MI for comparison (has log correction)
    mask_mi = MIs_ex > 1e-15
    if np.sum(mask_mi) > 2:
        c_mi = np.polyfit(np.log(etas[mask_mi]),
                          np.log(MIs_ex[mask_mi]), 1)
    else:
        c_mi = [2.0, 0.0]

    # Plot purity deficit data
    ax.loglog(etas, pds_ex, 'ko', markersize=8,
              label=r'$\Delta_{\mathrm{pur}}$ exact', zorder=3)
    if scaling_results['pds_sim'] is not None:
        ax.loglog(etas, scaling_results['pds_sim'], 'bs', markersize=6,
                  label=r'$\Delta_{\mathrm{pur}}$ simulator', zorder=2,
                  alpha=0.7)
    if scaling_results['pds_hw'] is not None:
        ax.loglog(etas, scaling_results['pds_hw'], 'r^', markersize=7,
                  label=rf'$\Delta_{{\mathrm{{pur}}}}$ IBM '
                        rf'{scaling_results["hw_backend"]}', zorder=4)

    # Plot MI data (lighter, for reference)
    ax.loglog(etas, MIs_ex, 'k+', markersize=8,
              label=rf'$I(S:E)$ exact ($\propto\eta^{{{c_mi[0]:.2f}}}$)',
              zorder=2, alpha=0.5)

    eta_line = np.linspace(etas.min() * 0.8, etas.max() * 1.2, 100)
    # Purity deficit fit (primary)
    ax.loglog(eta_line, np.exp(c_pd[1]) * eta_line**c_pd[0],
              'r-', linewidth=2,
              label=rf'Fit $\Delta_{{\mathrm{{pur}}}}$: '
                    rf'$\propto\eta^{{{c_pd[0]:.2f}}}$')
    # Pure η² reference
    ax.loglog(eta_line, np.exp(c_pd[1]) * eta_line**2, 'k--',
              alpha=0.4, label=r'Reference: $\propto\eta^2$')

    ax.set_xlabel(r'$\eta = \|H_{\mathrm{int}}\|_2\,/\,\|H\|_2$',
                  fontsize=11)
    ax.set_ylabel(r'$\Delta_{\mathrm{pur}},\; I(S:E)$', fontsize=11)
    ax.set_title(
        rf'(C) $1 - \mathrm{{Tr}}(\rho_S^2) \propto \eta^{{{c_pd[0]:.2f}}}$ '
        rf'at $t={T_FIXED}$',
        fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which='both')

    # ── Panel D: Purity decay (Thm 8.1) ──
    ax = fig.add_subplot(gs[1, 0])
    for i, lam in enumerate(LAMBDAS_TIME):
        r = time_results[lam]
        eta = r['eta']
        ax.plot(times, r['exact']['pur'], color=colors[i], linewidth=2,
                label=rf'$\eta={eta:.3f}$')
        if 'sim' in r:
            ax.plot(times, r['sim']['pur'], color=colors[i], linestyle='--',
                    linewidth=1.5, alpha=0.7)
        if 'hw' in r:
            ax.plot(times, r['hw']['pur'], 'o', color=colors[i],
                    markersize=4, alpha=0.7)
    ax.set_xlabel('$t$', fontsize=11)
    ax.set_ylabel(r'Purity $\mathrm{Tr}(\rho_S^2)$', fontsize=11)
    ax.set_title(r'(D) Decoherence controlled by $\eta$ — Thm 8.1',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel E: Trotter error in S(ρ_S) ──
    ax = fig.add_subplot(gs[1, 1])
    has_sim = any('sim' in time_results[lam] for lam in LAMBDAS_TIME)
    if has_sim:
        for i, lam in enumerate(LAMBDAS_TIME):
            r = time_results[lam]
            if 'sim' in r:
                err_S = np.abs(r['sim']['S'] - r['exact']['S'])
                # Avoid log(0) issues
                err_S = np.maximum(err_S, 1e-16)
                ax.semilogy(times, err_S, color=colors[i], linewidth=2,
                            label=rf'$\lambda={lam}$')
        ax.set_xlabel('$t$', fontsize=11)
        ax.set_ylabel(
            r'$|S_{\mathrm{Trotter}} - S_{\mathrm{exact}}|$', fontsize=11)
        ax.set_title(r'(E) Trotter error in $S(\rho_S)$',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No simulator data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, alpha=0.5)
        ax.set_title('(E) Trotter error', fontsize=11, fontweight='bold')

    # ── Panel F: Bloch vector |r|(t) ──
    ax = fig.add_subplot(gs[1, 2])
    for i, lam in enumerate(LAMBDAS_TIME):
        r = time_results[lam]
        r_ex = np.sqrt(r['exact']['sx']**2 + r['exact']['sy']**2
                       + r['exact']['sz']**2)
        ax.plot(times, r_ex, color=colors[i], linewidth=2,
                label=rf'$\lambda={lam}$')
        if 'sim' in r:
            r_sim = np.sqrt(r['sim']['sx']**2 + r['sim']['sy']**2
                            + r['sim']['sz']**2)
            ax.plot(times, r_sim, color=colors[i], linestyle='--',
                    linewidth=1.5, alpha=0.7)
        if 'hw' in r:
            r_hw = np.sqrt(r['hw']['sx']**2 + r['hw']['sy']**2
                           + r['hw']['sz']**2)
            ax.plot(times, r_hw, 'o', color=colors[i],
                    markersize=4, alpha=0.7)
    ax.axhline(0, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('$t$', fontsize=11)
    ax.set_ylabel(r'$|\vec{r}(t)|$', fontsize=11)
    ax.set_title('(F) Bloch vector magnitude (purity indicator)',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Legend annotation
    legend_text = 'Solid: exact (numpy)  |  Dashed: Qiskit simulator (Trotter)'
    if any('hw' in time_results[lam] for lam in LAMBDAS_TIME):
        legend_text += '  |  Dots: IBM hardware'
    fig.text(0.5, 0.01, legend_text,
             ha='center', fontsize=10, style='italic', alpha=0.7)

    out_path = os.path.join(OUT_DIR, 'ibm_stability_validation.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  → Saved {out_path}")

    return c_pd[0]  # Return purity deficit fitted exponent (clean η²)


# ═══════════════════════════════════════════════════════════
# CSV EXPORT
# ═══════════════════════════════════════════════════════════

def export_csv(time_results, scaling_results):
    """Export results to CSV files."""

    # ── Time evolution data ──
    times = np.arange(K_MAX + 1) * DT
    out_path = os.path.join(OUT_DIR, 'stability_time_evolution.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['k', 't']
        for lam in LAMBDAS_TIME:
            header += [f'MI_exact_lam{lam}', f'Pur_exact_lam{lam}',
                       f'S_exact_lam{lam}']
            if 'sim' in time_results[lam]:
                header += [f'MI_sim_lam{lam}', f'Pur_sim_lam{lam}',
                           f'S_sim_lam{lam}']
        writer.writerow(header)

        for k in range(K_MAX + 1):
            row = [k, f'{times[k]:.3f}']
            for lam in LAMBDAS_TIME:
                r = time_results[lam]
                row += [
                    f'{r["exact"]["MI"][k]:.8f}',
                    f'{r["exact"]["pur"][k]:.8f}',
                    f'{r["exact"]["S"][k]:.8f}'
                ]
                if 'sim' in r:
                    row += [
                        f'{r["sim"]["MI"][k]:.8f}',
                        f'{r["sim"]["pur"][k]:.8f}',
                        f'{r["sim"]["S"][k]:.8f}'
                    ]
            writer.writerow(row)
    print(f"  → Saved {out_path}")

    # ── Scaling data ──
    out_path = os.path.join(OUT_DIR, 'stability_eta_scaling.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['lambda', 'eta', 'MI_exact']
        if scaling_results['MIs_sim'] is not None:
            header.append('MI_sim')
        if scaling_results['MIs_hw'] is not None:
            header.append('MI_hw')
        writer.writerow(header)

        for i, lam in enumerate(LAMBDAS_SCAN):
            row = [
                f'{lam}',
                f'{scaling_results["etas"][i]:.8f}',
                f'{scaling_results["MIs_exact"][i]:.8f}'
            ]
            if scaling_results['MIs_sim'] is not None:
                row.append(f'{scaling_results["MIs_sim"][i]:.8f}')
            if scaling_results['MIs_hw'] is not None:
                row.append(f'{scaling_results["MIs_hw"][i]:.8f}')
            writer.writerow(row)
    print(f"  → Saved {out_path}")


# ═══════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════

def print_summary(time_results, scaling_results, fitted_exponent):
    """Print verification summary."""
    print(f"\n{'='*70}")
    print("STABILITY THEOREMS — NUMERICAL SUMMARY")
    print(f"{'='*70}")
    print(f"\nModel: H = {B1}σz⊗I + {B2}I⊗σz + λ(σ·σ)")
    print(f"Initial state: |10⟩ (pure product state, non-eigenstate)")
    print(f"Trotter: dt={DT}, k_max={K_MAX}, t_max={K_MAX*DT:.1f}")

    # Purity deficit and MI fits
    etas = scaling_results['etas']
    MIs_ex = scaling_results['MIs_exact']
    pds_ex = scaling_results['pds_exact']
    mask_mi = MIs_ex > 1e-15
    if np.sum(mask_mi) > 2:
        c_mi = np.polyfit(np.log(etas[mask_mi]), np.log(MIs_ex[mask_mi]), 1)
        exp_mi = c_mi[0]
    else:
        exp_mi = 2.0

    print(f"\n--- Theorem 3.1: Quadratic scaling ---")
    print(f"  Purity deficit fit:   η^{fitted_exponent:.4f}  (clean η²)")
    print(f"  MI fit:               η^{exp_mi:.4f}  (has log correction)")
    print(f"  Theoretical:          η^2.0000")
    print(f"  Rel. error (Δ_pur):   {abs(fitted_exponent - 2) / 2 * 100:.2f}%")

    # Trotter error analysis
    has_sim = any('sim' in time_results[lam] for lam in LAMBDAS_TIME)
    if has_sim:
        print(f"\n--- Trotter error (dt={DT}) ---")
        for lam in LAMBDAS_TIME:
            r = time_results[lam]
            if 'sim' in r:
                err = np.max(np.abs(r['sim']['S'] - r['exact']['S']))
                print(f"  λ={lam:5.3f}: max |S_trotter - S_exact| = {err:.6f}")

    print(f"\n--- Theorem 8.1: Almost-unitary dynamics ---")
    print(f"  Small η → purity ≈ 1 (almost unitary)")
    print(f"  Large η → purity drops (strong decoherence)")
    for lam in LAMBDAS_TIME:
        r = time_results[lam]
        pur_final = r['exact']['pur'][-1]
        pur_min = np.min(r['exact']['pur'])
        print(f"  λ={lam:5.3f} (η={r['eta']:.3f}): "
              f"Pur_final={pur_final:.4f}, Pur_min={pur_min:.4f}")

    print(f"\n--- Hardware readiness ---")
    n_circuits = len(LAMBDAS_TIME) * (K_MAX + 1) + len(LAMBDAS_SCAN)
    print(f"  Total circuits: {n_circuits}")
    print(f"  Qubits needed: {N_QUBITS}")
    print(f"  Estimated QPU time: ~{n_circuits * 0.15:.0f}s")
    print(f"  Command: python run_ibm_stability.py --mode hardware")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='IBM Quantum validation of stability & uniqueness theorems')
    parser.add_argument(
        '--mode', choices=['simulator', 'hardware', 'both'],
        default='simulator',
        help='Execution mode (default: simulator)')
    parser.add_argument(
        '--shots', type=int, default=4096,
        help='Shots per circuit on hardware (default: 4096)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 65)
    print("IBM Quantum Validation — Stability & Uniqueness Theorems")
    print("=" * 65)
    print(f"  Model: H = {B1}σz⊗I + {B2}I⊗σz + λ(σ·σ)")
    print(f"  Qubits: {N_QUBITS} (1 system + 1 environment)")
    print(f"  Trotter: dt={DT}, k_max={K_MAX} → t_max={K_MAX*DT:.1f}")
    print(f"  Mode: {args.mode}")
    print(f"  λ values (time evolution): {LAMBDAS_TIME}")
    print(f"  λ values (η² scan): {LAMBDAS_SCAN}")

    # ── Step 1: Time evolution for multiple λ ──
    print(f"\n[1/3] Time evolution ({len(LAMBDAS_TIME)} λ values "
          f"× {K_MAX+1} steps)...")
    time_results, hw_backend = validate_time_evolution(args.mode, args.shots)

    # ── Step 2: η² scaling scan ──
    print(f"\n[2/3] η² scaling ({len(LAMBDAS_SCAN)} λ values "
          f"at t={T_FIXED})...")
    scaling_results = validate_eta_scaling(args.mode, args.shots)

    # ── Step 3: Output ──
    print(f"\n[3/3] Generating output...")
    fitted_exp = plot_results(time_results, scaling_results, args.mode)
    export_csv(time_results, scaling_results)
    print_summary(time_results, scaling_results, fitted_exp)

    print(f"\n{'='*65}")
    print("VALIDATION COMPLETE")
    if args.mode == 'simulator':
        print("  Trotterized circuits validated on StatevectorEstimator.")
        print("  Ready for QPU: python run_ibm_stability.py --mode hardware")
    elif args.mode == 'hardware':
        print(f"  Results from IBM Quantum hardware ({hw_backend}).")
    else:
        print("  Full comparison: exact vs simulator vs hardware.")
    print(f"{'='*65}")


if __name__ == '__main__':
    main()
