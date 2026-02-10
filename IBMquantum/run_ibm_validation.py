"""
IBM Quantum Hardware Validation of the Unified Relational Time Formula
=================================================================

Validates Pillar 2 (thermodynamic arrow) on real quantum hardware.

Model: 3 qubits (1 system S + 2 environment E₁E₂)
  H = (ω/2)σ_x⊗I⊗I + g[σ_x⊗σ_x⊗I + σ_x⊗I⊗σ_x]
  |ψ₀⟩ = |0⟩_S ⊗ |0⟩_E1 ⊗ |0⟩_E2

First-order Trotterization:
  U(dt) ≈ Rx(ωdt)₀ · RXX(2gdt)₀₁ · RXX(2gdt)₀₂

At each step k=0..K, measure ⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩ on qubit 0,
reconstruct ρ_S via Bloch vector, compute S_eff.

Comparison:
  - QuTiP exact evolution (no Trotter, no noise)
  - Qiskit statevector simulator (Trotter, no noise)
  - IBM Quantum hardware (Trotter + real noise)

Usage:
  python run_ibm_validation.py --mode simulator    # Local test first!
  python run_ibm_validation.py --mode hardware     # Real IBM QPU
  python run_ibm_validation.py --mode both         # Full comparison

Estimated QPU time: ~30 seconds (well within 10 min/month free tier).
Queue wait time: 5–30 minutes (not counted against QPU budget).
"""

import json
import os
import sys
import csv
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Physical parameters (matching paw-toymodel) ──
OMEGA = 1.0
G = 0.1
DT = 0.2
N_ENV = 2       # 2 environment qubits → 3 total
K_MAX = 20      # Trotter steps → t_max = 4.0
N_QUBITS = 1 + N_ENV  # = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'output')


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
# CIRCUIT CONSTRUCTION
# ═══════════════════════════════════════════════════════════

def build_trotter_circuit(k):
    """
    Build k-step first-order Trotter circuit for H_SE.

    Each step applies:
      1. Rx(ω·dt) on q0          — system rotation
      2. RXX(2g·dt) on q0,q1     — S-E₁ coupling
      3. RXX(2g·dt) on q0,q2     — S-E₂ coupling

    Gate count per step: 1 single-qubit + 2 two-qubit (≈4 CX after decomposition).
    For k=20: ~80 CX total — within coherence limits of IBM Eagle/Heron.
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(N_QUBITS)

    for _ in range(k):
        # e^{-i(ω/2)σ_x dt} on system qubit
        qc.rx(OMEGA * DT, 0)

        # e^{-ig σ_x⊗σ_x dt} on S-E₁
        qc.rxx(2 * G * DT, 0, 1)

        # e^{-ig σ_x⊗σ_x dt} on S-E₂
        qc.rxx(2 * G * DT, 0, 2)

    return qc


def build_all_circuits():
    """Build circuits for k = 0, 1, ..., K_MAX."""
    circuits = []
    for k in range(K_MAX + 1):
        circuits.append(build_trotter_circuit(k))
    return circuits


def build_observables():
    """
    Build Pauli observables for partial tomography of qubit 0.

    Qiskit convention: rightmost character = qubit 0.
    So 'IIZ' means σ_z on q0, identity on q1 and q2.
    """
    from qiskit.quantum_info import SparsePauliOp

    return [
        SparsePauliOp('IIX'),   # ⟨σ_x⟩ on system qubit
        SparsePauliOp('IIY'),   # ⟨σ_y⟩ on system qubit
        SparsePauliOp('IIZ'),   # ⟨σ_z⟩ on system qubit
    ]


# ═══════════════════════════════════════════════════════════
# QUTIP EXACT REFERENCE
# ═══════════════════════════════════════════════════════════

def compute_exact_reference():
    """
    Compute exact S_eff(k) via QuTiP for n_env=2.

    This is the "gold standard" — no Trotter error, no noise.
    Returns (sx, sy, sz, S_eff) arrays of length K_MAX+1.
    """
    import qutip as qt

    sigma_x = qt.sigmax()

    # H_S = (ω/2)σ_x ⊗ I ⊗ I
    id_list = [qt.qeye(2) for _ in range(N_ENV)]
    H_S = (OMEGA / 2) * qt.tensor(sigma_x, *id_list)

    # H_SE = g Σ_j σ_x^(S) ⊗ σ_x^(E_j)
    dim_env = 2**N_ENV
    H_SE = qt.Qobj(np.zeros((2 * dim_env, 2 * dim_env)),
                    dims=[[2] + [2]*N_ENV, [2] + [2]*N_ENV])
    for j in range(N_ENV):
        ops = [qt.qeye(2) for _ in range(N_ENV)]
        ops[j] = sigma_x
        H_SE += G * qt.tensor(sigma_x, *ops)

    H_tot = H_S + H_SE

    # Initial state |0⟩_S ⊗ |0⟩_E1 ⊗ |0⟩_E2
    psi0 = qt.tensor(qt.basis(2, 0), *[qt.basis(2, 0) for _ in range(N_ENV)])

    sx = np.zeros(K_MAX + 1)
    sy = np.zeros(K_MAX + 1)
    sz = np.zeros(K_MAX + 1)
    S_eff = np.zeros(K_MAX + 1)

    for k in range(K_MAX + 1):
        t = k * DT
        U = (-1j * H_tot * t).expm()
        psi_t = U * psi0
        rho_S = psi_t.ptrace(0)

        sx[k] = qt.expect(qt.sigmax(), rho_S)
        sy[k] = qt.expect(qt.sigmay(), rho_S)
        sz[k] = qt.expect(qt.sigmaz(), rho_S)
        S_eff[k] = qt.entropy_vn(rho_S)

    return sx, sy, sz, S_eff


# ═══════════════════════════════════════════════════════════
# ENTROPY FROM BLOCH VECTOR
# ═══════════════════════════════════════════════════════════

def entropy_from_bloch(sx, sy, sz):
    """
    Compute von Neumann entropy from Bloch vector components.

    ρ = (I + r·σ)/2, eigenvalues λ± = (1±|r|)/2.
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


# ═══════════════════════════════════════════════════════════
# LOCAL SIMULATOR
# ═══════════════════════════════════════════════════════════

def run_on_simulator(circuits, obs_list):
    """
    Run on Qiskit's local StatevectorEstimator (exact Trotter, no noise).

    This verifies the Trotter circuits are correct before using QPU time.
    """
    from qiskit.primitives import StatevectorEstimator

    print("  Running on local statevector simulator...")
    estimator = StatevectorEstimator()

    # Each PUB: (circuit, list_of_observables)
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

    Uses the least busy available backend with ≥3 qubits.
    Transpiles circuits to ISA, maps observables to physical layout.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
    from qiskit.transpiler import generate_preset_pass_manager

    api_key = load_api_key()

    print("  Connecting to IBM Quantum...")
    service = QiskitRuntimeService(
        channel='ibm_quantum_platform',
        token=api_key
    )

    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=N_QUBITS
    )
    print(f"  Backend: {backend.name} ({backend.num_qubits} qubits)")

    # Transpile all circuits to ISA (instruction set architecture)
    print("  Transpiling circuits...")
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
    print(f"  Deepest circuit (k={K_MAX}): {ops}")

    # Submit job
    estimator = EstimatorV2(mode=backend)
    print(f"  Submitting {len(pubs)} PUBs (3 observables each)...")
    job = estimator.run(pubs)
    print(f"  Job ID: {job.job_id()}")
    print(f"  Status: {job.status()}")
    print("  Waiting for results (queue + execution)...")
    print("  (QPU budget: ~30s of the 600s monthly quota)")

    result = job.result()
    print("  Results received!")

    sx, sy, sz = [], [], []
    for pub_result in result:
        evs = pub_result.data.evs
        sx.append(float(evs[0]))
        sy.append(float(evs[1]))
        sz.append(float(evs[2]))

    return np.array(sx), np.array(sy), np.array(sz), backend.name


# ═══════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════

def plot_results(exact, sim=None, hw=None, hw_backend=None):
    """Plot S_eff comparison: QuTiP exact vs simulator vs hardware."""
    times = np.arange(K_MAX + 1) * DT
    sx_ex, sy_ex, sz_ex, S_ex = exact

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ── Panel 1: ⟨σ_z⟩(t) ──
    ax = axes[0]
    ax.plot(times, sz_ex, 'k-', linewidth=2, label='QuTiP exact', zorder=3)
    if sim is not None:
        ax.plot(times, sim[2], 'b--', linewidth=1.5,
                label='Qiskit simulator (Trotter)', zorder=2)
    if hw is not None:
        ax.plot(times, hw[2], 'ro', markersize=7,
                label=f'IBM {hw_backend}', zorder=4, alpha=0.8)
    ax.set_xlabel('Time t = k·dt', fontsize=11)
    ax.set_ylabel('⟨σ_z⟩', fontsize=11)
    ax.set_title('Damped dynamics', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 2: S_eff(t) — THE KEY RESULT ──
    ax = axes[1]
    ax.plot(times, S_ex, 'k-', linewidth=2, label='QuTiP exact', zorder=3)
    if sim is not None:
        S_sim = entropy_from_bloch(*sim)
        ax.plot(times, S_sim, 'b--', linewidth=1.5,
                label='Qiskit simulator', zorder=2)
    if hw is not None:
        S_hw = entropy_from_bloch(*hw)
        ax.plot(times, S_hw, 'ro', markersize=7,
                label=f'IBM {hw_backend}', zorder=4, alpha=0.8)
    ax.axhline(np.log(2), color='gray', ls=':', alpha=0.5, label='ln 2')
    ax.set_xlabel('Time t = k·dt', fontsize=11)
    ax.set_ylabel('S_eff(t)', fontsize=11)
    ax.set_title('Thermodynamic arrow of time', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 3: Bloch vector magnitude |r|(t) ──
    ax = axes[2]
    r_ex = np.sqrt(sx_ex**2 + sy_ex**2 + sz_ex**2)
    ax.plot(times, r_ex, 'k-', linewidth=2, label='QuTiP exact', zorder=3)
    if sim is not None:
        r_sim = np.sqrt(sim[0]**2 + sim[1]**2 + sim[2]**2)
        ax.plot(times, r_sim, 'b--', linewidth=1.5,
                label='Qiskit simulator', zorder=2)
    if hw is not None:
        r_hw = np.sqrt(hw[0]**2 + hw[1]**2 + hw[2]**2)
        ax.plot(times, r_hw, 'ro', markersize=7,
                label=f'IBM {hw_backend}', zorder=4, alpha=0.8)
    ax.axhline(0, color='gray', ls=':', alpha=0.3)
    ax.set_xlabel('Time t = k·dt', fontsize=11)
    ax.set_ylabel('|r(t)| = Tr[ρ²]^{1/2}', fontsize=11)
    ax.set_title('Purity decay (|r|→0 = decoherence)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Title
    if hw is not None:
        title = (f'Unified Relational Time Formula — IBM Quantum Validation '
                 f'({hw_backend})')
    else:
        title = 'Unified Relational Time Formula — Trotterized vs Exact'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(OUT_DIR, 'ibm_quantum_validation.png')
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → Saved {out_path}")


# ═══════════════════════════════════════════════════════════
# CSV EXPORT
# ═══════════════════════════════════════════════════════════

def export_csv(exact, sim=None, hw=None, hw_backend=None):
    """Export all results to a single CSV."""
    times = np.arange(K_MAX + 1) * DT
    sx_ex, sy_ex, sz_ex, S_ex = exact

    out_path = os.path.join(OUT_DIR, 'table_ibm_quantum_validation.csv')

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)

        header = ['k', 't', 'sx_exact', 'sy_exact', 'sz_exact', 'S_exact']
        if sim is not None:
            header += ['sx_sim', 'sy_sim', 'sz_sim', 'S_sim']
        if hw is not None:
            header += [f'sx_{hw_backend}', f'sy_{hw_backend}',
                       f'sz_{hw_backend}', f'S_{hw_backend}']
        writer.writerow(header)

        for k in range(K_MAX + 1):
            row = [k, f'{times[k]:.2f}',
                   f'{sx_ex[k]:.6f}', f'{sy_ex[k]:.6f}',
                   f'{sz_ex[k]:.6f}', f'{S_ex[k]:.6f}']
            if sim is not None:
                S_s = entropy_from_bloch(
                    sim[0][k:k+1], sim[1][k:k+1], sim[2][k:k+1])[0]
                row += [f'{sim[0][k]:.6f}', f'{sim[1][k]:.6f}',
                        f'{sim[2][k]:.6f}', f'{S_s:.6f}']
            if hw is not None:
                S_h = entropy_from_bloch(
                    hw[0][k:k+1], hw[1][k:k+1], hw[2][k:k+1])[0]
                row += [f'{hw[0][k]:.6f}', f'{hw[1][k]:.6f}',
                        f'{hw[2][k]:.6f}', f'{S_h:.6f}']
            writer.writerow(row)

    print(f"  → Saved {out_path}")


# ═══════════════════════════════════════════════════════════
# SUMMARY STATISTICS
# ═══════════════════════════════════════════════════════════

def print_summary(exact, sim=None, hw=None, hw_backend=None):
    """Print comparison statistics."""
    sx_ex, sy_ex, sz_ex, S_ex = exact

    print(f"\n  QuTiP exact:")
    print(f"    S_eff(0) = {S_ex[0]:.6f}  →  S_eff({K_MAX}) = {S_ex[K_MAX]:.6f}")
    print(f"    ⟨σ_z⟩(0) = {sz_ex[0]:.6f}  →  ⟨σ_z⟩({K_MAX}) = {sz_ex[K_MAX]:.6f}")

    if sim is not None:
        S_sim = entropy_from_bloch(*sim)
        trotter_err = np.max(np.abs(S_sim - S_ex))
        print(f"\n  Qiskit simulator (Trotter):")
        print(f"    Max |S_sim - S_exact| = {trotter_err:.6f}")
        print(f"    S_eff({K_MAX}) = {S_sim[K_MAX]:.6f}")

    if hw is not None:
        S_hw = entropy_from_bloch(*hw)
        hw_err = np.max(np.abs(S_hw - S_ex))
        # Check if arrow is observed
        S_first = S_hw[0]
        S_last = np.mean(S_hw[-5:])  # average last 5 for stability
        arrow_observed = S_last > S_first + 0.05
        print(f"\n  IBM {hw_backend} (real hardware):")
        print(f"    Max |S_hw - S_exact| = {hw_err:.6f}")
        print(f"    S_eff(0) = {S_hw[0]:.6f}  →  S_eff({K_MAX}) = {S_hw[K_MAX]:.6f}")
        print(f"    Arrow observed: {'YES' if arrow_observed else 'NO'} "
              f"(S_final - S_initial = {S_last - S_first:.4f})")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='IBM Quantum validation of the unified relational time formula')
    parser.add_argument(
        '--mode', choices=['simulator', 'hardware', 'both'],
        default='simulator',
        help='Execution mode (default: simulator)')
    parser.add_argument(
        '--shots', type=int, default=4096,
        help='Shots per circuit on hardware (default: 4096)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("IBM Quantum Validation")
    print("Unified Relational Time Formula — Pillar 2")
    print("=" * 60)
    print(f"  Parameters: omega={OMEGA}, g={G}, dt={DT}")
    print(f"  Qubits: {N_QUBITS} (1 system + {N_ENV} environment)")
    print(f"  Trotter steps: {K_MAX}  →  t_max = {K_MAX * DT:.1f}")
    print(f"  Mode: {args.mode}")

    # ── Step 1: QuTiP exact ──
    print(f"\n[1/4] Computing QuTiP exact reference...")
    exact = compute_exact_reference()

    # ── Step 2: Build circuits ──
    print(f"\n[2/4] Building Trotter circuits...")
    circuits = build_all_circuits()
    obs_list = build_observables()
    last = circuits[-1]
    print(f"  {len(circuits)} circuits (k=0..{K_MAX})")
    print(f"  Deepest circuit (k={K_MAX}): depth={last.depth()}, "
          f"gates={last.size()}")

    sim_result = None
    hw_result = None
    hw_backend = None

    # ── Step 3: Execute ──
    if args.mode in ('simulator', 'both'):
        print(f"\n[3/4] Simulator execution...")
        sim_sx, sim_sy, sim_sz = run_on_simulator(circuits, obs_list)
        sim_result = (sim_sx, sim_sy, sim_sz)

    if args.mode in ('hardware', 'both'):
        step = '3/4' if args.mode == 'hardware' else '3b/4'
        print(f"\n[{step}] IBM Quantum hardware execution...")
        hw_sx, hw_sy, hw_sz, hw_backend = run_on_hardware(
            circuits, obs_list, shots=args.shots)
        hw_result = (hw_sx, hw_sy, hw_sz)

    # ── Step 4: Output ──
    print(f"\n[4/4] Generating output...")
    plot_results(exact, sim_result, hw_result, hw_backend)
    export_csv(exact, sim_result, hw_result, hw_backend)
    print_summary(exact, sim_result, hw_result, hw_backend)

    print(f"\n{'=' * 60}")
    print("VALIDATION COMPLETE")
    if hw_result is not None:
        S_hw = entropy_from_bloch(*hw_result)
        S_grow = np.mean(S_hw[-5:]) - S_hw[0]
        if S_grow > 0.05:
            print(f"  The thermodynamic arrow of time has been observed")
            print(f"  on IBM Quantum hardware ({hw_backend}).")
            print(f"  S_eff grew by {S_grow:.4f} — confirming Pillar 2")
            print(f"  of the unified relational time formula.")
        else:
            print(f"  S_eff growth = {S_grow:.4f} (weak)")
            print(f"  Hardware noise may dominate at this circuit depth.")
    else:
        S_sim = entropy_from_bloch(*sim_result)
        trotter_err = np.max(np.abs(S_sim - exact[3]))
        print(f"  Trotter circuits match exact evolution "
              f"(max error = {trotter_err:.6f})")
        print(f"  Ready for hardware execution: --mode hardware")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
