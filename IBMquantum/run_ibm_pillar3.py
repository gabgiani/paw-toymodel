"""
IBM Quantum Validation — Pillar 3: Observer-Dependent Time
===========================================================

Validates Pillar 3 on quantum hardware (or simulator):
  Two clocks (dt=0.20 and dt=0.35) applied to the same Hamiltonian
  produce different ⟨σ_z⟩ dynamics and different entropy trajectories.

Model: 3 qubits (1 system S + 2 environment E₁E₂)
  H = (ω/2)σ_x⊗I⊗I + g[σ_x⊗σ_x⊗I + σ_x⊗I⊗σ_x]

Each clock defines a different Trotter step size:
  Clock C:  U_C(dt)  = Rx(ω·dt_C)₀ · RXX(2g·dt_C)₀₁ · RXX(2g·dt_C)₀₂
  Clock C': U_C'(dt) = Rx(ω·dt_C')₀ · RXX(2g·dt_C')₀₁ · RXX(2g·dt_C')₀₂

Both see the same Hamiltonian, same initial state — but different temporal
descriptions. Neither is "correct"; both are valid relational accounts.

Usage:
  python IBMquantum/run_ibm_pillar3.py --mode simulator   # Local test (default)
  python IBMquantum/run_ibm_pillar3.py --mode hardware    # Real IBM QPU
  python IBMquantum/run_ibm_pillar3.py --mode both        # Full comparison

Estimated QPU time: ~40 seconds (42 circuits × 2 clocks).
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
N_ENV = 2       # 2 environment qubits → 3 total
N_QUBITS = 1 + N_ENV  # = 3
K_MAX = 20      # Trotter steps per clock

# Two clocks with different time steps
DT_C1 = 0.20
DT_C2 = 0.35

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


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

def build_trotter_circuit(k, dt):
    """
    Build k-step first-order Trotter circuit with time step dt.

    Each step applies:
      1. Rx(ω·dt) on q0          — system rotation
      2. RXX(2g·dt) on q0,q1     — S-E₁ coupling
      3. RXX(2g·dt) on q0,q2     — S-E₂ coupling
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(N_QUBITS)
    for _ in range(k):
        qc.rx(OMEGA * dt, 0)
        qc.rxx(2 * G * dt, 0, 1)
        qc.rxx(2 * G * dt, 0, 2)
    return qc


def build_all_circuits(dt):
    """Build circuits for k = 0, 1, ..., K_MAX with given dt."""
    return [build_trotter_circuit(k, dt) for k in range(K_MAX + 1)]


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

def compute_exact_reference(dt):
    """
    Compute exact S_eff(k) via QuTiP for n_env=2 with given dt.

    This is the "gold standard" — no Trotter error, no noise.
    Returns (sx, sy, sz, S_eff) arrays of length K_MAX+1.
    """
    import qutip as qt

    sigma_x = qt.sigmax()

    # H_S = (ω/2)σ_x ⊗ I ⊗ I
    H_S = (OMEGA / 2) * qt.tensor(sigma_x, qt.qeye(2), qt.qeye(2))

    # H_SE = g(σ_x⊗σ_x⊗I + σ_x⊗I⊗σ_x)
    H_SE = (G * qt.tensor(sigma_x, sigma_x, qt.qeye(2)) +
            G * qt.tensor(sigma_x, qt.qeye(2), sigma_x))

    H_tot = H_S + H_SE

    # Initial state |0⟩_S ⊗ |0⟩_E1 ⊗ |0⟩_E2
    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))

    sx = np.zeros(K_MAX + 1)
    sy = np.zeros(K_MAX + 1)
    sz = np.zeros(K_MAX + 1)
    S_eff = np.zeros(K_MAX + 1)

    for k in range(K_MAX + 1):
        t = k * dt
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

def run_on_hardware(circuits, obs_list, backend):
    """
    Run on IBM Quantum hardware via EstimatorV2.

    Transpiles circuits to ISA, maps observables to physical layout.
    """
    from qiskit_ibm_runtime import EstimatorV2
    from qiskit.transpiler import generate_preset_pass_manager

    # Transpile all circuits to ISA
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
    print(f"    Deepest circuit (k={K_MAX}): {ops}")

    # Submit job
    estimator = EstimatorV2(mode=backend)
    print(f"    Submitting {len(pubs)} PUBs (3 observables each)...")
    job = estimator.run(pubs)
    print(f"    Job ID: {job.job_id()}")
    print(f"    Waiting for results...")

    result = job.result()
    print("    Results received!")

    sx, sy, sz = [], [], []
    for pub_result in result:
        evs = pub_result.data.evs
        sx.append(float(evs[0]))
        sy.append(float(evs[1]))
        sz.append(float(evs[2]))

    return np.array(sx), np.array(sy), np.array(sz)


# ═══════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════

def plot_results(exact_c1, exact_c2, data_c1, data_c2, source_label,
                 hw_backend=None):
    """
    Plot Pillar 3 comparison: two clocks → different dynamics & arrows.

    4 panels: ⟨σ_z⟩ for C1, ⟨σ_z⟩ for C2, S_eff for C1, S_eff for C2.
    """
    sx_ex1, sy_ex1, sz_ex1, S_ex1 = exact_c1
    sx_ex2, sy_ex2, sz_ex2, S_ex2 = exact_c2
    ks = np.arange(K_MAX + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    backend_label = f'IBM {hw_backend}' if hw_backend else source_label

    # ── Top-left: ⟨σ_z⟩ Clock C1 ──
    ax = axes[0, 0]
    ax.plot(ks, sz_ex1, 'k-', linewidth=2, label='QuTiP exact', zorder=3)
    ax.plot(ks, data_c1[2], 'o', color='#2196F3', markersize=6,
            label=backend_label, zorder=4, alpha=0.8)
    ax.set_xlabel('Clock tick k')
    ax.set_ylabel(r'$\langle\sigma_z\rangle$')
    ax.set_title(f'Clock C  (Δt = {DT_C1})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-1.15, 1.15)

    # ── Top-right: ⟨σ_z⟩ Clock C2 ──
    ax = axes[0, 1]
    ax.plot(ks, sz_ex2, 'k-', linewidth=2, label='QuTiP exact', zorder=3)
    ax.plot(ks, data_c2[2], 's', color='#E91E63', markersize=6,
            label=backend_label, zorder=4, alpha=0.8)
    ax.set_xlabel('Clock tick k')
    ax.set_ylabel(r'$\langle\sigma_z\rangle$')
    ax.set_title(f"Clock C' (Δt = {DT_C2})", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-1.15, 1.15)

    # ── Bottom-left: S_eff Clock C1 ──
    ax = axes[1, 0]
    ax.plot(ks, S_ex1, 'k-', linewidth=2, label='QuTiP exact', zorder=3)
    S_data1 = entropy_from_bloch(*data_c1)
    ax.plot(ks, S_data1, 'o', color='#2196F3', markersize=6,
            label=backend_label, zorder=4, alpha=0.8)
    ax.axhline(np.log(2), color='gray', ls=':', alpha=0.5, label='ln 2')
    ax.set_xlabel('Clock tick k')
    ax.set_ylabel(r'$S_{\mathrm{eff}}$')
    ax.set_title(f'Entropy — Clock C  (Δt = {DT_C1})', fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ── Bottom-right: S_eff Clock C2 ──
    ax = axes[1, 1]
    ax.plot(ks, S_ex2, 'k-', linewidth=2, label='QuTiP exact', zorder=3)
    S_data2 = entropy_from_bloch(*data_c2)
    ax.plot(ks, S_data2, 's', color='#E91E63', markersize=6,
            label=backend_label, zorder=4, alpha=0.8)
    ax.axhline(np.log(2), color='gray', ls=':', alpha=0.5, label='ln 2')
    ax.set_xlabel('Clock tick k')
    ax.set_ylabel(r'$S_{\mathrm{eff}}$')
    ax.set_title(f"Entropy — Clock C' (Δt = {DT_C2})", fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    title = 'Unified Relational Time Formula — Pillar 3: Observer-Dependent Time'
    if hw_backend:
        title += f' ({hw_backend})'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.text(0.5, 0.00,
             r"Same $|\Psi\rangle$, same Hamiltonian, same formula — "
             r"different clock $C$ yields different temporal description",
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(OUT_DIR, 'ibm_pillar3_validation.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved {out_path}")


# ═══════════════════════════════════════════════════════════
# CSV EXPORT
# ═══════════════════════════════════════════════════════════

def export_csv(exact_c1, exact_c2, data_c1, data_c2, source_label):
    """Export all results to CSV."""
    ks = np.arange(K_MAX + 1)
    _, _, sz_ex1, S_ex1 = exact_c1
    _, _, sz_ex2, S_ex2 = exact_c2
    S_data1 = entropy_from_bloch(*data_c1)
    S_data2 = entropy_from_bloch(*data_c2)

    out_path = os.path.join(OUT_DIR, 'table_ibm_pillar3.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'k',
            f't_C1 (dt={DT_C1})', 'sz_exact_C1', 'S_exact_C1',
            f'sz_{source_label}_C1', f'S_{source_label}_C1',
            f't_C2 (dt={DT_C2})', 'sz_exact_C2', 'S_exact_C2',
            f'sz_{source_label}_C2', f'S_{source_label}_C2',
        ])
        for k in range(K_MAX + 1):
            writer.writerow([
                k,
                f'{k*DT_C1:.2f}', f'{sz_ex1[k]:.6f}', f'{S_ex1[k]:.6f}',
                f'{data_c1[2][k]:.6f}', f'{S_data1[k]:.6f}',
                f'{k*DT_C2:.2f}', f'{sz_ex2[k]:.6f}', f'{S_ex2[k]:.6f}',
                f'{data_c2[2][k]:.6f}', f'{S_data2[k]:.6f}',
            ])
    print(f"  → Saved {out_path}")


# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════

def print_summary(exact_c1, exact_c2, data_c1, data_c2, source_label):
    """Print Pillar 3 comparison statistics."""
    _, _, sz_ex1, S_ex1 = exact_c1
    _, _, sz_ex2, S_ex2 = exact_c2
    S_data1 = entropy_from_bloch(*data_c1)
    S_data2 = entropy_from_bloch(*data_c2)

    print(f"\n  Clock C  (dt = {DT_C1}):")
    print(f"    ⟨σ_z⟩(0) = {data_c1[2][0]:.4f}  →  ⟨σ_z⟩({K_MAX}) = {data_c1[2][K_MAX]:.4f}")
    print(f"    S_eff(0)  = {S_data1[0]:.4f}  →  S_eff({K_MAX})  = {S_data1[K_MAX]:.4f}")
    print(f"    ΔS_eff    = {S_data1[K_MAX] - S_data1[0]:.4f}")

    print(f"\n  Clock C' (dt = {DT_C2}):")
    print(f"    ⟨σ_z⟩(0) = {data_c2[2][0]:.4f}  →  ⟨σ_z⟩({K_MAX}) = {data_c2[2][K_MAX]:.4f}")
    print(f"    S_eff(0)  = {S_data2[0]:.4f}  →  S_eff({K_MAX})  = {S_data2[K_MAX]:.4f}")
    print(f"    ΔS_eff    = {S_data2[K_MAX] - S_data2[0]:.4f}")

    # The key Pillar 3 test: are the two histories different?
    sz_diff = np.max(np.abs(data_c1[2] - data_c2[2]))
    S_diff = np.max(np.abs(S_data1 - S_data2))

    print(f"\n  Observer-dependence test:")
    print(f"    Max |⟨σ_z⟩_C - ⟨σ_z⟩_C'| = {sz_diff:.4f}")
    print(f"    Max |S_C - S_C'|            = {S_diff:.4f}")

    if sz_diff > 0.01:
        print(f"    ✓ Temporal descriptions DIFFER — Pillar 3 confirmed")
    else:
        print(f"    ✗ Temporal descriptions too similar — check parameters")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='IBM Quantum validation — Pillar 3: Observer-Dependent Time')
    parser.add_argument(
        '--mode', choices=['simulator', 'hardware', 'both'],
        default='simulator',
        help='Execution mode (default: simulator)')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("IBM Quantum Validation — Pillar 3")
    print("Unified Relational Time Formula — Observer-Dependent Time")
    print("=" * 60)
    print(f"  Parameters: omega={OMEGA}, g={G}")
    print(f"  Clock C:  dt = {DT_C1}  (k=0..{K_MAX}  →  t_max = {K_MAX*DT_C1:.1f})")
    print(f"  Clock C': dt = {DT_C2}  (k=0..{K_MAX}  →  t_max = {K_MAX*DT_C2:.1f})")
    print(f"  Qubits: {N_QUBITS} (1 system + {N_ENV} environment)")
    print(f"  Mode: {args.mode}")

    # ── Step 1: QuTiP exact references ──
    print(f"\n[1/4] Computing QuTiP exact references...")
    exact_c1 = compute_exact_reference(DT_C1)
    exact_c2 = compute_exact_reference(DT_C2)
    print(f"  Clock C  (dt={DT_C1}): S(0)={exact_c1[3][0]:.4f} → S({K_MAX})={exact_c1[3][K_MAX]:.4f}")
    print(f"  Clock C' (dt={DT_C2}): S(0)={exact_c2[3][0]:.4f} → S({K_MAX})={exact_c2[3][K_MAX]:.4f}")

    # ── Step 2: Build circuits ──
    print(f"\n[2/4] Building Trotter circuits...")
    circuits_c1 = build_all_circuits(DT_C1)
    circuits_c2 = build_all_circuits(DT_C2)
    obs_list = build_observables()
    print(f"  Clock C:  {len(circuits_c1)} circuits (dt={DT_C1})")
    print(f"  Clock C': {len(circuits_c2)} circuits (dt={DT_C2})")
    print(f"  Total: {len(circuits_c1) + len(circuits_c2)} circuits")

    data_c1 = None
    data_c2 = None
    hw_backend = None

    # ── Step 3: Execute ──
    if args.mode in ('simulator', 'both'):
        print(f"\n[3/4] Simulator execution...")
        print(f"  Clock C  (dt={DT_C1}):")
        sx1, sy1, sz1 = run_on_simulator(circuits_c1, obs_list)
        data_c1 = (sx1, sy1, sz1)
        print(f"    Done.")

        print(f"  Clock C' (dt={DT_C2}):")
        sx2, sy2, sz2 = run_on_simulator(circuits_c2, obs_list)
        data_c2 = (sx2, sy2, sz2)
        print(f"    Done.")

    if args.mode in ('hardware', 'both'):
        from qiskit_ibm_runtime import QiskitRuntimeService

        step = '3/4' if args.mode == 'hardware' else '3b/4'
        print(f"\n[{step}] IBM Quantum hardware execution...")

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
        hw_backend = backend.name
        print(f"  Backend: {hw_backend} ({backend.num_qubits} qubits)")

        print(f"\n  Clock C  (dt={DT_C1}):")
        sx1, sy1, sz1 = run_on_hardware(circuits_c1, obs_list, backend)
        data_c1 = (sx1, sy1, sz1)

        print(f"\n  Clock C' (dt={DT_C2}):")
        sx2, sy2, sz2 = run_on_hardware(circuits_c2, obs_list, backend)
        data_c2 = (sx2, sy2, sz2)

    # ── Step 4: Output ──
    source = hw_backend if hw_backend else 'simulator'
    print(f"\n[4/4] Generating output...")
    plot_results(exact_c1, exact_c2, data_c1, data_c2, source, hw_backend)
    export_csv(exact_c1, exact_c2, data_c1, data_c2, source)
    print_summary(exact_c1, exact_c2, data_c1, data_c2, source)

    print(f"\n{'=' * 60}")
    print("PILLAR 3 VALIDATION COMPLETE")
    print(f"{'=' * 60}")

    sz_diff = np.max(np.abs(data_c1[2] - data_c2[2]))
    S_data1 = entropy_from_bloch(*data_c1)
    S_data2 = entropy_from_bloch(*data_c2)
    S_diff = np.max(np.abs(S_data1 - S_data2))

    if sz_diff > 0.01:
        print(f"  Two clocks produce DIFFERENT temporal descriptions.")
        print(f"  ⟨σ_z⟩ histories differ by up to {sz_diff:.4f}")
        print(f"  Entropy trajectories differ by up to {S_diff:.4f}")
        print(f"  → Pillar 3 confirmed: time is observer-dependent.")
        if hw_backend:
            print(f"  → Validated on IBM Quantum hardware ({hw_backend}).")
        else:
            print(f"  → Validated on Qiskit statevector simulator.")
            print(f"  → Ready for hardware: --mode hardware")
    else:
        print(f"  WARNING: temporal descriptions too similar.")
        print(f"  Max ⟨σ_z⟩ difference = {sz_diff:.6f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
