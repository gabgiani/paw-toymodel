"""
IBM Quantum Enhanced Validation — Noise Characterization + Error Bars
=====================================================================

Extends run_ibm_validation.py with:
  1. Backend noise properties (T1, T2, gate error rates)
  2. Multiple shots for error bar estimation
  3. Pillar 1 validation (1 qubit, no environment) on hardware
  4. Enhanced publication-quality figure

Usage:
  python IBMquantum/run_ibm_enhanced.py --mode noise-only     # Just query backend props
  python IBMquantum/run_ibm_enhanced.py --mode pillar1        # Pillar 1 on hardware
  python IBMquantum/run_ibm_enhanced.py --mode errorbars      # Re-run Pillar 2 with error bars
  python IBMquantum/run_ibm_enhanced.py --mode all            # Everything
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

# ── Physical parameters ──
OMEGA = 1.0
G = 0.1
DT = 0.2
K_MAX = 20

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def load_api_key():
    key_path = os.path.join(SCRIPT_DIR, '..', 'apikey.json')
    with open(key_path) as f:
        return json.load(f)['apikey']


def get_backend_and_service():
    """Connect to IBM Quantum and return (service, backend)."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    api_key = load_api_key()
    service = QiskitRuntimeService(
        channel='ibm_quantum_platform',
        token=api_key
    )
    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=3
    )
    return service, backend


# ═══════════════════════════════════════════════════════════
# 1. NOISE CHARACTERIZATION
# ═══════════════════════════════════════════════════════════

def get_noise_properties(backend):
    """Extract T1, T2, gate error rates from backend."""
    print(f"\n{'='*60}")
    print(f"Backend Noise Properties: {backend.name}")
    print(f"{'='*60}")
    print(f"  Total qubits: {backend.num_qubits}")

    # Get target (calibration data)
    target = backend.target

    # Collect T1, T2 from qubit properties
    t1_vals = []
    t2_vals = []
    for q in range(min(backend.num_qubits, 10)):  # sample first 10
        props = target.qubit_properties
        if props and q < len(props) and props[q]:
            t1 = getattr(props[q], 't1', None)
            t2 = getattr(props[q], 't2', None)
            if t1 is not None:
                t1_vals.append(t1 * 1e6)  # convert to μs
            if t2 is not None:
                t2_vals.append(t2 * 1e6)

    # Gate error rates
    sx_errors = []
    cz_errors = []
    for op_name in target.operation_names:
        qargs_list = target.qargs_for_operation_name(op_name)
        if qargs_list is None:
            continue
        for qargs in qargs_list:
            prop = target[op_name].get(qargs, None)
            if prop and prop.error is not None:
                if op_name == 'sx' and len(qargs) == 1:
                    sx_errors.append(prop.error)
                elif op_name in ('cz', 'cx', 'ecr') and len(qargs) == 2:
                    cz_errors.append(prop.error)

    # Readout errors
    meas_errors = []
    for op_name in target.operation_names:
        if 'measure' in op_name.lower() or op_name == 'measure':
            qargs_list = target.qargs_for_operation_name(op_name)
            if qargs_list is None:
                continue
            for qargs in qargs_list:
                prop = target[op_name].get(qargs, None)
                if prop and prop.error is not None:
                    meas_errors.append(prop.error)

    results = {}

    if t1_vals:
        results['T1_mean_us'] = np.mean(t1_vals)
        results['T1_median_us'] = np.median(t1_vals)
        print(f"\n  T1: mean = {np.mean(t1_vals):.1f} μs, "
              f"median = {np.median(t1_vals):.1f} μs, "
              f"range = [{min(t1_vals):.1f}, {max(t1_vals):.1f}] μs")

    if t2_vals:
        results['T2_mean_us'] = np.mean(t2_vals)
        results['T2_median_us'] = np.median(t2_vals)
        print(f"  T2: mean = {np.mean(t2_vals):.1f} μs, "
              f"median = {np.median(t2_vals):.1f} μs, "
              f"range = [{min(t2_vals):.1f}, {max(t2_vals):.1f}] μs")

    if sx_errors:
        results['sx_error_mean'] = np.mean(sx_errors)
        results['sx_error_median'] = np.median(sx_errors)
        print(f"\n  Single-qubit gate (SX) error rate:")
        print(f"    mean = {np.mean(sx_errors)*100:.3f}%, "
              f"median = {np.median(sx_errors)*100:.3f}%")

    if cz_errors:
        results['2q_error_mean'] = np.mean(cz_errors)
        results['2q_error_median'] = np.median(cz_errors)
        print(f"  Two-qubit gate (CZ/ECR) error rate:")
        print(f"    mean = {np.mean(cz_errors)*100:.3f}%, "
              f"median = {np.median(cz_errors)*100:.3f}%")

    if meas_errors:
        results['meas_error_mean'] = np.mean(meas_errors)
        print(f"  Measurement error rate:")
        print(f"    mean = {np.mean(meas_errors)*100:.3f}%")

    results['backend'] = backend.name
    results['num_qubits'] = backend.num_qubits

    # Save to JSON
    out_path = os.path.join(OUT_DIR, 'backend_noise_properties.json')
    # Convert numpy types to Python types
    json_results = {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in results.items()}
    with open(out_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  → Saved {out_path}")

    return results


# ═══════════════════════════════════════════════════════════
# 2. PILLAR 1 — PURE DYNAMICS (1 QUBIT, NO ENVIRONMENT)
# ═══════════════════════════════════════════════════════════

def build_pillar1_circuits():
    """Build Trotter circuits for Pillar 1: H = (ω/2)σ_x, 1 qubit."""
    from qiskit import QuantumCircuit

    circuits = []
    for k in range(K_MAX + 1):
        qc = QuantumCircuit(1)
        for _ in range(k):
            qc.rx(OMEGA * DT, 0)
        circuits.append(qc)

    return circuits


def build_pillar1_observables():
    """Observables for single qubit: σ_x, σ_y, σ_z."""
    from qiskit.quantum_info import SparsePauliOp
    obs_x = SparsePauliOp.from_list([('X', 1.0)])
    obs_y = SparsePauliOp.from_list([('Y', 1.0)])
    obs_z = SparsePauliOp.from_list([('Z', 1.0)])
    return [obs_x, obs_y, obs_z]


def compute_pillar1_exact():
    """Exact Pillar 1: ⟨σ_z⟩(k) = cos(ωkdt)."""
    steps = np.arange(K_MAX + 1)
    sz = np.cos(OMEGA * steps * DT)
    sx = np.zeros(K_MAX + 1)
    sy = -np.sin(OMEGA * steps * DT)
    # Pure state → S = 0 always
    S = np.zeros(K_MAX + 1)
    return sx, sy, sz, S


def run_pillar1_hardware(backend):
    """Run Pillar 1 on real hardware."""
    from qiskit_ibm_runtime import EstimatorV2
    from qiskit.transpiler import generate_preset_pass_manager

    print("\n  Building Pillar 1 circuits (1 qubit)...")
    circuits = build_pillar1_circuits()
    obs_list = build_pillar1_observables()

    print("  Transpiling...")
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    isa_circuits = pm.run(circuits)

    pubs = []
    for isa_circ in isa_circuits:
        isa_obs = [ob.apply_layout(isa_circ.layout) for ob in obs_list]
        pubs.append((isa_circ, isa_obs))

    print(f"  Submitting {len(pubs)} PUBs for Pillar 1...")
    estimator = EstimatorV2(mode=backend)
    job = estimator.run(pubs)
    print(f"  Job ID: {job.job_id()}")
    print(f"  Waiting...")

    result = job.result()
    print("  Results received!")

    sx, sy, sz = [], [], []
    for pub_result in result:
        evs = pub_result.data.evs
        sx.append(float(evs[0]))
        sy.append(float(evs[1]))
        sz.append(float(evs[2]))

    return np.array(sx), np.array(sy), np.array(sz)


# ═══════════════════════════════════════════════════════════
# 3. PILLAR 2 WITH ERROR BARS (MULTIPLE RUNS)
# ═══════════════════════════════════════════════════════════

def build_pillar2_circuits():
    """Build Trotter circuits for Pillar 2: 3 qubits."""
    from qiskit import QuantumCircuit
    circuits = []
    for k in range(K_MAX + 1):
        qc = QuantumCircuit(3)
        for _ in range(k):
            qc.rx(OMEGA * DT, 0)
            qc.rxx(2 * G * DT, 0, 1)
            qc.rxx(2 * G * DT, 0, 2)
        circuits.append(qc)
    return circuits


def build_pillar2_observables():
    """Observables for qubit 0 in 3-qubit register."""
    from qiskit.quantum_info import SparsePauliOp
    obs_x = SparsePauliOp.from_list([('IIX', 1.0)])
    obs_y = SparsePauliOp.from_list([('IIY', 1.0)])
    obs_z = SparsePauliOp.from_list([('IIZ', 1.0)])
    return [obs_x, obs_y, obs_z]


def run_pillar2_with_errorbars(backend, n_runs=3):
    """Run Pillar 2 multiple times to estimate error bars."""
    from qiskit_ibm_runtime import EstimatorV2
    from qiskit.transpiler import generate_preset_pass_manager

    circuits = build_pillar2_circuits()
    obs_list = build_pillar2_observables()

    print(f"\n  Building & transpiling Pillar 2 circuits...")
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    isa_circuits = pm.run(circuits)

    pubs = []
    for isa_circ in isa_circuits:
        isa_obs = [ob.apply_layout(isa_circ.layout) for ob in obs_list]
        pubs.append((isa_circ, isa_obs))

    all_sx, all_sy, all_sz = [], [], []

    for run_idx in range(n_runs):
        print(f"\n  Run {run_idx+1}/{n_runs}...")
        estimator = EstimatorV2(mode=backend)
        job = estimator.run(pubs)
        print(f"    Job ID: {job.job_id()}")
        print(f"    Waiting...")

        result = job.result()
        print(f"    Results received!")

        sx, sy, sz = [], [], []
        for pub_result in result:
            evs = pub_result.data.evs
            sx.append(float(evs[0]))
            sy.append(float(evs[1]))
            sz.append(float(evs[2]))

        all_sx.append(sx)
        all_sy.append(sy)
        all_sz.append(sz)

    all_sx = np.array(all_sx)  # shape (n_runs, K_MAX+1)
    all_sy = np.array(all_sy)
    all_sz = np.array(all_sz)

    return all_sx, all_sy, all_sz


# ═══════════════════════════════════════════════════════════
# 4. EXACT REFERENCES
# ═══════════════════════════════════════════════════════════

def compute_pillar2_exact():
    """QuTiP exact evolution for Pillar 2 (3 qubits)."""
    import qutip as qt
    sigma_x = qt.sigmax()
    H_S = (OMEGA / 2) * qt.tensor(sigma_x, qt.qeye(2), qt.qeye(2))
    H_SE = (G * qt.tensor(sigma_x, sigma_x, qt.qeye(2)) +
            G * qt.tensor(sigma_x, qt.qeye(2), sigma_x))
    H_tot = H_S + H_SE

    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0), qt.basis(2, 0))

    sx_list, sy_list, sz_list, S_list = [], [], [], []
    for k in range(K_MAX + 1):
        t = k * DT
        U = (-1j * H_tot * t).expm()
        psi_t = U * psi0
        rho_S = psi_t.ptrace(0)

        sx_list.append(qt.expect(qt.sigmax(), rho_S))
        sy_list.append(qt.expect(qt.sigmay(), rho_S))
        sz_list.append(qt.expect(qt.sigmaz(), rho_S))
        S_list.append(float(qt.entropy_vn(rho_S)))

    return (np.array(sx_list), np.array(sy_list),
            np.array(sz_list), np.array(S_list))


def entropy_from_bloch(sx, sy, sz):
    """Compute S_eff from Bloch components."""
    r = np.sqrt(np.asarray(sx)**2 + np.asarray(sy)**2 + np.asarray(sz)**2)
    r = np.clip(r, 1e-15, 1.0 - 1e-15)
    p_plus = (1 + r) / 2
    p_minus = (1 - r) / 2
    S = -p_plus * np.log(p_plus) - p_minus * np.log(p_minus)
    return S


# ═══════════════════════════════════════════════════════════
# 5. PUBLICATION-QUALITY FIGURE
# ═══════════════════════════════════════════════════════════

def plot_enhanced(exact, hw_mean, hw_std, noise_props, backend_name,
                  pillar1_exact=None, pillar1_hw=None):
    """Publication-quality figure with error bars and noise annotation."""
    times = np.arange(K_MAX + 1) * DT
    sx_ex, sy_ex, sz_ex, S_ex = exact

    S_hw_mean = entropy_from_bloch(hw_mean[0], hw_mean[1], hw_mean[2])

    # Propagate error bars to entropy
    n_boot = 200
    S_hw_samples = []
    for _ in range(n_boot):
        sx_s = hw_mean[0] + np.random.randn(K_MAX+1) * hw_std[0]
        sy_s = hw_mean[1] + np.random.randn(K_MAX+1) * hw_std[1]
        sz_s = hw_mean[2] + np.random.randn(K_MAX+1) * hw_std[2]
        S_hw_samples.append(entropy_from_bloch(sx_s, sy_s, sz_s))
    S_hw_std = np.std(S_hw_samples, axis=0)

    # Determine number of panels
    has_p1 = pillar1_exact is not None and pillar1_hw is not None
    n_panels = 4 if has_p1 else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 6))

    # ── Panel 1: ⟨σ_z⟩(t) with error bars ──
    ax = axes[0]
    ax.plot(times, sz_ex, 'k-', linewidth=2.5, label='QuTiP exact', zorder=3)
    ax.errorbar(times, hw_mean[2], yerr=hw_std[2],
                fmt='ro', markersize=6, capsize=3, linewidth=1.2,
                label=f'IBM {backend_name}', zorder=4, alpha=0.85)
    ax.set_xlabel('Time t = k·dt', fontsize=13)
    ax.set_ylabel('⟨σ_z⟩', fontsize=13)
    ax.set_title('Damped dynamics (Pillar 2)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)

    # ── Panel 2: S_eff(t) with error bars — THE KEY RESULT ──
    ax = axes[1]
    ax.plot(times, S_ex, 'k-', linewidth=2.5, label='QuTiP exact', zorder=3)
    ax.errorbar(times, S_hw_mean, yerr=S_hw_std,
                fmt='ro', markersize=6, capsize=3, linewidth=1.2,
                label=f'IBM {backend_name}', zorder=4, alpha=0.85)
    ax.axhline(np.log(2), color='gray', ls=':', alpha=0.5, label='ln 2')
    ax.set_xlabel('Time t = k·dt', fontsize=13)
    ax.set_ylabel('S_eff(t)', fontsize=13)
    ax.set_title('Thermodynamic arrow (Pillar 2)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # ── Panel 3: |r| purity decay with error bars ──
    ax = axes[2]
    r_ex = np.sqrt(sx_ex**2 + sy_ex**2 + sz_ex**2)
    r_hw = np.sqrt(hw_mean[0]**2 + hw_mean[1]**2 + hw_mean[2]**2)
    # Error propagation for |r|
    r_hw_err = np.sqrt((hw_mean[0]*hw_std[0])**2 +
                        (hw_mean[1]*hw_std[1])**2 +
                        (hw_mean[2]*hw_std[2])**2) / np.maximum(r_hw, 1e-10)
    ax.plot(times, r_ex, 'k-', linewidth=2.5, label='QuTiP exact', zorder=3)
    ax.errorbar(times, r_hw, yerr=r_hw_err,
                fmt='ro', markersize=6, capsize=3, linewidth=1.2,
                label=f'IBM {backend_name}', zorder=4, alpha=0.85)
    ax.set_xlabel('Time t = k·dt', fontsize=13)
    ax.set_ylabel('|r(t)|', fontsize=13)
    ax.set_title('Purity decay', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # ── Panel 4 (optional): Pillar 1 — pure ⟨σ_z⟩ ──
    if has_p1:
        ax = axes[3]
        p1_sx_ex, p1_sy_ex, p1_sz_ex, _ = pillar1_exact
        ax.plot(times, p1_sz_ex, 'k-', linewidth=2.5,
                label='Analytic cos(ωkdt)', zorder=3)
        ax.plot(times, pillar1_hw[2], 'bs', markersize=6,
                label=f'IBM {backend_name}', zorder=4, alpha=0.85)
        ax.set_xlabel('Time t = k·dt', fontsize=13)
        ax.set_ylabel('⟨σ_z⟩', fontsize=13)
        ax.set_title('Pure dynamics (Pillar 1)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    # ── Noise annotation box ──
    noise_text_parts = []
    if 'T1_median_us' in noise_props:
        noise_text_parts.append(f"T₁ = {noise_props['T1_median_us']:.0f} μs")
    if 'T2_median_us' in noise_props:
        noise_text_parts.append(f"T₂ = {noise_props['T2_median_us']:.0f} μs")
    if '2q_error_median' in noise_props:
        noise_text_parts.append(
            f"2Q gate err = {noise_props['2q_error_median']*100:.2f}%")
    if 'sx_error_median' in noise_props:
        noise_text_parts.append(
            f"1Q gate err = {noise_props['sx_error_median']*100:.3f}%")

    if noise_text_parts:
        noise_text = f"{backend_name} calibration\n" + "\n".join(noise_text_parts)
        fig.text(0.99, 0.02, noise_text, fontsize=9,
                 ha='right', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                           edgecolor='gray', alpha=0.9),
                 family='monospace')

    fig.suptitle('Unified Relational Time Formula — IBM Quantum Hardware Validation',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])

    out_path = os.path.join(OUT_DIR, 'ibm_quantum_enhanced.png')
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"\n  → Saved {out_path}")

    return S_hw_mean, S_hw_std


def export_enhanced_csv(exact, hw_mean, hw_std, S_hw_mean, S_hw_std,
                        backend_name, pillar1_exact=None, pillar1_hw=None):
    """Export enhanced results with error bars."""
    times = np.arange(K_MAX + 1) * DT
    sx_ex, sy_ex, sz_ex, S_ex = exact

    out_path = os.path.join(OUT_DIR, 'table_ibm_enhanced.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['k', 't',
                  'sz_exact', 'S_exact',
                  'sz_hw_mean', 'sz_hw_std',
                  'S_hw_mean', 'S_hw_std']
        if pillar1_exact is not None:
            header += ['sz_p1_exact', 'sz_p1_hw']
        writer.writerow(header)

        for k in range(K_MAX + 1):
            row = [k, f'{times[k]:.2f}',
                   f'{sz_ex[k]:.6f}', f'{S_ex[k]:.6f}',
                   f'{hw_mean[2][k]:.6f}', f'{hw_std[2][k]:.6f}',
                   f'{S_hw_mean[k]:.6f}', f'{S_hw_std[k]:.6f}']
            if pillar1_exact is not None and pillar1_hw is not None:
                row += [f'{pillar1_exact[2][k]:.6f}',
                        f'{pillar1_hw[2][k]:.6f}']
            writer.writerow(row)

    print(f"  → Saved {out_path}")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='all',
                      choices=['noise-only', 'pillar1', 'errorbars', 'all'])
    parser.add_argument('--n-runs', type=int, default=3,
                      help='Number of repeated runs for error bars')
    args = parser.parse_args()

    print("=" * 60)
    print("IBM Quantum Enhanced Validation")
    print("Unified Relational Time Formula — Pillars 1 & 2")
    print("=" * 60)

    # Always connect
    print("\nConnecting to IBM Quantum...")
    service, backend = get_backend_and_service()
    print(f"  Backend: {backend.name} ({backend.num_qubits} qubits)")

    # ── Step 1: Noise properties (always) ──
    noise_props = get_noise_properties(backend)

    if args.mode == 'noise-only':
        print("\nDone (noise-only mode).")
        return

    # ── Step 2: Exact references ──
    print("\nComputing exact references...")
    p2_exact = compute_pillar2_exact()
    p1_exact = compute_pillar1_exact()
    print("  Done.")

    # ── Step 3: Pillar 1 on hardware ──
    pillar1_hw = None
    if args.mode in ('pillar1', 'all'):
        print(f"\n{'='*60}")
        print("Pillar 1 — Pure Schrödinger dynamics (1 qubit)")
        print(f"{'='*60}")
        pillar1_hw = run_pillar1_hardware(backend)

        # Quick summary
        p1_dev = np.max(np.abs(pillar1_hw[2] - p1_exact[2]))
        print(f"\n  Max |⟨σ_z⟩_hw - cos(ωkdt)| = {p1_dev:.4f}")
        print(f"  (Pure noise — zero Trotter error, zero entanglement)")

    # ── Step 4: Pillar 2 with error bars ──
    hw_all_sx, hw_all_sy, hw_all_sz = None, None, None
    if args.mode in ('errorbars', 'all'):
        print(f"\n{'='*60}")
        print(f"Pillar 2 — Thermodynamic arrow ({args.n_runs} runs)")
        print(f"{'='*60}")
        print(f"  Estimated QPU time: ~{args.n_runs * 30}s of 510s remaining")

        hw_all_sx, hw_all_sy, hw_all_sz = run_pillar2_with_errorbars(
            backend, n_runs=args.n_runs)

    # ── Step 5: Generate outputs ──
    if hw_all_sx is not None:
        hw_mean = (np.mean(hw_all_sx, axis=0),
                   np.mean(hw_all_sy, axis=0),
                   np.mean(hw_all_sz, axis=0))
        hw_std = (np.std(hw_all_sx, axis=0),
                  np.std(hw_all_sy, axis=0),
                  np.std(hw_all_sz, axis=0))
    else:
        # Load from previous run CSV if available
        prev_csv = os.path.join(OUT_DIR, 'table_ibm_quantum_validation.csv')
        if os.path.exists(prev_csv):
            print("\n  Loading previous hardware data for plotting...")
            import pandas as pd
            df = pd.read_csv(prev_csv)
            hw_cols = [c for c in df.columns if 'ibm' in c.lower()]
            if hw_cols:
                sx_col = [c for c in hw_cols if 'sx' in c]
                sy_col = [c for c in hw_cols if 'sy' in c]
                sz_col = [c for c in hw_cols if 'sz' in c]
                if sx_col and sy_col and sz_col:
                    hw_mean = (df[sx_col[0]].values,
                               df[sy_col[0]].values,
                               df[sz_col[0]].values)
                    # No error bars from single run — use shot noise estimate
                    shots = 4096
                    shot_err = 1.0 / np.sqrt(shots)
                    hw_std = (np.full(K_MAX+1, shot_err),
                              np.full(K_MAX+1, shot_err),
                              np.full(K_MAX+1, shot_err))
                else:
                    print("  ERROR: Could not find hardware columns in CSV")
                    return
            else:
                print("  ERROR: No hardware data found")
                return
        else:
            print("  ERROR: No hardware data available. Run with --mode errorbars")
            return

    print(f"\n{'='*60}")
    print("Generating enhanced outputs...")
    print(f"{'='*60}")

    S_hw_mean, S_hw_std = plot_enhanced(
        p2_exact, hw_mean, hw_std, noise_props, backend.name,
        pillar1_exact=p1_exact, pillar1_hw=pillar1_hw)

    export_enhanced_csv(p2_exact, hw_mean, hw_std, S_hw_mean, S_hw_std,
                       backend.name,
                       pillar1_exact=p1_exact, pillar1_hw=pillar1_hw)

    # ── Final summary ──
    S_exact = p2_exact[3]
    print(f"\n{'='*60}")
    print("ENHANCED VALIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Backend: {backend.name} ({backend.num_qubits} qubits)")
    if 'T1_median_us' in noise_props:
        print(f"  T1 = {noise_props['T1_median_us']:.0f} μs, "
              f"T2 = {noise_props.get('T2_median_us', 0):.0f} μs")
    if '2q_error_median' in noise_props:
        print(f"  2Q gate error = {noise_props['2q_error_median']*100:.2f}%")
    print(f"\n  Pillar 2 (arrow):")
    print(f"    S_eff exact:    0.000 → {S_exact[-1]:.3f}")
    print(f"    S_eff hardware: {S_hw_mean[0]:.3f} → {S_hw_mean[-1]:.3f} "
          f"± {S_hw_std[-1]:.3f}")
    print(f"    Agreement: {S_hw_mean[-1]/S_exact[-1]*100:.1f}% of exact")

    if pillar1_hw is not None:
        p1_dev = np.max(np.abs(pillar1_hw[2] - p1_exact[2]))
        print(f"\n  Pillar 1 (dynamics):")
        print(f"    Max |⟨σ_z⟩_hw - cos(ωkdt)| = {p1_dev:.4f}")

    print(f"\n  QPU budget used this session: ~{(args.n_runs * 30 + 15) if hw_all_sx is not None else 0}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
