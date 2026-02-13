"""
Condition-Violation Tests for the Unified Relational Time Formula
=================================================================

Each FAQ condition states a requirement for the formula to work.
This script *deliberately violates* each condition and confirms
that the expected pillar degrades or disappears.

Tests:
  V1  High initial entropy     (violates FAQ 3, cond. i)
  V2  Unstable partition        (violates FAQ 3, cond. ii)
  V3  Zero S–E interaction      (violates FAQ 3, cond. iii)
  V4  Non-orthogonal clock      (violates FAQ 14, cond. i)
  V5  Recohering (wrapping) clock (violates FAQ 14, cond. ii)

Expected outcomes:
  V1 → arrow absent (S_eff starts at max, cannot grow)
  V2 → arrow erratic (no monotonic growth)
  V3 → arrow absent (S_eff = 0 for all k)
  V4 → dynamics blur (Pillar 1 degrades)
  V5 → dynamics non-monotonic (temporal ordering breaks)
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import os

os.makedirs("output", exist_ok=True)

# ── Shared parameters ───────────────────────────────────────────
N       = 30
dt      = 0.2
omega   = 1.0
g       = 0.1
n_env   = 4
ln2     = np.log(2)

sigma_x = qt.sigmax()
sigma_z = qt.sigmaz()


# ═══════════════════════════════════════════════════════════════
# BASELINE: standard formula for comparison
# ═══════════════════════════════════════════════════════════════

def build_H_total(omega, g, n_env):
    """H_S + H_SE on S⊗E (qubit dims)."""
    n = 1 + n_env
    ops = [qt.qeye(2)] * n
    ops[0] = sigma_x
    H_S = (omega / 2) * qt.tensor(ops)

    H_SE = 0
    for j in range(n_env):
        ops = [qt.qeye(2)] * n
        ops[0] = sigma_x
        ops[1 + j] = sigma_x
        H_SE = H_SE + g * qt.tensor(ops)

    return H_S + H_SE


def run_standard(N, dt, omega, g, n_env, initial_SE=None):
    """Run the standard PaW formula and return (sz, S_eff)."""
    H = build_H_total(omega, g, n_env)
    n = 1 + n_env

    if initial_SE is None:
        initial_SE = qt.tensor([qt.basis(2, 0)] * n)

    sz  = np.zeros(N)
    s_eff = np.zeros(N)

    for k in range(N):
        tk = k * dt
        U = (-1j * H * tk).expm()
        psi_k = U * initial_SE
        rho_S = psi_k.ptrace(0)
        sz[k]    = qt.expect(sigma_z, rho_S)
        s_eff[k] = qt.entropy_vn(rho_S)

    return sz, s_eff


# ── Baseline run ────────────────────────────────────────────────
sz_base, seff_base = run_standard(N, dt, omega, g, n_env)
ks = np.arange(N)


# ═══════════════════════════════════════════════════════════════
# V1: HIGH INITIAL ENTROPY  (FAQ 3, condition i)
#     Start from maximally mixed system state ρ_S(0) = I/2
#     → S_eff(0) = ln 2, no room to grow
# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("V1 — High initial entropy (violates FAQ 3, cond. i)")
print("=" * 60)

def run_v1_high_entropy():
    """
    Use an initial state |ψ_SE⟩ that gives ρ_S(0) = I/2.
    Construct: |ψ_SE⟩ = (1/√2)(|0⟩_S ⊗ |0⟩_E1 + |1⟩_S ⊗ |1⟩_E1) ⊗ |0⟩_{E2..E4}
    This is a Bell state on S–E1, so Tr_E[ρ_SE] = I/2.
    """
    n = 1 + n_env
    # Bell state on S and E1
    bell = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
          + qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
    # Tensor with |0⟩ for remaining environment qubits
    rest = qt.tensor([qt.basis(2, 0)] * (n_env - 1))
    initial_SE = qt.tensor(bell, rest)

    H = build_H_total(omega, g, n_env)
    sz  = np.zeros(N)
    s_eff = np.zeros(N)

    for k in range(N):
        tk = k * dt
        U = (-1j * H * tk).expm()
        psi_k = U * initial_SE
        rho_S = psi_k.ptrace(0)
        sz[k]    = qt.expect(sigma_z, rho_S)
        s_eff[k] = qt.entropy_vn(rho_S)

    return sz, s_eff

sz_v1, seff_v1 = run_v1_high_entropy()
print(f"  S_eff(0)  = {seff_v1[0]:.4f}  (expected ≈ ln2 = {ln2:.4f})")
print(f"  S_eff(29) = {seff_v1[-1]:.4f}")
print(f"  Arrow Δ   = {seff_v1[-1] - seff_v1[0]:.4f}  (baseline: {seff_base[-1] - seff_base[0]:.4f})")


# ═══════════════════════════════════════════════════════════════
# V2: UNSTABLE PARTITION  (FAQ 3, condition ii)
#     At each tick k, randomly rotate which qubit is "system"
#     → arrow should become erratic
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("V2 — Unstable partition (violates FAQ 3, cond. ii)")
print("=" * 60)

def run_v2_unstable_partition():
    """
    Evolve the standard state, but at each k trace out a *different*
    set of qubits as "environment" — rotating which qubit is S.
    """
    H = build_H_total(omega, g, n_env)
    n = 1 + n_env
    initial_SE = qt.tensor([qt.basis(2, 0)] * n)

    sz  = np.zeros(N)
    s_eff = np.zeros(N)

    for k in range(N):
        tk = k * dt
        U = (-1j * H * tk).expm()
        psi_k = U * initial_SE

        # At each k, pick a different qubit as "the system"
        sys_qubit = k % n   # cycles through 0, 1, 2, 3, 4
        rho_S = psi_k.ptrace(sys_qubit)
        sz[k]    = qt.expect(sigma_z, rho_S)
        s_eff[k] = qt.entropy_vn(rho_S)

    return sz, s_eff

sz_v2, seff_v2 = run_v2_unstable_partition()

# Monotonicity check
diffs = np.diff(seff_v2)
monotonic_frac = np.sum(diffs >= -1e-10) / len(diffs)
print(f"  Monotonicity fraction = {monotonic_frac:.3f}  (baseline: 1.000)")
print(f"  S_eff range: [{min(seff_v2):.4f}, {max(seff_v2):.4f}]")


# ═══════════════════════════════════════════════════════════════
# V3: ZERO INTERACTION  (FAQ 3, condition iii)
#     Set g = 0 → no entanglement forms → S_eff = 0 all k
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("V3 — Zero S–E interaction (violates FAQ 3, cond. iii)")
print("=" * 60)

sz_v3, seff_v3 = run_standard(N, dt, omega, g=0.0, n_env=n_env)
print(f"  S_eff(0)  = {seff_v3[0]:.6f}")
print(f"  S_eff(29) = {seff_v3[-1]:.6f}")
print(f"  Max S_eff = {max(seff_v3):.6f}")
print(f"  Arrow present? {'YES' if max(seff_v3) > 0.01 else 'NO (as expected)'}")


# ═══════════════════════════════════════════════════════════════
# V4: NON-ORTHOGONAL CLOCK  (FAQ 14, condition i)
#     Use Gaussian-overlapping clock states instead of |k⟩
#     → dynamics blur, Pillar 1 degrades
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("V4 — Non-orthogonal clock (violates FAQ 14, cond. i)")
print("=" * 60)

def run_v4_nonorthogonal_clock(sigma_clock):
    """
    Replace the sharp clock projection |k⟩⟨k|_C with a Gaussian
    smeared POVM: M_k = Σ_j w(j-k) |j⟩⟨j| where w is Gaussian.

    This mixes components from neighbouring "times", blurring dynamics.
    """
    H = build_H_total(omega, g, n_env)
    n = 1 + n_env
    initial_SE = qt.tensor([qt.basis(2, 0)] * n)

    # Pre-compute all evolved states
    evolved = []
    for k in range(N):
        tk = k * dt
        U = (-1j * H * tk).expm()
        evolved.append(U * initial_SE)

    sz  = np.zeros(N)
    s_eff = np.zeros(N)

    for k in range(N):
        # Gaussian weights centred on k
        weights = np.array([np.exp(-0.5 * ((j - k) / sigma_clock) ** 2)
                            for j in range(N)])
        weights /= weights.sum()

        # Mixed conditioned state: ρ_S(k) = Σ_j w_j ρ_S(j)
        rho_S = qt.Qobj(np.zeros((2, 2)))
        for j in range(N):
            rho_j = evolved[j].ptrace(0)
            rho_S = rho_S + weights[j] * rho_j

        sz[k]    = qt.expect(sigma_z, rho_S)
        s_eff[k] = qt.entropy_vn(rho_S)

    return sz, s_eff

# Test with several clock widths
clock_sigmas = [0.0, 1.0, 3.0, 6.0]
v4_results = {}
for sig in clock_sigmas:
    if sig == 0.0:
        v4_results[sig] = (sz_base, seff_base)
    else:
        v4_results[sig] = run_v4_nonorthogonal_clock(sig)

# Measure Pillar 1 degradation: deviation from cos(ω k dt) at early k
analytic = np.cos(omega * ks * dt)
print(f"  {'σ_clock':>10}  {'Max dev (k<15)':>15}  {'Fidelity loss':>14}")
for sig in clock_sigmas:
    sz_v4 = v4_results[sig][0]
    dev = np.max(np.abs(sz_v4[:15] - analytic[:15]))
    print(f"  {sig:>10.1f}  {dev:>15.4f}  {'baseline' if sig == 0 else f'{dev / max(1e-16, np.max(np.abs(sz_base[:15] - analytic[:15]))):.1f}× worse'}")


# ═══════════════════════════════════════════════════════════════
# V5: RECOHERING / WRAPPING CLOCK  (FAQ 14, condition ii)
#     Use a clock that wraps around: k → k mod (N/3)
#     → temporal ordering breaks, dynamics jump
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("V5 — Recohering (wrapping) clock (violates FAQ 14, cond. ii)")
print("=" * 60)

def run_v5_wrapping_clock(period):
    """
    The clock wraps around with a short period, so it re-visits
    earlier readings. The observer sees k_eff = k mod period.
    Temporal ordering breaks — the same clock reading maps to
    multiple physical configurations.
    """
    H = build_H_total(omega, g, n_env)
    n = 1 + n_env
    initial_SE = qt.tensor([qt.basis(2, 0)] * n)

    sz  = np.zeros(N)
    s_eff = np.zeros(N)

    for k in range(N):
        # The clock wraps: actual physical time is t_k,
        # but the clock *reads* k_eff = k mod period
        # This means at k=period, the clock shows 0 again
        k_eff = k % period
        tk = k_eff * dt    # observer uses clock reading, not true time

        U = (-1j * H * tk).expm()
        psi_k = U * initial_SE
        rho_S = psi_k.ptrace(0)
        sz[k]    = qt.expect(sigma_z, rho_S)
        s_eff[k] = qt.entropy_vn(rho_S)

    return sz, s_eff

wrap_period = 10
sz_v5, seff_v5 = run_v5_wrapping_clock(wrap_period)

# Check temporal ordering: does ⟨σ_z⟩(k) change direction at wrap points?
wraps = [k for k in range(1, N) if k % wrap_period == 0]
print(f"  Wrap period = {wrap_period}")
print(f"  Wrap points: {wraps}")
print(f"  S_eff at wraps: {[f'{seff_v5[k]:.4f}' for k in wraps]}")
print(f"  S_eff resets to ~0 at wraps? {'YES — ordering broken' if any(seff_v5[k] < 0.1 for k in wraps) else 'NO'}")

diffs_v5 = np.diff(seff_v5)
mono_v5 = np.sum(diffs_v5 >= -1e-10) / len(diffs_v5)
print(f"  Monotonicity fraction = {mono_v5:.3f}  (baseline: 1.000)")


# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY — Condition violation outcomes")
print("=" * 60)

results = [
    ("Baseline",        "—",                   seff_base[-1],  seff_base[-1] / ln2, 1.000),
    ("V1: High S(0)",   "FAQ 3 cond. i",       seff_v1[-1],    (seff_v1[-1] - seff_v1[0]) / ln2, None),
    ("V2: Unstable ∂",  "FAQ 3 cond. ii",      seff_v2[-1],    None, monotonic_frac),
    ("V3: g = 0",       "FAQ 3 cond. iii",     seff_v3[-1],    seff_v3[-1] / ln2, None),
    ("V4: σ_C = 6",     "FAQ 14 cond. i",      v4_results[6.0][1][-1], v4_results[6.0][1][-1] / ln2, None),
    ("V5: wrap=10",     "FAQ 14 cond. ii",     seff_v5[-1],    None, mono_v5),
]

print(f"  {'Test':<20} {'Violated':<18} {'S_eff(29)':>10} {'Arrow':>10} {'Monoton.':>10}")
print(f"  {'─'*20} {'─'*18} {'─'*10} {'─'*10} {'─'*10}")
for name, viol, sf, arrow, mono in results:
    a_str = f"{arrow:.3f}" if arrow is not None else "—"
    m_str = f"{mono:.3f}" if mono is not None else "—"
    print(f"  {name:<20} {viol:<18} {sf:>10.4f} {a_str:>10} {m_str:>10}")


# ═══════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Condition-Violation Tests: Breaking the Formula",
             fontsize=14, fontweight='bold')

# V1 — High initial entropy
ax = axes[0, 0]
ax.plot(ks, seff_base, 'k--', alpha=0.4, label='Baseline')
ax.plot(ks, seff_v1, 'r-o', markersize=3, label='V1: S(0) = ln 2')
ax.axhline(ln2, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('k')
ax.set_ylabel('S_eff')
ax.set_title('V1: High initial entropy\n(FAQ 3, cond. i)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# V2 — Unstable partition
ax = axes[0, 1]
ax.plot(ks, seff_base, 'k--', alpha=0.4, label='Baseline')
ax.plot(ks, seff_v2, 'b-s', markersize=3, label='V2: rotating S qubit')
ax.axhline(ln2, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('k')
ax.set_ylabel('S_eff')
ax.set_title('V2: Unstable partition\n(FAQ 3, cond. ii)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# V3 — Zero interaction
ax = axes[0, 2]
ax.plot(ks, seff_base, 'k--', alpha=0.4, label='Baseline')
ax.plot(ks, seff_v3, 'g-^', markersize=3, label='V3: g = 0')
ax.axhline(ln2, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('k')
ax.set_ylabel('S_eff')
ax.set_title('V3: Zero S–E interaction\n(FAQ 3, cond. iii)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# V4 — Non-orthogonal clock (dynamics)
ax = axes[1, 0]
for sig in clock_sigmas:
    sz_v4 = v4_results[sig][0]
    lbl = f'σ = {sig:.0f}' if sig > 0 else 'σ = 0 (ideal)'
    ax.plot(ks, sz_v4, '-', markersize=3, label=lbl, alpha=0.8)
ax.plot(ks, analytic, 'k:', alpha=0.3, label='cos(ωkdt)')
ax.set_xlabel('k')
ax.set_ylabel('⟨σ_z⟩')
ax.set_title('V4: Non-orthogonal clock\n(FAQ 14, cond. i) — dynamics blur')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# V4 — Non-orthogonal clock (entropy)
ax = axes[1, 1]
for sig in clock_sigmas:
    seff_v4 = v4_results[sig][1]
    lbl = f'σ = {sig:.0f}' if sig > 0 else 'σ = 0 (ideal)'
    ax.plot(ks, seff_v4, '-', markersize=3, label=lbl, alpha=0.8)
ax.axhline(ln2, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('k')
ax.set_ylabel('S_eff')
ax.set_title('V4: Non-orthogonal clock\n(FAQ 14, cond. i) — entropy')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# V5 — Wrapping clock
ax = axes[1, 2]
ax.plot(ks, seff_base, 'k--', alpha=0.4, label='Baseline')
ax.plot(ks, seff_v5, 'm-D', markersize=3, label=f'V5: wrap period={wrap_period}')
for wp in wraps:
    ax.axvline(wp, color='red', ls=':', alpha=0.3)
ax.axhline(ln2, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('k')
ax.set_ylabel('S_eff')
ax.set_title('V5: Recohering clock\n(FAQ 14, cond. ii) — ordering breaks')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('output/condition_violations.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nPlot saved: output/condition_violations.png")


# ═══════════════════════════════════════════════════════════════
# CSV export
# ═══════════════════════════════════════════════════════════════
import csv

with open('output/table_condition_violations.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['k', 'S_eff_baseline', 'S_eff_V1_high_entropy',
                'S_eff_V2_unstable', 'S_eff_V3_no_interaction',
                'S_eff_V4_sigma6', 'S_eff_V5_wrap10'])
    for k in range(N):
        w.writerow([k, f"{seff_base[k]:.6f}", f"{seff_v1[k]:.6f}",
                     f"{seff_v2[k]:.6f}", f"{seff_v3[k]:.6f}",
                     f"{v4_results[6.0][1][k]:.6f}", f"{seff_v5[k]:.6f}"])

print("CSV saved: output/table_condition_violations.csv")
print("\n✓ All condition-violation tests complete.")
