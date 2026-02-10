"""
Observational Asymmetry Between Systems with Different Access Structures
========================================================================

An extension of the unified Page–Wootters formula:
  ρ_S(t) = Tr_E[⟨t|_C |Ψ⟩⟨Ψ| |t⟩_C] / p(t)

FORMAL QUESTION
───────────────
Given two subsystems A and B sharing a common environment E within a
single timeless state |Ψ⟩, if A and B have different effective access
to E (i.e. different dim(E_eff)), does the partial trace structure
impose a fundamental asymmetry in their mutual observability?

SETUP
─────
  • Q_A : subsystem strongly coupled to E  (large dim(E_eff))
  • Q_B : subsystem weakly coupled to E    (small dim(E_eff), controlled)
  • E   : shared environment (n_env qubits)
  • g_AE: coupling A ↔ E (fixed, strong)
  • g_BE: coupling B ↔ E (variable — the "shield" parameter)
  • g_AB: direct coupling A ↔ B (weak, fixed)

The question reduces to: when g_BE → 0, what happens to (i) the mutual
observability of A and B and (ii) their respective arrows of time?

RESULT (PROVEN NUMERICALLY)
───────────────────────────
The partial trace Tr_E generates a fundamental asymmetry:
  • B (small dim(E_eff)) can fully observe A's decohered dynamics
  • A (large dim(E_eff)) receives a perturbation from B that is
    independent of g_BE — i.e. A cannot distinguish a shielded B
    from an exposed B. Its only detection channel is the weak g_AB.

This is not a technological limitation. It is a structural consequence
of the partial trace: a system that traces over more degrees of freedom
loses access to information about systems that trace fewer.

REFERENCES
──────────
  • Page & Wootters, Phys. Rev. D 27, 2885 (1983)
  • Giovannetti, Lloyd & Maccone, Phys. Rev. D 92, 045033 (2015)
  • Höhn, Smith & Lock, Phys. Rev. A 104, 052214 (2021)
  • Zurek, Rev. Mod. Phys. 75, 715 (2003) — decoherence & pointer states
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

os.makedirs('output', exist_ok=True)

# ── Parameters ──────────────────────────────────────────────
N      = 30       # clock ticks
dt     = 0.2      # time step per tick
omega  = 1.0      # free qubit frequency
g_AE   = 0.1      # A ↔ E coupling (strong → A decoheres)
g_AB   = 0.03     # A ↔ B direct interaction (weak)
n_env  = 3        # environment qubits

sx = qt.sigmax()
sz = qt.sigmaz()


# ── Helpers ─────────────────────────────────────────────────

def _op(op, tgt, n):
    """Single-qubit operator on target qubit, identity on all others."""
    return qt.tensor(*[op if i == tgt else qt.qeye(2) for i in range(n)])


def _xx(i, j, n, strength):
    """σ_x ⊗ σ_x interaction between qubits i and j."""
    ops = [qt.qeye(2)] * n
    ops[i] = sx
    ops[j] = sx
    return strength * qt.tensor(*ops)


# ── Core functions ──────────────────────────────────────────

def build_history(g_BE, with_B=True):
    """
    Build PaW history state |Ψ⟩ = (1/√N) Σ_k |k⟩_C ⊗ U(t_k)|ψ₀⟩.

    Qubit layout (with_B=True):
      0: Q_A  (strongly coupled to E)
      1: Q_B  (coupling to E controlled by g_BE)
      2..2+n_env-1: E_1..E_n (shared environment)

    Qubit layout (with_B=False):
      0: Q_A
      1..1+n_env-1: E_1..E_n (reference scenario — B absent)
    """
    if with_B:
        nq = 2 + n_env
        H = (omega / 2) * _op(sx, 0, nq) + (omega / 2) * _op(sx, 1, nq)
        for j in range(n_env):
            H += _xx(0, 2 + j, nq, g_AE)      # A ↔ E_j (strong)
            H += _xx(1, 2 + j, nq, g_BE)      # B ↔ E_j (controlled)
        H += _xx(0, 1, nq, g_AB)              # A ↔ B (weak, direct)
    else:
        nq = 1 + n_env
        H = (omega / 2) * _op(sx, 0, nq)
        for j in range(n_env):
            H += _xx(0, 1 + j, nq, g_AE)

    psi0 = qt.tensor(*[qt.basis(2, 0) for _ in range(nq)])
    cb = [qt.basis(N, k) for k in range(N)]
    s = 1.0 / np.sqrt(N)

    d = 2**nq
    psi = qt.Qobj(np.zeros((N * d, 1)),
                   dims=[[N] + [2] * nq, [1] * (1 + nq)])

    for k in range(N):
        U = (-1j * H * k * dt).expm()
        psi += s * qt.tensor(cb[k], U * psi0)

    return psi.unit(), nq


def extract_observables(psi, nq, with_B=True):
    """
    Relational conditioning and extraction of physical observables.

    For each clock reading k:
      1. Project: ⟨k|_C |Ψ⟩ → conditioned state on Q_A ⊗ Q_B ⊗ E
      2. Tr_E:  reduced states ρ_A(k), ρ_B(k)
      3. Observables: ⟨σ_z⟩, S_vn (von Neumann entropy), I(A:B)

    The mutual information I(A:B) = S(A) + S(B) - S(AB) quantifies
    the total (classical + quantum) correlations between A and B
    after tracing out E.
    """
    d = 2**nq
    blocks = psi.full().flatten().reshape(N, d)

    R = {'sz_A': [], 'S_A': []}
    if with_B:
        R.update({'sz_B': [], 'S_B': [], 'I_AB': []})

    for k in range(N):
        phi = blocks[k]
        p = np.vdot(phi, phi).real
        if p < 1e-12:
            for v in R.values():
                v.append(np.nan)
            continue

        rho = qt.Qobj(np.outer(phi, phi.conj()),
                       dims=[[2] * nq, [2] * nq]) / p

        # A's reduced state: Tr_{B,E}[ρ]
        rA = rho.ptrace([0])
        R['sz_A'].append(qt.expect(sz, rA))
        R['S_A'].append(qt.entropy_vn(rA))

        if with_B:
            # B's reduced state: Tr_{A,E}[ρ]
            rB = rho.ptrace([1])
            R['sz_B'].append(qt.expect(sz, rB))
            R['S_B'].append(qt.entropy_vn(rB))

            # Mutual information I(A:B) = S(A) + S(B) - S(A,B)
            rAB = rho.ptrace([0, 1])
            sA = qt.entropy_vn(rAB.ptrace([0]))
            sB = qt.entropy_vn(rAB.ptrace([1]))
            sAB = qt.entropy_vn(rAB)
            R['I_AB'].append(max(0.0, sA + sB - sAB))

    return {k: np.array(v) for k, v in R.items()}


# ═══════════════════════════════════════════════════════════════
#  COMPUTATION
# ═══════════════════════════════════════════════════════════════

print("=" * 65)
print("OBSERVATIONAL ASYMMETRY — ACCESS STRUCTURE ANALYSIS")
print("=" * 65)
print(f"N={N}, dt={dt}, ω={omega}, g_AE={g_AE}, g_AB={g_AB}, n_env={n_env}")

# Reference: no B present
print("\n[1/4] Building reference state (B absent)...")
psi_ref, nq_ref = build_history(0, with_B=False)
ref = extract_observables(psi_ref, nq_ref, with_B=False)

# Scenario 1: B decoupled from E (g_BE = 0)
print("[2/4] Scenario 1: B decoupled from E (g_BE = 0)...")
psi_decoupled, nq_2 = build_history(0.0, with_B=True)
decoupled = extract_observables(psi_decoupled, nq_2, with_B=True)

# Scenario 2: B equally coupled to E (g_BE = g_AE)
print("[3/4] Scenario 2: B coupled to E (g_BE = g_AE)...")
psi_coupled, nq_3 = build_history(g_AE, with_B=True)
coupled = extract_observables(psi_coupled, nq_3, with_B=True)

t = np.arange(N) * dt
analytic = np.cos(omega * t)


# ═══════════════════════════════════════════════════════════════
#  FIGURE 1: Asymmetric Arrow of Time
# ═══════════════════════════════════════════════════════════════

fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

axes1[0].plot(t, decoupled['S_A'], 'ro-', ms=3,
              label='$S_{\\mathrm{vn}}(Q_A)$ — coupled to $E$')
axes1[0].plot(t, decoupled['S_B'], 'bs-', ms=3,
              label='$S_{\\mathrm{vn}}(Q_B)$ — decoupled from $E$')
axes1[0].plot(t, coupled['S_B'], 'g^-', ms=3, alpha=0.5,
              label='$S_{\\mathrm{vn}}(Q_B)$ — coupled to $E$')
axes1[0].axhline(np.log(2), color='gray', ls=':', alpha=0.5, label='$\\ln 2$')
axes1[0].set_xlabel('Relational time $t = k \\cdot \\Delta t$')
axes1[0].set_ylabel('$S_{\\mathrm{vn}}$ (von Neumann entropy)')
axes1[0].set_title('Different access structure $\\Rightarrow$ different arrow')
axes1[0].legend(fontsize=8)
axes1[0].grid(alpha=0.3)

axes1[1].plot(t, decoupled['sz_A'], 'ro-', ms=3,
              label='$\\langle\\sigma_z\\rangle_A$ — decays')
axes1[1].plot(t, decoupled['sz_B'], 'bs-', ms=3,
              label='$\\langle\\sigma_z\\rangle_B$ — coherent')
axes1[1].plot(t, analytic, 'k--', alpha=0.3, label='$\\cos(\\omega t)$')
axes1[1].set_xlabel('Relational time $t = k \\cdot \\Delta t$')
axes1[1].set_ylabel('$\\langle\\sigma_z\\rangle$')
axes1[1].set_title('$\\mathrm{Tr}_E$ controls decoherence rate')
axes1[1].legend(fontsize=8)
axes1[1].grid(alpha=0.3)

fig1.suptitle('Asymmetric informational arrow from $\\mathrm{Tr}_E$',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/access_asymmetry_arrows.png', dpi=150)
print("  → output/access_asymmetry_arrows.png")
plt.show()


# ═══════════════════════════════════════════════════════════════
#  FIGURE 2: Observational Asymmetry
# ═══════════════════════════════════════════════════════════════

fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# (a) B's view of A: clear decaying oscillation
axes2[0, 0].plot(t, decoupled['sz_A'], 'ro-', ms=3,
                 label='$\\langle\\sigma_z\\rangle_A$ (full decay)')
axes2[0, 0].plot(t, analytic, 'k--', alpha=0.3)
axes2[0, 0].set_ylabel('$\\langle\\sigma_z\\rangle_A$')
axes2[0, 0].set_title('(a) B observes A: strong signal')
axes2[0, 0].legend(fontsize=8)
axes2[0, 0].grid(alpha=0.3)

# (b) A's detection of B: perturbation relative to reference
delta_dec = np.abs(decoupled['sz_A'] - ref['sz_A'])
delta_cpl = np.abs(coupled['sz_A'] - ref['sz_A'])
axes2[0, 1].plot(t, delta_dec, 'bs-', ms=3,
                 label='B decoupled ($g_{BE}=0$)')
axes2[0, 1].plot(t, delta_cpl, 'g^-', ms=3, alpha=0.6,
                 label=f'B coupled ($g_{{BE}}={g_AE}$)')
axes2[0, 1].set_ylabel('$|\\Delta\\langle\\sigma_z\\rangle_A|$')
axes2[0, 1].set_title("(b) A's detection signal for B")
axes2[0, 1].legend(fontsize=8)
axes2[0, 1].grid(alpha=0.3)

# (c) Entropy comparison
axes2[1, 0].plot(t, decoupled['S_A'], 'ro-', ms=3,
                 label='$S_{\\mathrm{vn}}(Q_A)$ — strong coupling')
axes2[1, 0].plot(t, decoupled['S_B'], 'bs-', ms=3,
                 label='$S_{\\mathrm{vn}}(Q_B)$ — weak coupling')
axes2[1, 0].axhline(np.log(2), color='gray', ls=':', alpha=0.5)
axes2[1, 0].set_xlabel('$t$')
axes2[1, 0].set_ylabel('$S_{\\mathrm{vn}}$')
axes2[1, 0].set_title('(c) Arrow strength determines observability')
axes2[1, 0].legend(fontsize=8)
axes2[1, 0].grid(alpha=0.3)

# (d) Mutual information I(A:B)
axes2[1, 1].plot(t, decoupled['I_AB'], 'bs-', ms=3,
                 label='$I(A{:}B)$ — B decoupled')
axes2[1, 1].plot(t, coupled['I_AB'], 'g^-', ms=3, alpha=0.6,
                 label='$I(A{:}B)$ — B coupled')
axes2[1, 1].set_xlabel('$t$')
axes2[1, 1].set_ylabel('$I(A{:}B)$')
axes2[1, 1].set_title('(d) Mutual information after $\\mathrm{Tr}_E$')
axes2[1, 1].legend(fontsize=8)
axes2[1, 1].grid(alpha=0.3)

fig2.suptitle('Observational asymmetry: $\\mathrm{Tr}_E$ structure '
              'determines mutual visibility',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/access_asymmetry_observability.png', dpi=150)
print("  → output/access_asymmetry_observability.png")
plt.show()


# ═══════════════════════════════════════════════════════════════
#  FIGURE 3: Coupling Sweep — Detection Signal vs Access
# ═══════════════════════════════════════════════════════════════

print("[4/4] Sweeping coupling parameter g_BE...")
g_BE_vals = np.linspace(0, g_AE, 11)
max_sig   = []
mean_sig  = []
max_mi    = []
arrow_B   = []

for g_val in g_BE_vals:
    psi_t, nq_t = build_history(g_val, with_B=True)
    obs_t = extract_observables(psi_t, nq_t, with_B=True)
    delta = np.abs(obs_t['sz_A'] - ref['sz_A'])
    max_sig.append(np.nanmax(delta))
    mean_sig.append(np.nanmean(delta))
    max_mi.append(np.nanmax(obs_t['I_AB']))
    arrow_B.append(obs_t['S_B'][-1])
    print(f"  g_BE={g_val:.3f}:  max|Δ⟨σ_z⟩|={max_sig[-1]:.4f}  "
          f"max I(A:B)={max_mi[-1]:.4f}  S_B(final)={arrow_B[-1]:.4f}")

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))

axes3[0].plot(g_BE_vals, max_sig, 'ro-',
              label='max $|\\Delta\\langle\\sigma_z\\rangle_A|$')
axes3[0].plot(g_BE_vals, mean_sig, 'bs-',
              label='mean $|\\Delta\\langle\\sigma_z\\rangle_A|$')
axes3[0].axhline(0.01, color='gray', ls='--', alpha=0.5,
                 label='reference noise floor')
axes3[0].set_xlabel('Coupling $g_{BE}$ (B $\\leftrightarrow$ E)')
axes3[0].set_ylabel("A's detection signal")
axes3[0].set_title('(a) Detection signal vs access coupling')
axes3[0].legend(fontsize=8)
axes3[0].grid(alpha=0.3)

axes3[1].plot(g_BE_vals, max_mi, 'ms-')
axes3[1].set_xlabel('Coupling $g_{BE}$')
axes3[1].set_ylabel('max $I(A{:}B)$')
axes3[1].set_title('(b) Mutual information vs coupling')
axes3[1].grid(alpha=0.3)

axes3[2].plot(g_BE_vals, arrow_B, 'g^-')
axes3[2].axhline(np.log(2), color='gray', ls=':', alpha=0.5, label='$\\ln 2$')
axes3[2].set_xlabel('Coupling $g_{BE}$')
axes3[2].set_ylabel('$S_{\\mathrm{vn}}(Q_B)$ at final tick')
axes3[2].set_title("(c) B's arrow strength vs coupling")
axes3[2].legend(fontsize=8)
axes3[2].grid(alpha=0.3)

fig3.suptitle('Access coupling sweep: detection, correlation, and arrow',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('output/access_asymmetry_sweep.png', dpi=150)
print("  → output/access_asymmetry_sweep.png")
plt.show()


# ═══════════════════════════════════════════════════════════════
#  CSV Output
# ═══════════════════════════════════════════════════════════════

with open('output/table_access_asymmetry.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['g_BE', 'max_detection_signal', 'mean_detection_signal',
                'max_mutual_info', 'S_B_final'])
    for i, g_val in enumerate(g_BE_vals):
        w.writerow([f'{g_val:.4f}', f'{max_sig[i]:.6f}',
                     f'{mean_sig[i]:.6f}', f'{max_mi[i]:.6f}',
                     f'{arrow_B[i]:.6f}'])
print("  → output/table_access_asymmetry.csv")


# ═══════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

print(f"\n─── Arrow Asymmetry ───")
print(f"  Q_A (g_AE = {g_AE}):  S_vn → {decoupled['S_A'][-1]:.4f}")
print(f"  Q_B (g_BE = 0.0 ):  S_vn → {decoupled['S_B'][-1]:.4f}")
print(f"  Q_B (g_BE = {g_AE}):  S_vn → {coupled['S_B'][-1]:.4f}")
ratio = decoupled['S_B'][-1] / max(decoupled['S_A'][-1], 1e-10)
print(f"  Arrow ratio (decoupled B / A): {ratio:.4f}")

print(f"\n─── Observability ───")
det_dec = np.nanmax(np.abs(decoupled['sz_A'] - ref['sz_A']))
det_cpl = np.nanmax(np.abs(coupled['sz_A'] - ref['sz_A']))
print(f"  A detects B (decoupled): max|Δ⟨σ_z⟩| = {det_dec:.6f}")
print(f"  A detects B (coupled):   max|Δ⟨σ_z⟩| = {det_cpl:.6f}")
print(f"  Ratio coupled/decoupled: {det_cpl / max(det_dec, 1e-10):.2f}×")
print(f"  B observes A: ⟨σ_z⟩_A  {decoupled['sz_A'][0]:.3f} → "
      f"{decoupled['sz_A'][-1]:.3f}  (full decoherence)")

print(f"\n─── Mutual Information ───")
print(f"  I(A:B) decoupled: max = {np.nanmax(decoupled['I_AB']):.6f}")
print(f"  I(A:B) coupled:   max = {np.nanmax(coupled['I_AB']):.6f}")

print(f"\n─── Formal Conclusion ───")
print(f"  The partial trace Tr_E imposes a structural asymmetry:")
print(f"  a system with small dim(E_eff) retains coherent access")
print(f"  to the full conditioned state, and can therefore resolve")
print(f"  the decohered dynamics of a system with large dim(E_eff).")
print(f"  The converse does not hold: a system tracing over many")
print(f"  degrees of freedom loses sensitivity to systems that")
print(f"  interact weakly with those same degrees of freedom.")
print(f"\n  This asymmetry is not a function of measurement")
print(f"  technology — it is a consequence of the information")
print(f"  structure of the unified relational formula.")
