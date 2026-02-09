"""
PaW Toy Model Core — Page-Wootters Mechanism Demonstrator
=========================================================

Minimal demonstrator for Sections 5-6 of:
"The Observer as a Local Breakdown of Atemporality"
by Gabriel Giani Moreno (2026).

CORRECTION NOTE
---------------
The paper specifies H_S = (ω/2)σ_z and claims ⟨σ_z⟩(t) ≈ cos(ωt).
However, σ_z generates phase rotations only, leaving populations
(and hence ⟨σ_z⟩) time-independent for ANY initial state.

To reproduce ⟨σ_z⟩(t) = cos(ωt) we use H_S = (ω/2)σ_x with |ψ₀⟩ = |0⟩.
This is related to the paper's setup by a π/2 basis rotation and is
physically equivalent. Alternatively, keep H_S = (ω/2)σ_z and measure
⟨σ_x⟩ instead of ⟨σ_z⟩.
"""

import numpy as np
import qutip as qt
from dataclasses import dataclass, field


# ── Result containers ─────────────────────────────────────────────

@dataclass
class VersionAResult:
    """Results from Version A (no environment)."""
    k_values: np.ndarray
    sz_values: np.ndarray
    sz_theory: np.ndarray
    params: dict = field(default_factory=dict)


@dataclass
class VersionBResult:
    """Results from Version B (with environment)."""
    k_values: np.ndarray
    sz_values: np.ndarray
    s_eff_values: np.ndarray
    sz_theory: np.ndarray
    fidelity_values: np.ndarray
    params: dict = field(default_factory=dict)


# ── Version A ─────────────────────────────────────────────────────

def run_version_a(N: int = 30, dt: float = 0.2, omega: float = 1.0,
                  build_full_history: bool = False) -> VersionAResult:
    """
    PaW history state *without* environment.

    |Ψ⟩ = (1/√N) Σ_k |k⟩_C ⊗ U_S(t_k)|ψ₀⟩_S

    Parameters
    ----------
    N     : clock levels
    dt    : time step between ticks
    omega : system frequency
    build_full_history : if True, constructs the full tensor history
                         state in H_C ⊗ H_S (pedagogical/verification)

    Returns
    -------
    VersionAResult with conditional ⟨σ_z⟩ and theoretical curve.
    """
    H_S = (omega / 2) * qt.sigmax()
    psi0 = qt.basis(2, 0)

    ks = np.arange(N)
    sz_theory = np.cos(omega * ks * dt)
    sz = np.zeros(N)

    if build_full_history:
        # ── Full PaW history state construction (pedagogical) ──
        cb = [qt.basis(N, k) for k in range(N)]
        parts = []
        for k in range(N):
            U = (-1j * H_S * k * dt).expm()
            parts.append(qt.tensor(cb[k], U * psi0))

        psi_hist = parts[0]
        for p in parts[1:]:
            psi_hist = psi_hist + p
        psi_hist = psi_hist / np.sqrt(N)

        for k in range(N):
            proj = qt.tensor(cb[k] * cb[k].dag(), qt.qeye(2))
            cond = proj * psi_hist
            pk = cond.norm() ** 2
            if pk > 1e-14:
                rho_S = cond.ptrace(1) / pk
                sz[k] = qt.expect(qt.sigmaz(), rho_S)
    else:
        # ── Efficient: equivalent direct computation ──
        for k in range(N):
            U = (-1j * H_S * k * dt).expm()
            sz[k] = qt.expect(qt.sigmaz(), U * psi0)

    return VersionAResult(ks, sz, sz_theory,
                          {'N': N, 'dt': dt, 'omega': omega})


# ── Version B helpers ─────────────────────────────────────────────

def _build_H_total(omega: float, g: float, n_env: int):
    """
    Build H_total = H_S + H_SE on H_S ⊗ H_E (all qubits).

    H_S  = (ω/2) σ_x^(S) ⊗ I_E
    H_SE = g Σ_j σ_z^(S) ⊗ σ_z^(E_j)
    H_E  = 0  (inert environment for simplicity)
    """
    n = 1 + n_env

    # H_S = (ω/2) σ_x on system, identity on environment
    ops = [qt.qeye(2)] * n
    ops[0] = qt.sigmax()
    H_S = (omega / 2) * qt.tensor(ops)

    # H_SE = g Σ_j σ_x^(S) ⊗ σ_x^(E_j)
    # Note: σ_z⊗σ_z fails when E starts in |0⟩ (eigenstate of σ_z)
    # σ_x⊗σ_x creates transitions since σ_x|0⟩ = |1⟩
    terms = []
    for j in range(n_env):
        ops = [qt.qeye(2)] * n
        ops[0] = qt.sigmax()
        ops[1 + j] = qt.sigmax()
        terms.append(g * qt.tensor(ops))

    H_SE = terms[0]
    for t in terms[1:]:
        H_SE = H_SE + t

    return H_S + H_SE


def run_version_b(N: int = 30, dt: float = 0.2, omega: float = 1.0,
                  g: float = 0.1, n_env: int = 4) -> VersionBResult:
    """
    PaW history state *with* environment.

    |Ψ⟩ = (1/√N) Σ_k |k⟩_C ⊗ U_SE(t_k)|ψ₀, e₀⟩

    Conditions on clock, traces out E, and computes ⟨σ_z⟩(k),
    S_eff(k), and fidelity vs ideal Schrödinger evolution.
    """
    H = _build_H_total(omega, g, n_env)

    # |ψ₀⟩ = |0⟩_S ⊗ |0⟩^{⊗n_env}
    n = 1 + n_env
    psi0_SE = qt.tensor([qt.basis(2, 0)] * n)
    rho0_S = qt.basis(2, 0) * qt.basis(2, 0).dag()
    H_ideal = (omega / 2) * qt.sigmax()

    ks = np.arange(N)
    sz = np.zeros(N)
    s_eff = np.zeros(N)
    fid = np.zeros(N)
    sz_th = np.cos(omega * ks * dt)

    for k in range(N):
        tk = k * dt
        U = (-1j * H * tk).expm()
        psi_k = U * psi0_SE
        rho_S = psi_k.ptrace(0)        # trace out all env qubits

        sz[k] = qt.expect(qt.sigmaz(), rho_S)
        s_eff[k] = qt.entropy_vn(rho_S)

        # Fidelity with ideal Schrödinger evolution (no coupling)
        Uid = (-1j * H_ideal * tk).expm()
        rho_id = Uid * rho0_S * Uid.dag()
        fid[k] = qt.fidelity(rho_S, rho_id) ** 2

    return VersionBResult(ks, sz, s_eff, sz_th, fid,
                          {'N': N, 'dt': dt, 'omega': omega,
                           'g': g, 'n_env': n_env})


# ── Clock metrics ────────────────────────────────────────────────

def clock_back_action(N: int, dt: float):
    """
    Analytical back-action ΔE_C(k) for the finite Salecker–Wigner clock.

    H_C = (2π / N dt) Σ_k k |k⟩⟨k|
    ⟨H_C⟩_uncond = (2π / N dt) · (N−1)/2
    ΔE_C(k) = (2π / N dt) · (k − (N−1)/2)

    Note: trivially linear for orthogonal clock readouts.
    For non-ideal (overlapping) clock POVMs the metric becomes
    non-trivial, but this is beyond the current toy model.
    """
    omega_C = 2 * np.pi / (N * dt)
    ks = np.arange(N)
    return ks, omega_C * (ks - (N - 1) / 2)
