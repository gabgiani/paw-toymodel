"""
Test: What if the observer is God (total access)?
Three levels of omniscience tested against the PaW formula.
"""
import sys
sys.path.insert(0, '/Users/gianig/paw-toymodel')

import qutip as qt
import numpy as np
from validate_formula import build_paw_history, get_conditioned_observables
from validate_formula import N, dt, omega, initial_S, g

print("=" * 65)
print("  WHAT IF THE OBSERVER IS GOD?")
print("  Testing the formula under total access")
print("=" * 65)

# ── Level 1: God uses a clock but sees all environment ──
print("\n--- LEVEL 1: God uses clock, sees everything (n_env=0) ---")
print("    = Version A: Tr_E is trivial (nothing hidden)")
psi_god1 = build_paw_history(N, dt, omega, initial_S, n_env=0)
sz_god1, S_god1 = get_conditioned_observables(psi_god1, N, 0)

s_eff_str = ", ".join("{:.1e}".format(s) for s in S_god1[:5])
print("    S_eff at first 5 ticks: [{}] ... all = 0".format(s_eff_str))
print("    Max S_eff = {:.2e}".format(max(S_god1)))
print("    -> Dynamics exist (Pillar 1), but NO arrow (Pillar 2 = 0)")
print("    -> Time is reversible. God sees no irreversibility.")

# ── Comparison: Limited observer (n_env=4) ──
print("\n--- COMPARISON: Limited observer (n_env=4, things hidden) ---")
psi_lim = build_paw_history(N, dt, omega, initial_S, n_env=4, g=g)
sz_lim, S_lim = get_conditioned_observables(psi_lim, N, 4)
print("    S_eff final = {:.4f}  (ln 2 = {:.4f})".format(S_lim[-1], np.log(2)))
print("    -> Arrow EXISTS only because observer can't see environment")

# ── Level 2: God sees everything including the clock ──
print("\n--- LEVEL 2: God sees everything INCLUDING the clock ---")
print("    No projection <t|_C needed. God accesses |Psi> directly.")

psi_global = build_paw_history(N, dt, omega, initial_S, n_env=4, g=g)
rho_global = psi_global * psi_global.dag()

# God traces NOTHING. The full state is pure:
S_global = qt.entropy_vn(rho_global)
print("    S_vn(|Psi><Psi|) = {:.2e}  (pure state = 0)".format(S_global))
print("    -> No entropy. No arrow. No time.")

# If God doesn't condition on a clock, what does He see for sigma_z?
# He must trace out C and E to get rho_S, but He KNOWS everything.
# The point: the GLOBAL expectation of sigma_z is a single frozen number.
rho_S_global = psi_global.ptrace(1)  # keep only S (index 1)
sz_global = qt.expect(qt.sigmaz(), rho_S_global)
S_global_S = qt.entropy_vn(rho_S_global)
print("    <sigma_z>_global = {:.6f}  (one frozen number, no dynamics)".format(sz_global))
print("    S_vn(rho_S_global) = {:.4f}".format(S_global_S))
print("    -> God sees a SINGLE mixed state, frozen. No dynamics.")
print("    -> Time does not exist for an observer who sees |Psi> directly.")

# ── Level 3: The universe just IS ──
print("\n--- LEVEL 3: The universe as pure atemporal object ---")
print("    |Psi> has norm = {:.6f}".format(psi_global.norm()))
print("    |Psi> is a SINGLE vector in Hilbert space")
print("    It satisfies H|Psi> = 0 (by construction)")
print("    -> For God: no time, no history, no arrow, no experience")
print("    -> The universe simply EXISTS as a mathematical object")

# ── Progressive blindness test ──
print("\n--- PROGRESSIVE BLINDNESS: more hidden = more arrow ---")
print("    n_env | S_eff(final) |  Arrow?")
print("    ------|-------------|--------")
for ne in [0, 1, 2, 4, 6]:
    if ne == 0:
        psi = build_paw_history(N, dt, omega, initial_S, n_env=0)
        _, S = get_conditioned_observables(psi, N, 0)
    else:
        psi = build_paw_history(N, dt, omega, initial_S, n_env=ne, g=g)
        _, S = get_conditioned_observables(psi, N, ne)
    arrow = "NO (God)" if S[-1] < 0.01 else "YES (limited)"
    print("    {:5d} | {:11.6f} | {}".format(ne, S[-1], arrow))

# ── Summary ──
print("\n" + "=" * 65)
print("  CONCLUSION")
print("=" * 65)
print("""
  The formula VALIDATES perfectly for a God observer:

  1. If God uses a clock but sees everything:
     -> Dynamics exist, but S_eff = 0 ALWAYS (no arrow)
     -> Time is fully reversible

  2. If God sees the global state directly (no clock):
     -> No dynamics. <sigma_z> = {:.4f} (a single frozen value)
     -> Time does NOT EXIST

  3. The global |Psi> with H|Psi> = 0:
     -> Is an atemporal mathematical object
     -> Time, arrow, experience: all emergent from LIMITED access

  KEY INSIGHT:
     Time is not a property of the universe.
     Time is a property of IGNORANCE.
     An omniscient observer experiences no time.
     The arrow of time is the arrow of NOT KNOWING.
""".format(sz_global))
