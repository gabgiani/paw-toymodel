# The Unified Relational Formula — Theory

## The Central Claim

Time is not a fundamental property of the universe. It is an emergent feature of **conditioned correlations under limited access**.

The universe, described by a global state |Ψ⟩, satisfies a stationarity constraint:

```
Ĉ |Ψ⟩ = 0
```

Nothing evolves. Nothing flows. There is no "before" or "after."

Yet an observer — a physical subsystem that selects a clock and lacks access to all degrees of freedom — extracts from this timeless object a description that contains dynamics, irreversibility, and frame dependence. All three emerge from a single expression.

---

## The Formula

```
                    Tr_E [ ⟨t|_C  |Ψ⟩⟨Ψ|  |t⟩_C ]
    ρ_S(t)   =    ─────────────────────────────────
                                p(t)
```

This is **not** the original Page–Wootters formula. Page and Wootters (1983) introduced the projection ⟨t|\_C — the conditioning on a clock subsystem. That is one ingredient. The full expression above extends the PaW mechanism by incorporating three distinct operations, each from a different lineage in the literature, which jointly produce the three pillars of the problem of time.

The unification — recognizing that all three pillars are already contained in this single expression applied to a single timeless state — is the central contribution of this work.

---

## Three Pillars from One Formula

### Pillar 1: Quantum Dynamics — from ⟨t|\_C (projection)

**Source:** Page & Wootters (1983)

Projecting the global state onto successive clock readings ⟨k|\_C extracts a sequence of conditional states. In the good-clock limit, this sequence obeys an effective Schrödinger equation:

```
i ∂_t |ψ_S(t)⟩ ≈ H_S |ψ_S(t)⟩
```

No external time parameter is needed. The ordering comes entirely from correlations between the clock subsystem C and the system S.

**Numerical validation:** With no environment (n\_env = 0), the conditional ⟨σ\_z⟩(k) oscillates as cos(ωkdt) with machine-precision agreement (max deviation ~ 4×10⁻¹⁶).

| Script | Output |
|--------|--------|
| `validate_formula.py` | `output/validation_pillar1.png` |
| `run_all.py` | `output/version_A_oscillation.png` |

![Pillar 1 — Projection yields Schrödinger dynamics](output/validation_pillar1.png)

The blue dots are computed from the formula; the dashed line is the analytic cos(ωkdt). They overlap exactly.

---

### Pillar 2: Thermodynamic Arrow — from Tr\_E (partial trace)

**Source:** Shaari (2014), extended in this work

When the observer cannot access environmental degrees of freedom E, the partial trace Tr\_E produces a mixed state ρ\_S(t) even though the global |Ψ⟩ is pure. The effective entropy:

```
S_eff(k) = −Tr[ρ_S(k) ln ρ_S(k)]
```

grows with the clock reading k. This is an **informational arrow of time**: irreversibility that arises not from non-unitary dynamics, not from special initial conditions, but **solely from the observer's restricted access**.

**Numerical validation:** With n\_env = 4 environment qubits, oscillations damp and S\_eff grows monotonically from 0 to ln 2 ≈ 0.693.

| Script | Output |
|--------|--------|
| `validate_formula.py` | `output/validation_unified.png` |
| `run_all.py` | `output/version_B_n4.png`, `output/entropy_comparison.png` |

![Pillar 2 — Partial trace yields the informational arrow](output/validation_unified.png)

Left: damped ⟨σ\_z⟩. Center: entropy growth (the arrow). Right: Version A (reversible) vs Version B (irreversible) — same formula, different access structure.

**Scaling with environment size:**

The arrow strengthens with more environment qubits. Recurrences become exponentially suppressed:

| Script | Output |
|--------|--------|
| `run_all.py` | `output/multi_nenv_grid.png` |

![Multi-environment comparison](output/multi_nenv_grid.png)

---

### Pillar 3: Observer-Dependent Time — from the locality of C

**Source:** Höhn, Smith & Lock (2021), operationalized in this work

The clock C is a physical subsystem, not a universal parameter. A different observer choosing a different clock C′ (with different spacing dt′) will perform the same three operations on the same global state |Ψ⟩ and obtain a **different** temporal description.

Neither description is wrong. Neither is more fundamental. The difference is the content of Pillar 3: time is relational.

**Numerical validation:** Two clocks (dt = 0.20 vs dt = 0.35) applied to the same global state with the same 4-qubit environment produce divergent dynamics:

| k | ⟨σ\_z⟩ (C₁) | S\_eff (C₁) | ⟨σ\_z⟩ (C₂) | S\_eff (C₂) |
|---|-----------|-----------|-----------|-----------|
| 0 | 1.000 | 0.000 | 1.000 | 0.000 |
| 5 | 0.498 | 0.164 | −0.139 | 0.348 |
| 10 | −0.300 | 0.405 | −0.320 | 0.633 |
| 20 | −0.154 | 0.665 | 0.001 | 0.693 |
| 29 | 0.023 | 0.693 | −0.029 | 0.692 |

Both observers reach the same asymptotic entropy (ln 2), but their entire dynamical narrative differs.

| Script | Output |
|--------|--------|
| `generate_pillar3_plot.py` | `output/validation_pillar3_two_clocks.png` |
| | `output/table_pillar3_two_clocks.csv` |

![Pillar 3 — Two clocks, two narratives, one universe](output/validation_pillar3_two_clocks.png)

---

## Diagnostic Metrics

The framework includes three quantitative metrics to characterize clock quality and the transition from reversible to irreversible dynamics:

### Back-action: ΔE\_C(k)

How much does the system–environment interaction disturb the clock?

| Script | Output |
|--------|--------|
| `run_all.py` | `output/back_action.png` |

![Back-action on the clock](output/back_action.png)

### Fidelity: F(k)

How far does the conditioned state deviate from ideal Schrödinger evolution?

| Script | Output |
|--------|--------|
| `run_all.py` | `output/fidelity_comparison.png` |

![Fidelity decay](output/fidelity_comparison.png)

### Entropy comparison: S\_eff across environments

| Script | Output |
|--------|--------|
| `run_all.py` | `output/entropy_comparison.png` |

---

## The Observer as an Anomaly

The three pillars converge on a single insight:

> **The observer is not the center of the universe. The observer is the anomaly.**

An observer is a physical system that:
1. Selects a degree of freedom as a temporal reference (clock)
2. Lacks access to other degrees of freedom (environment)
3. Calls "before" and "after" the ordering that emerges from that loss

Time does not flow into the observer. The observer generates it by existing as a constrained subsystem inside an atemporal whole.

---

## Attribution

| Component of the formula | Origin |
|--------------------------|--------|
| ⟨t\|\_C (projection onto clock) | Page & Wootters (1983) |
| Tr\_E (partial trace over environment) | Shaari (2014) |
| C is local (temporal QRFs) | Höhn, Smith & Lock (2021) |
| **Unification: three pillars from one expression** | **This work** |

---

*Next: [Scripts & Outputs](SCRIPTS.md) — detailed guide to reproducing all results.*

*Boundary analysis: [The Omniscient Observer](GOD_OBSERVER.md) — what happens when the observer has complete access.*
