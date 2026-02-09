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

![Pillar 1 — Projection yields Schrödinger dynamics](../output/validation_pillar1.png)

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

![Pillar 2 — Partial trace yields the informational arrow](../output/validation_unified.png)

Left: damped ⟨σ\_z⟩. Center: entropy growth (the arrow). Right: Version A (reversible) vs Version B (irreversible) — same formula, different access structure.

**Scaling with environment size:**

The arrow strengthens with more environment qubits. Recurrences become exponentially suppressed:

| Script | Output |
|--------|--------|
| `run_all.py` | `output/multi_nenv_grid.png` |

![Multi-environment comparison](../output/multi_nenv_grid.png)

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

![Pillar 3 — Two clocks, two narratives, one universe](../output/validation_pillar3_two_clocks.png)

---

## Diagnostic Metrics

The framework includes three quantitative metrics to characterize clock quality and the transition from reversible to irreversible dynamics:

### Back-action: ΔE\_C(k)

How much does the system–environment interaction disturb the clock?

| Script | Output |
|--------|--------|
| `run_all.py` | `output/back_action.png` |

![Back-action on the clock](../output/back_action.png)

### Fidelity: F(k)

How far does the conditioned state deviate from ideal Schrödinger evolution?

| Script | Output |
|--------|--------|
| `run_all.py` | `output/fidelity_comparison.png` |

![Fidelity decay](../output/fidelity_comparison.png)

### Entropy comparison: S\_eff across environments

| Script | Output |
|--------|--------|
| `run_all.py` | `output/entropy_comparison.png` |

---

## Geometric Structure Underlying the Framework

The three pillars have a geometric interpretation that connects the previous numerical results to the mathematical structure of the theory.

### 1. The Fundamental Object

The fundamental geometric object is a **stationary pure state** |Ψ⟩ satisfying Ĉ|Ψ⟩ = 0, residing as a fixed point on the constraint hypersurface within the projective Hilbert space CP(H).

- There is no metric, no foliation, no temporal structure at this level.
- S\_eff = 0: the global state carries no entropy, no arrow.
- This is the **timeless object** from which everything else is extracted.

### 2. The Relational Bundle

Upon an operational tensor factorization H = H\_C ⊗ H\_S ⊗ H\_E, the conditional map

```
k  ↦  ρ_S(k) = Tr_E [ ⟨k|_C |Ψ⟩⟨Ψ| |k⟩_C ] / p(k)
```

defines a **section** of a trivial quantum bundle over the base space of distinguishable clock readings in C. The observer's projection ⟨k|\_C selects a particular fiber, generating apparent temporal evolution.

### 3. The Arrow as Geometry

The thermodynamic arrow is geometrically encoded as the **monotonic increase** of the von Neumann entropy S\_eff(k) along the curve ρ\_S(k) in the convex manifold of reduced density operators D(H\_S). The partial trace Tr\_E — a contractive CPTP map — systematically displaces states toward the maximally mixed interior.

In the **Bloch ball** representation (valid for the qubit system):
- **Version A** (n\_env = 0): the Bloch vector stays on the sphere surface (|r| = 1, pure, reversible) — tracing a great circle.
- **Version B** (n\_env = 4, Tr\_E): the Bloch vector spirals inward (|r| → 0.025, mixed, irreversible) — converging toward I/2 at the center.

The arrow of time is the direction of this spiral.

| Script | Output |
|--------|--------|
| `generate_geometry_plots.py` | `output/geometric_interpretation.png` |
| `generate_geometry_plots.py` | `output/bloch_trajectory.png` |

![Geometric interpretation — from timeless state to emergent temporal curve](../output/geometric_interpretation.png)

Left: |Ψ⟩ as a fixed point on the constraint surface Ĉ = 0 in CP(H). Center: the relational bundle over clock readings, with the section ρ\_S(k) connecting the fibers. Right: the trajectory in the Bloch disk (y–z plane) — Version B spirals from the surface toward I/2.

![Bloch trajectory — purity decay and entropy growth](../output/bloch_trajectory.png)

Left: Bloch disk showing Version A (circle on boundary) vs Version B (spiral toward center, colored by S\_eff). Right: Bloch radius decay (|r| → 0) mirroring entropy growth (S\_eff → ln 2). The two curves are dual descriptions of the same geometric fact: partial trace drives states toward the maximally mixed interior.

---

## Gravity Robustness

The formula operates within non-relativistic quantum mechanics with a fixed tensor-product structure H\_C ⊗ H\_S ⊗ H\_E. In quantum gravity, this structure may not be globally well-defined. Three computational tests probe whether the mechanism is fragile or structurally robust against perturbations that mimic gravitational effects.

### Test 1 — Clock backreaction

In the standard model, the clock C is ideal — it records time without affecting the system. In gravity, the clock's mass-energy would shift the system's energy levels. We model this by adding a k-dependent term:

```
H_eff(k) = H_SE + ε (k/N) σ_z ⊗ I_E
```

Result: the arrow degrades gradually with ε but persists. Even at ε = 1.0 (10× the S–E coupling g), the arrow strength remains 0.29 — weakened but not destroyed.

### Test 2 — Fuzzy subsystem boundaries

The "problem of subsystems" in QG: what we call S and E may not have sharp definitions. We apply a partial SWAP between S and E₁ before tracing:

```
V(θ) = cos(θ) I − i sin(θ) SWAP_{S,E₁}
```

Result: at θ = π/2 (S and E₁ fully swapped), the arrow **recovers** to strength 0.882 with perfect monotonicity. This demonstrates that the arrow does not depend on which subsystem we label as "system" — it is a structural consequence of the partial trace over a large Hilbert space.

### Test 3 — Clock uncertainty

Gravitational time dilation makes clock readings inherently uncertain. We replace sharp projections |k⟩ with Gaussian-smeared states:

```
|k̃⟩ = Σ_j c_j |j⟩,   c_j ∝ exp(−(j−k)²/(2σ²))
```

Result: the arrow is **essentially immune** to clock uncertainty. Even σ = 4.0 (smearing over ±12 clock ticks) yields arrow strength 0.997 with perfect monotonicity.

### Summary

| Test | Max perturbation | Arrow strength | Monotonicity |
|------|-----------------|----------------|--------------|
| Backreaction (ε) | 1.0 (10× coupling) | 0.290 | 0.586 |
| Fuzzy boundaries (θ) | π/2 (full swap) | 0.882 | 1.000 |
| Clock uncertainty (σ) | 4.0 (±12 ticks) | 0.997 | 1.000 |

**Conclusion:** the arrow of time is structurally robust — it is not an artifact of idealized assumptions. The dynamics (Pillar 1) degrade more readily, as expected: perfect cos(ωkdt) requires an ideal Hamiltonian. But irreversibility (Pillar 2) is generic: it survives fuzzy clocks, fuzzy boundaries, and backreaction.

| Script | Output |
|--------|--------|
| `generate_gravity_robustness.py` | `output/gravity_robustness_curves.png` |
| `generate_gravity_robustness.py` | `output/gravity_robustness_summary.png` |

![Gravity robustness — entropy and dynamics under perturbation](../output/gravity_robustness_curves.png)

![Gravity robustness — arrow survival summary](../output/gravity_robustness_summary.png)

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

*Step-by-step derivation: [Derivation](DERIVATION.md) — the formula developed from first principles, operation by operation.*

*Geometric companion: [Geometric Structure](GEOMETRY.md) — Bloch trajectory, relational bundle, and the arrow as geometry.*

*Boundary analysis: [The Omniscient Observer](GOD_OBSERVER.md) — what happens when the observer has complete access.*
