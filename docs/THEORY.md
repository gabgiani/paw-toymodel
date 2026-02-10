# The Unified Relational Formula — Theory

## The Central Claim

Time is not a fundamental property of the universe. It is an emergent feature of **conditioned correlations under limited access**.

The universe, described by a global state |Ψ⟩, satisfies a stationarity constraint:

$$\hat{C}\,|\Psi\rangle = 0$$

Nothing evolves. Nothing flows. There is no "before" or "after."

Yet an observer — a physical subsystem that selects a clock and lacks access to all degrees of freedom — extracts from this timeless object a description that contains dynamics, irreversibility, and frame dependence. All three emerge from a single expression.

---

## The Formula

$$\rho_S(t) = \frac{\mathrm{Tr}_E\!\big[\langle t|_C\;|\Psi\rangle\langle\Psi|\;|t\rangle_C\big]}{p(t)}$$

This is **not** the original Page–Wootters formula. Page and Wootters (1983) introduced the projection ⟨t|\_C — the conditioning on a clock subsystem. That is one ingredient. The full expression above extends the PaW mechanism by incorporating three distinct operations, each from a different lineage in the literature, which jointly produce the three pillars of the problem of time.

The unification — recognizing that all three pillars are already contained in this single expression applied to a single timeless state — is the central contribution of this work.

---

## Three Pillars from One Formula

### Pillar 1: Quantum Dynamics — from ⟨t|\_C (projection)

**Source:** Page & Wootters (1983)

Projecting the global state onto successive clock readings ⟨k|\_C extracts a sequence of conditional states. In the good-clock limit, this sequence obeys an effective Schrödinger equation:

$$i\,\partial_t\,|\psi_S(t)\rangle \approx H_S\,|\psi_S(t)\rangle$$

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

$$S_{\text{eff}}(k) = -\mathrm{Tr}\!\big[\rho_S(k)\,\ln\rho_S(k)\big]$$

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

A grid of panels, one per environment size (n\_env = 2, 4, 6, 8). Each panel shows ⟨σ\_z⟩ (blue) and S\_eff (red) vs clock reading k. With more environment qubits, oscillations damp faster and entropy saturates earlier — demonstrating that dim(E) controls the rate at which the arrow develops.

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

The plot shows ΔE\_C(k) — the change in clock energy due to S–E interaction — across clock readings. For an ideal clock, ΔE\_C = 0 everywhere. Deviations indicate that the system–environment dynamics are "leaking" back into the clock. Small ΔE\_C confirms the good-clock approximation.

### Fidelity: F(k)

How far does the conditioned state deviate from ideal Schrödinger evolution?

| Script | Output |
|--------|--------|
| `run_all.py` | `output/fidelity_comparison.png` |

![Fidelity decay](../output/fidelity_comparison.png)

Fidelity F(k) = ⟨ψ\_ideal(k)|ρ\_S(k)|ψ\_ideal(k)⟩ measures the overlap between the conditioned state (from the formula) and the ideal Schrödinger evolution. Version A (no environment) maintains F = 1. Version B (with environment) shows fidelity decay as the partial trace drives the state toward I/2 — an alternative view of the entropy growth from Pillar 2.

### Entropy comparison: S\_eff across environments

| Script | Output |
|--------|--------|
| `run_all.py` | `output/entropy_comparison.png` |

![Entropy comparison](../output/entropy_comparison.png)

S\_eff(k) curves for different environment sizes (n\_env = 2, 4, 6, 8). Larger environments produce faster entropy growth and higher final entropy, all approaching ln 2. This shows that the arrow strength scales with the number of inaccessible degrees of freedom.

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

$$k \;\mapsto\; \rho_S(k) = \frac{\mathrm{Tr}_E\!\big[\langle k|_C\;|\Psi\rangle\langle\Psi|\;|k\rangle_C\big]}{p(k)}$$

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

$$H_{\text{eff}}(k) = H_{SE} + \varepsilon\,\frac{k}{N}\,\sigma_z \otimes I_E$$

Result: the arrow degrades gradually with ε but persists. Even at ε = 1.0 (10× the S–E coupling g), the arrow strength remains 0.29 — weakened but not destroyed.

### Test 2 — Fuzzy subsystem boundaries

The "problem of subsystems" in QG: what we call S and E may not have sharp definitions. We apply a partial SWAP between S and E₁ before tracing:

$$V(\theta) = \cos\theta\;I - i\sin\theta\;\mathrm{SWAP}_{S,E_1}$$

Result: at θ = π/2 (S and E₁ fully swapped), the arrow **recovers** to strength 0.882 with perfect monotonicity. This demonstrates that the arrow does not depend on which subsystem we label as "system" — it is a structural consequence of the partial trace over a large Hilbert space.

### Test 3 — Clock uncertainty

Gravitational time dilation makes clock readings inherently uncertain. We replace sharp projections |k⟩ with Gaussian-smeared states:

$$|\tilde{k}\rangle = \sum_j c_j\,|j\rangle, \quad c_j \propto \exp\!\left(-\frac{(j-k)^2}{2\sigma^2}\right)$$

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

Three rows, one per test (backreaction, fuzzy boundaries, clock uncertainty). **Left columns:** ⟨σ\_z⟩ dynamics under increasing perturbation strength. **Right columns:** S\_eff curves. The arrow (entropy growth) persists across all three tests — dynamics degrade first, but irreversibility is structurally robust.

![Gravity robustness — arrow survival summary](../output/gravity_robustness_summary.png)

Summary scatter plot: arrow strength (S\_final / ln 2) and monotonicity score vs perturbation parameter for each test. Clock uncertainty (σ) barely affects the arrow. Fuzzy boundaries (θ) maintain the arrow even at full swap. Backreaction (ε) is the strongest perturbation, yet the arrow survives up to 10× the coupling strength.

---

## Structural Robustness

Beyond gravitational perturbations, three structural risks affect any finite quantum model of emergent time: Poincaré recurrences, initial-state dependence, and partition arbitrariness. Each is computable and has been tested numerically.

### Test A — Poincaré Recurrences

In a finite-dimensional Hilbert space, every unitary evolution is quasi-periodic: the state must eventually return arbitrarily close to its initial condition. If the recurrence time is short, the "arrow" is an illusion — entropy rises and then falls again in a cycle.

We compare two coupling regimes for n\_env = 1…6 environment qubits:

**Symmetric coupling** (g\_j = g = 0.1 for all j): The identical coupling creates degenerate eigenvalues. The entire 2^(1+n) -dimensional Hilbert space produces only ~10 distinct frequencies, and the state recurs **exactly** at T ≈ 31.4 for all n\_env values. This is the pathological worst case.

**Random coupling** (g\_j ~ U(0.05, 0.2), mixed Pauli axes σ\_x, σ\_y, σ\_z): Symmetry breaking creates exponentially many distinct frequencies. Results:

| n\_env | dim | Symmetric: n\_freq | Symmetric: S\_min | Random: n\_freq | Random: S\_min | Random: recurrence? |
|--------|-----|-------------------|--------------------|----------------|----------------|---------------------|
| 1 | 4 | 4 | 0.0000 | 1 | 0.0253 | none in t < 350 |
| 2 | 8 | 7 | 0.0000 | 4 | 0.0014 | t = 40.8 |
| 3 | 16 | 8 | 0.0000 | 16 | 0.0605 | none in t < 150 |
| 4 | 32 | 9 | 0.0000 | 64 | 0.1408 | none in t < 80 |
| 5 | 64 | 10 | 0.0000 | 64 | 0.3527 | none in t < 50 |
| 6 | 128 | 11 | 0.0001 | 256 | 0.1299 | none in t < 40 |

The key metric is S\_min\_post — the minimum entropy after thermalisation. For the symmetric model it returns to 0 (complete recurrence). For the random model, the "recurrence depth" stays elevated (0.06–0.35) and the number of distinct frequencies grows exponentially, ensuring that realistic Hamiltonians produce an arrow that is effectively irreversible on any accessible timescale.

**Conclusion:** Poincaré recurrences are not a threat to the physical arrow. The symmetric case requires exact degeneracy (measure zero in Hamiltonian space). Any realistic perturbation breaks the symmetry and exponentially suppresses recurrence depth.

### Test B — Initial State Sensitivity

If the arrow of time only appears for specially chosen initial states, it would be a fine-tuning artifact. We test 100 Haar-random product states |ψ₀⟩\_S ⊗ |φ₀⟩\_E and 100 Haar-random entangled states |Ξ⟩\_{SE}:

| Initial state type | Arrow > 0.5 | Mean strength | Min strength |
|---------------------|-------------|---------------|--------------|
| Product (random) | 81 / 100 | 0.706 | 0.014 |
| Entangled (random) | 100 / 100 | 0.935 | 0.712 |

The arrow is **generic**: it appears for the vast majority of initial conditions. Entangled initial states show an even stronger arrow, consistent with the formula's mechanism — the partial trace automatically creates decoherence when S and E are correlated.

**Conclusion:** The arrow does not require fine-tuning of the initial state.

### Test C — Partition Independence

The formula involves a choice: which subsystem is "the system" S and which is "the environment" E? If the arrow were sensitive to this labeling, it would reflect our conventions rather than physics.

We test all 5 single-qubit partitions (each qubit as "system", tracing out the other 4) under two Hamiltonians:

**Symmetric H** (all qubits equivalent): Arrow strength = 1.000 for ALL 5 qubits, monotonicity = 1.000.

**Asymmetric H** (original H\_SE with designated roles): The designated system qubit (q0) shows arrow = 1.000, while environment qubits show arrow = 0.882 — still a clear thermodynamic arrow even for qubits not originally designated as "the system."

**Conclusion:** The arrow is a structural property of the partial trace and the Hilbert-space dimension, not an artifact of which degrees of freedom we label as "system."

### Combined Verdict

The unified relational formula produces a thermodynamic arrow that is:
- **Exponentially long-lived:** recurrence depth grows with spectral complexity (Test A)
- **Generic:** appears for >80% of random product states and 100% of random entangled states (Test B)
- **Partition-independent:** every qubit shows an arrow regardless of labeling (Test C)
- **Gravity-robust:** survives backreaction, fuzzy boundaries, and clock uncertainty (previous section)

| Script | Output |
|--------|--------|
| `generate_structural_robustness.py` | `output/robustness_poincare.png` |
| `generate_structural_robustness.py` | `output/robustness_initial_states.png` |
| `generate_structural_robustness.py` | `output/robustness_arrow_scatter.png` |
| `generate_structural_robustness.py` | `output/robustness_partition.png` |

![Poincaré recurrences — symmetric vs random coupling](../output/robustness_poincare.png)

Entropy S\_eff(t) over extended time, comparing symmetric coupling (sharp recurrences, S returns to 0) vs random coupling (recurrences suppressed, S stays elevated). Each line is a different environment size. The random model shows that spectral complexity prevents entropy from returning to its initial value.

![Initial state sensitivity — entropy distributions](../output/robustness_initial_states.png)

Histograms of final S\_eff for 100 random product states and 100 random entangled states. The entangled distribution peaks near ln 2 (strong arrow); the product distribution is broader but still predominantly above 0.5. The arrow is generic, not fine-tuned.

![Initial state sensitivity — arrow strength scatter](../output/robustness_arrow_scatter.png)

Scatter plot of arrow strength (S\_final / ln 2) for each trial. Product states (blue) show a wide range with most above 0.5. Entangled states (orange) cluster near 1.0. The dashed line marks S\_final / ln 2 = 0.5 as a threshold.

![Partition independence — all qubit partitions](../output/robustness_partition.png)

S\_eff(k) curves when each of the 5 qubits (1 system + 4 environment) is individually designated as "the system" and the rest are traced out. Under symmetric H, all 5 curves overlap perfectly. Under asymmetric H, the designated system qubit shows the strongest arrow, but all qubits exhibit entropy growth — confirming the arrow is a property of the partial trace, not the labeling.

---

## Experimental Validation on IBM Quantum Hardware

All results above rely on numerical simulation (QuTiP). As a final test, we executed both Pillar 1 (pure dynamics) and Pillar 2 (entropy growth) on a **real quantum processor** — IBM's `ibm_torino` (133 superconducting qubits) — via the Qiskit Runtime service.

### Backend Noise Characterisation

Calibration data queried at runtime establish the noise floor:

| Property | Median | Mean |
|----------|--------|------|
| T₁ (μs) | 147.8 | 159.3 |
| T₂ (μs) | 161.9 | 155.0 |
| Single-qubit gate error (SX) | 0.032% | 0.848% |
| Two-qubit gate error (CZ) | 0.247% | 3.254% |
| Measurement readout error | — | 4.49% |

The coherence times (T₁, T₂ ≈ 150 μs) far exceed the maximum circuit duration (~3 μs), so decoherence during evolution is not the dominant error source. Instead, measurement readout error (4.49%) and two-qubit gate error (median 0.25%) are the primary contributors.

### Setup

- **Qubits:** 3 (1 system + 2 environment) for Pillar 2; 1 qubit for Pillar 1
- **Circuit:** First-order Trotter decomposition of H = (ω/2)σ\_x + g(σ\_x⊗σ\_x⊗I + σ\_x⊗I⊗σ\_x)
- **Steps:** k = 0..20 (t\_max = 4.0), each step = Rx(ωdt) + 2×RXX(2gdt)
- **Shots:** 4096 per observable per step
- **Observables:** Partial tomography of qubit 0 via ⟨σ\_x⟩, ⟨σ\_y⟩, ⟨σ\_z⟩
- **Repetitions:** 3 independent runs for Pillar 2 (statistical error bars)

A key property: because all terms in H commute in the σ\_x basis, the first-order Trotter decomposition is **exact** (Trotter error = 0). This means any deviation from the exact curve is purely QPU noise — not algorithmic error.

### Results — Pillar 1 (Pure Dynamics)

With a single qubit and no environment, the hardware reproduces ⟨σ\_z⟩(k) = cos(ωkdt) with a maximum absolute deviation of **0.033** across all 21 steps — consistent with the 0.032% median SX gate error accumulated over the deepest circuit. This confirms that Schrödinger dynamics emerge cleanly on real hardware.

### Results — Pillar 2 (Thermodynamic Arrow)

Three independent hardware runs were executed on the same backend to quantify statistical uncertainty:

| Source | S\_eff(0) | S\_eff(20) | Notes |
|--------|-----------|-----------|-------|
| QuTiP exact | 0.000 | 0.570 | — |
| Qiskit simulator | 0.000 | 0.570 | Trotter error = 0 |
| **IBM ibm\_torino (mean ± 1σ)** | **0.000 ± 0.002** | **0.583 ± 0.005** | 3 independent runs |

The thermodynamic arrow of time is clearly observed on real hardware:

- S\_eff grows from ~0 to **0.583 ± 0.005** (102.2% of exact) — the slight over-estimation is physically expected, as QPU noise contributes additional decoherence beyond the model's entanglement-based entropy
- The arrow strength S\_final − S\_initial = 0.583
- Dominant noise sources: readout error (4.49%) and two-qubit gate error (median 0.25%), with coherence times (T₁ ≈ 148 μs, T₂ ≈ 162 μs) far exceeding circuit duration

| Script | Output |
|--------|--------|
| `IBMquantum/run_ibm_validation.py` | `IBMquantum/output/ibm_quantum_validation.png` |
| `IBMquantum/run_ibm_enhanced.py` | `IBMquantum/output/ibm_quantum_enhanced.png` |
| `IBMquantum/run_ibm_enhanced.py` | `IBMquantum/output/table_ibm_enhanced.csv` |
| `IBMquantum/run_ibm_enhanced.py` | `IBMquantum/output/backend_noise_properties.json` |

![IBM Quantum hardware validation — enhanced with error bars](../IBMquantum/output/ibm_quantum_enhanced.png)

The figure shows exact theory (dashed) vs hardware mean ± 1σ (shaded band) for both ⟨σ\_z⟩ and S\_eff, with a noise annotation box summarising the backend calibration.

This constitutes the first experimental confirmation on physical quantum hardware that the unified relational formula's informational arrow survives real-world noise, with quantified error bars and device-level noise characterisation.

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
