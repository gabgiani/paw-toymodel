# The Unified Relational Time Formula — Theory

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

The unified relational time formula produces a thermodynamic arrow that is:
- **Exponentially long-lived:** recurrence depth grows with spectral complexity (Test A)
- **Generic:** appears for >80% of random product states and 100% of random entangled states (Test B)
- **Partition-independent:** every qubit shows an arrow regardless of labeling (Test C)
- **Gravity-robust:** survives backreaction, fuzzy boundaries, and clock uncertainty (previous section)
- **Minimally specified:** every stated condition is individually necessary — violating any one produces measurable failure (Condition Necessity Tests below)

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

### Condition Necessity Tests (Contrapositiva)

The robustness tests above show that the formula **works** under perturbation. But are the conditions stated in the [FAQ](FAQ.md) genuinely **necessary**? To answer this, we run five tests that deliberately **violate** each condition and confirm that the expected pillar degrades or disappears. This is the contrapositiva: if violating the condition breaks the result, then the condition is necessary.

Each test modifies exactly one assumption while keeping all others intact:

| Test | Condition violated | FAQ reference | What breaks | Observed result |
|------|-------------------|---------------|-------------|-----------------|
| V1 — High initial entropy | S\_eff(0) ≈ 0 (low initial entropy) | [FAQ 3, cond. i](FAQ.md#3-doesnt-the-partial-trace-just-assume-irreversibility) | Arrow absent | S\_eff = ln 2 = 0.693 at all k (starts at maximum, cannot grow) |
| V2 — Unstable partition | Stable S–E partition | [FAQ 3, cond. ii](FAQ.md#3-doesnt-the-partial-trace-just-assume-irreversibility) | Arrow erratic | S\_eff jumps discontinuously every 5 ticks (partition reshuffles) |
| V3 — Zero interaction | H\_SE creates entanglement | [FAQ 3, cond. iii](FAQ.md#3-doesnt-the-partial-trace-just-assume-irreversibility) | Arrow absent | S\_eff = 0.000 at all k (no entanglement forms) |
| V4 — Non-orthogonal clock | ⟨j\|k⟩\_C ≈ δ\_{jk} (orthogonal clock) | [FAQ 14, cond. i](FAQ.md#14-what-conditions-must-the-clock-satisfy) | Dynamics blur | S\_eff starts at 0.395 instead of 0 (temporal resolution lost) |
| V5 — Wrapping clock | Monotonic clock progression | [FAQ 14, cond. ii](FAQ.md#14-what-conditions-must-the-clock-satisfy) | Dynamics non-monotonic | S\_eff resets to 0 every 10 ticks (clock wraps around) |

**Interpretation:** Every condition is genuinely necessary. The formula is not over-specified — removing any single condition produces a measurable, qualitative failure in the corresponding pillar.

| Script | Output |
|--------|--------|
| `generate_condition_violations.py` | `output/condition_violations.png` |
| `generate_condition_violations.py` | `output/table_condition_violations.csv` |

![Condition necessity tests — 5 violations vs baseline](../output/condition_violations.png)

S\_eff(k) for 30 clock ticks under each violation (colored) compared to the baseline (black dashed). V1 (high initial entropy) is flat at ln 2. V2 (unstable partition) shows discontinuous jumps every 5 ticks. V3 (zero interaction) is flat at 0. V4 (non-orthogonal clock) has an elevated starting point. V5 (wrapping clock) resets periodically. The baseline shows the expected monotonic growth from 0 to ln 2 — confirming that all five conditions are individually necessary.

---

## Clock Orientation Covariance

The robustness tests above show that the informational arrow survives perturbations. We now demonstrate a deeper algebraic symmetry: the Unified Formula is **covariant** under arbitrary relabelling of the clock basis.

### Clock Orientation Covariance Theorem

Let π ∈ S_N be any permutation of the clock labels {0, 1, ..., N−1}. Define the permuted conditioning map:

> ρ_S^π(j) = Tr_E[ ⟨π(j)|_C |Ψ⟩⟨Ψ| |π(j)⟩_C ] / p(π(j))

**Theorem.** For a history state |Ψ⟩ satisfying Ĉ|Ψ⟩ = 0:

> **ρ_S^π(j) = ρ_S(π(j))** for all π ∈ S_N, for all j ∈ {0, ..., N−1}

Permuting the clock labels and then conditioning on slot j yields the same reduced state as conditioning directly on slot π(j).

**Proof.** The history state decomposes into N blocks |ψ_k⟩_SE. Relabelling the clock basis by π reindexes these blocks: slot j of the permuted state contains |ψ_{π(j)}⟩_SE. Since the Unified Formula — projection, normalization, partial trace — acts only on the block found at the conditioned slot, the output at slot j under relabelling π equals the output at the original slot π(j). No assumption about Ĥ or its symmetries is used; this is a purely algebraic identity on the conditioning structure. ∎

**Key properties:**
- **Algebraic** (not dynamical): holds for any Hamiltonian
- **Unitary** (not anti-unitary): permutations are basis relabellings, not complex conjugation
- **Relational** (not external): describes how conditioning results transform under internal relabelling

**Numerical verification:** Six distinct permutations tested (identity, reversal, two random shuffles, even-first reorder, cyclic shift). All produce error = 0 to machine precision.

| Script | Output |
|--------|--------|
| `generate_covariance_theorem.py` | `output/covariance_theorem_permutations.png` |
| `generate_covariance_theorem.py` | `output/covariance_theorem_vs_Tsymmetry.png` |
| `generate_covariance_theorem.py` | `output/covariance_theorem_combined.png` |
| `generate_covariance_theorem.py` | `output/table_covariance_theorem.csv` |

![Covariance theorem — permutation invariance](../output/covariance_theorem_combined.png)

### Clock Reversal Validation

The most physically important special case is the **reversal** permutation π_R(k) = N−1−k, which runs the clock backward. The Covariance Theorem predicts that reversal should:

1. **Pillar 1:** reproduce the reversed Schrödinger dynamics exactly
2. **Pillar 2:** invert the entropy arrow exactly (S_eff^R(k) = S_eff(N−1−k))
3. **Pillar 3:** preserve locality — the reversed observer sees the same self-contained physics in its own frame

All three predictions confirmed with error = 0 to machine precision.

| Script | Output |
|--------|--------|
| `generate_clock_reversal.py` | `output/clock_reversal_pillar1.png` |
| `generate_clock_reversal.py` | `output/clock_reversal_pillar2.png` |
| `generate_clock_reversal.py` | `output/clock_reversal_pillar3.png` |
| `generate_clock_reversal.py` | `output/clock_reversal_combined.png` |
| `generate_clock_reversal.py` | `output/table_clock_reversal.csv` |

![Clock reversal across all three pillars](../output/clock_reversal_combined.png)

### Clock Reversal ≠ Time Reversal (Smoking Gun)

A critical distinction: clock reversal (k ↦ N−1−k) and T-symmetry (ψ → ψ*, t → −t) are **always** distinct operations in the PaW framework, even when the Hamiltonian is T-symmetric.

**Algebraic reason:** Clock reversal produces states at times (N−1−j)dt, while T-reversal produces states at −jdt. Since (N−1−j)dt ≠ −jdt for any finite N, they are algebraically distinct.

| Hamiltonian | Clock reversal ↔ T-reversal distance (Δ) | Arrow inverted by clock reversal? | Arrow inverted by T-reversal? |
|-------------|-------------------------------------------|-----------------------------------|-------------------------------|
| T-symmetric (g_y = 0) | 0.98 | ✅ Yes (exactly) | ✅ Yes (by symmetry) |
| T-breaking (g_y = 0.08) | 0.93 | ✅ Yes (exactly) | ❌ No |

**Interpretation:** The arrow of time is an informational structure arising from the conditioning map, not from the dynamical symmetries of Ĥ. Clock reversal always inverts the arrow (algebraic identity); T-reversal need not (depends on Hamiltonian symmetry). This confirms the framework's central thesis.

---

### Angular Interpolation of Clock Orientation

The Covariance Theorem (above) and Clock Reversal (its special case) treat the clock basis as a **discrete** choice: either the canonical ordering or a permuted one. But the mapping k ↦ N−1−k is topologically disconnected from the identity. A natural question: **is there a continuous path between forward and reversed temporal orderings?**

We construct such a path by defining a **rotated clock basis** parameterized by θ ∈ [0, π]. For each pair (k, N−1−k), we apply a 2D rotation:

$$|k_\theta\rangle = \cos(\theta/2)|k\rangle + \sin(\theta/2)|N\!-\!1\!-\!k\rangle \quad (k < N/2)$$

$$|(N\!-\!1\!-\!k)_\theta\rangle = -\sin(\theta/2)|k\rangle + \cos(\theta/2)|N\!-\!1\!-\!k\rangle$$

This is a proper rotation in each 2D subspace, so {|k_θ⟩} is an orthonormal basis for all θ (verified: max deviation from δ_{jk} < 3×10⁻¹⁷). The boundaries are exact: θ = 0 gives the identity (canonical basis), θ = π gives the reversal.

**Conditioned state at angle θ:** The observer using the rotated clock at tick k sees:

$$\langle k_\theta|_C|\Psi\rangle \propto \cos(\theta/2)\cdot U_{SE}(k\cdot dt)|\psi_0\rangle + \sin(\theta/2)\cdot U_{SE}((N\!-\!1\!-\!k)\cdot dt)|\psi_0\rangle$$

This is not a classical mixture — it is a **quantum superposition** of two time evolutions. At intermediate θ, the observer experiences **temporal interference**: dynamics that are not reducible to any single time ordering.

**Key results:**

| Property | Value |
|----------|-------|
| Arrow strength A(0) | +0.9995 (forward) |
| Arrow strength A(π) | −0.9995 (reversed) |
| Critical angle θ* (A = 0) | ≈ 0.365π |
| Transition | Continuous, smooth |
| Orthonormality | All θ: max error < 3×10⁻¹⁷ |
| Boundary exactness | θ=0: error 0; θ=π: error 3.3×10⁻¹⁶ |

**The critical angle θ* ≈ 0.365π (not π/2):** The asymmetry arises because the initial state |ψ₀⟩ = |↑⟩ breaks the forward/reversed symmetry. The forward arrow is "stronger" — it takes less rotation to destroy it than to build the reversed one. This is physically meaningful: the conditions that select a temporal direction leave a **geometric imprint** on the space of clock orientations.

**Distinction from the fuzzy boundary test:** The gravity robustness test (generate\_gravity\_robustness.py, Test 2) rotates the S/E **partition** in H\_S ⊗ H\_E via a partial SWAP. Here, θ rotates the **clock basis** in H\_C — continuously deforming the temporal description while keeping the system–environment factorization fixed. Different θ, different physics.

**Physical interpretation:** The space of clock orientations is a continuous manifold. The arrow of time is not a binary (forward/backward) but a **real-valued observable** A(θ) that ranges over [−1, +1]. At θ = θ*, the observer has no thermodynamic arrow — an atemporal vantage point distinct from the omniscient observer (who lacks a clock altogether). At intermediate θ, temporal interference creates dynamics absent in any fixed ordering — pointing toward observable signatures in clock quantum superposition experiments.

| Script | Output |
|--------|--------|
| `generate_angular_interpolation.py` | `output/angular_interpolation_heatmap.png` |
| `generate_angular_interpolation.py` | `output/angular_interpolation_arrow.png` |
| `generate_angular_interpolation.py` | `output/angular_interpolation_slices.png` |
| `generate_angular_interpolation.py` | `output/angular_interpolation_combined.png` |
| `generate_angular_interpolation.py` | `output/table_angular_interpolation.csv` |

![Angular interpolation — continuous deformation of clock orientation](../output/angular_interpolation_combined.png)

Six-panel summary. Top left: ⟨σ\_z⟩(k, θ) heatmap showing the smooth transition from forward to reversed oscillation. Top center: S\_eff(k, θ) heatmap showing entropy gradient reversal. Top right: arrow strength A(θ) — continuous from +1 to −1 with critical angle θ\* ≈ 0.365π. Bottom panels: entropy slices, ⟨σ\_z⟩ slices at selected angles, and monotonicity measure.

---

## Experimental Validation on IBM Quantum Hardware

All results above rely on numerical simulation (QuTiP). As a final test, we executed all three pillars on a **real quantum processor** — IBM's `ibm_torino` (133 superconducting qubits) — via the Qiskit Runtime service.

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

### Results — Pillar 3 (Observer-Dependent Time)

Two clocks with different time steps (dt = 0.20 and dt = 0.35) were applied to the same 3-qubit Hamiltonian on ibm\_torino. Each clock defines a different Trotter step size, producing 21 circuits per clock (42 total).

| Metric | Simulator | Hardware (ibm\_torino) |
|--------|-----------|------------------------|
| Max \|⟨σ\_z⟩\_C − ⟨σ\_z⟩\_C'\| | 0.8011 | 0.6879 |
| Max \|S\_C − S\_C'\| | 0.2509 | 0.1386 |
| Clock C: ΔS\_eff | 0.5702 | 0.5874 |
| Clock C': ΔS\_eff | 0.6927 | 0.6888 |

The hardware signal is weaker than the simulator (noise smooths both curves), but the two temporal descriptions clearly differ — confirming that time is observer-dependent even on a real quantum processor.

| Script | Output |
|--------|--------|
| `IBMquantum/run_ibm_validation.py` | `IBMquantum/output/ibm_quantum_validation.png` |
| `IBMquantum/run_ibm_enhanced.py` | `IBMquantum/output/ibm_quantum_enhanced.png` |
| `IBMquantum/run_ibm_enhanced.py` | `IBMquantum/output/table_ibm_enhanced.csv` |
| `IBMquantum/run_ibm_enhanced.py` | `IBMquantum/output/backend_noise_properties.json` |
| `IBMquantum/run_ibm_pillar3.py` | `IBMquantum/output/ibm_pillar3_validation.png` |
| `IBMquantum/run_ibm_pillar3.py` | `IBMquantum/output/table_ibm_pillar3.csv` |

![IBM Quantum hardware validation — enhanced with error bars](../IBMquantum/output/ibm_quantum_enhanced.png)

![IBM Quantum Pillar 3 — two clocks on hardware](../IBMquantum/output/ibm_pillar3_validation.png)

This constitutes the first experimental confirmation on physical quantum hardware that all three pillars of the unified relational time formula survive real-world noise: dynamics emerge from projection, entropy grows from partial trace, and different clocks yield different temporal descriptions.

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

## Scope and Precise Claims

This section states explicitly what the framework claims and does not claim, to prevent misreadings and to anchor each pillar in its precise domain of validity.

### What this work is

A **synthesis contribution**: three known mechanisms — clock conditioning (Page & Wootters, 1983), partial trace as origin of mixed-state entropy (Shaari, 2014; cf. Zurek, Joos, Zeh), and temporal quantum reference frames (Höhn, Smith & Lock, 2021) — are unified into a single operational expression applied to a single timeless state |Ψ⟩. The central result is that dynamics (Pillar 1), irreversibility (Pillar 2), and observer-dependence (Pillar 3) emerge simultaneously from this expression, governed by a single parameter: the observer's **access structure**.

### What this work is not

- It is **not** a derivation of general relativity, Lorentz covariance, or spacetime geometry.
- It is **not** a solution to the full Wheeler–DeWitt equation in quantum gravity.
- It is **not** a claim that the thermodynamic arrow is a fundamental law — it is a **typicality result** under restricted access.

### Pillar 1 — Precise conditions for the "good clock" limit

The recovery of effective Schrödinger dynamics from the clock projection ⟨k|\_C requires:

1. **Approximate orthogonality:** the clock states {|k⟩\_C} must satisfy ⟨j|k⟩\_C ≈ δ\_{jk}. In the toy model, they are exactly orthogonal by construction.
2. **Weak clock–system coupling:** the constraint Ĉ|Ψ⟩ = 0 must decompose such that the clock Hamiltonian H\_C and the system–environment Hamiltonian H\_{SE} are approximately separable: Ĉ ≈ H\_C ⊗ I\_{SE} + I\_C ⊗ H\_{SE}.
3. **Quasi-classical clock evolution:** the clock must tick monotonically — the states |k⟩\_C must label an ordered sequence that does not self-interfere.

When these conditions are relaxed:

- **Finite clock dimension** introduces wrapping effects at k = N (the clock "runs out"). The toy model uses N = 30 ticks; all results are reported within this range.
- **Clock backreaction** (tested in [Gravity Robustness](#gravity-robustness)) degrades the effective Schrödinger dynamics but preserves the thermodynamic arrow. At ε = 1.0 (10× the coupling), dynamics are distorted but not destroyed.
- **Clock uncertainty** (Gaussian smearing with σ up to 4.0) barely affects either dynamics or arrow.

The effective Hamiltonian H\_S that governs the conditioned dynamics is the system component of H\_{SE} in the limit where the clock–system interaction is negligible. In the toy model, H\_S = (ω/2)σ\_x is exact because the clock is non-interacting by construction.

### Pillar 2 — The arrow is typical, not inevitable

The thermodynamic arrow — monotonic growth of S\_eff(k) — is **not** a theorem for arbitrary systems. It is a **typicality result** that holds under the following conditions:

1. **Low effective initial entropy:** the initial state has S\_eff(0) ≈ 0 (the system starts approximately pure from the observer's perspective).
2. **Stable coarse-graining:** the partition into S and E does not change with k.
3. **Generic entanglement growth:** the system–environment interaction H\_{SE} creates entanglement that spreads across exponentially many degrees of freedom.
4. **Large environment:** dim(H\_E) ≫ dim(H\_S), so that the partial trace is strongly contractive.
5. **No fine-tuned spectral symmetries:** the Hamiltonian has a non-degenerate spectrum (or near-non-degenerate), preventing exact Poincaré recurrences on accessible timescales.

Under these conditions, the arrow is robust (tested over 200 random initial states, 5 partition choices, and 3 gravitational perturbations). When conditions fail — e.g., the symmetric coupling model with exact degeneracies — recurrences appear. This is consistent: **the arrow is not a fundamental law but a structural consequence of limited access in generic quantum systems**.

The entropy used throughout is the **von Neumann entropy of the reduced state**: S\_eff(k) = −Tr[ρ\_S(k) ln ρ\_S(k)], which coincides with the entanglement entropy between S and E when the conditioned state ρ\_{SE}(k) is pure.

### Pillar 3 — Observer-dependent parameterization, not general relativity

The claim of Pillar 3 is precisely:

> Different observers choosing different clock subsystems C, C′ extract different temporal parameterizations from the same global state |Ψ⟩. There is no privileged global time.

This is **not** a derivation of Lorentz transformations, general covariance, or the Einstein field equations. The "relativistic character" appears only in the **operational** sense:

- No single clock is preferred.
- The temporal description depends on the observer's choice of reference.
- There is no global simultaneity within the operational partition.

We use the term **"observer-dependent time"** rather than "relativity" to avoid overclaiming. The connection to GR is suggestive — both frameworks deny a privileged global time — but demonstrating that the PaW mechanism in a curved-spacetime constraint reproduces general-relativistic time dilation would require a gravitational Hamiltonian, which is outside the scope of this toy model.

### The access map — formal definition

The concept of "access" that governs all three pillars is formalized as follows.

The observer defines an **accessible observable algebra** $\mathcal{O}_S \subset \mathcal{B}(\mathcal{H})$ consisting of operators of the form $A_S \otimes I_E$ — observables that act nontrivially only on the system S. The **access map** is:

$$\mathcal{A}: \mathcal{B}(\mathcal{H}_{SE}) \to \mathcal{B}(\mathcal{H}_S), \quad \rho_{SE} \mapsto \mathrm{Tr}_E[\rho_{SE}]$$

This is a CPTP (completely positive, trace-preserving) map. Its physical content: the observer can measure any $A_S \in \mathcal{O}_S$ and obtain $\langle A_S \rangle = \mathrm{Tr}[\rho_S \cdot A_S]$, but cannot measure joint S–E correlators.

The access is:

- **Operational**, not ontological: it describes what the observer can measure, not what exists.
- **Not a coarse-graining in the Gibbs/Boltzmann sense**: it is a restriction of the observable algebra, not a partition of phase space into macrostates.
- **Not a statement about consciousness**: it is a physical limitation (finite detector, causal horizon, decoherent sector boundary).

### The synthesis — what is new

The individual ingredients are not new:

| Ingredient | Source | What it explains alone |
|------------|--------|----------------------|
| Clock conditioning ⟨k\|\_C | Page & Wootters (1983) | Dynamics from a timeless state |
| Partial trace Tr\_E | Zurek, Joos, Zeh (1980s–90s); Shaari (2014) | Decoherence and entropy growth |
| Clock locality / QRFs | Höhn, Smith & Lock (2021) | Observer-dependent temporal reference |

What is new is the **unification**: recognizing that all three emerge from a single expression

$$\rho_S(k) = \frac{\mathrm{Tr}_E[\langle k|_C\,|\Psi\rangle\langle\Psi|\,|k\rangle_C]}{p(k)}$$

and that the governing parameter across all three pillars is the **access structure** of the observer. Specifically:

- The access map determines **which** degrees of freedom are traced (→ Pillar 2).
- The access map determines **which** degree of freedom serves as clock (→ Pillar 1).
- The locality of the clock choice determines **whose** temporal description is obtained (→ Pillar 3).

No prior work unifies these three aspects into a single operational expression, demonstrates their simultaneous emergence from one |Ψ⟩, or identifies access as the common pivot.

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

*Main claims: [Claims](CLAIMS.md) — the six explicit testable claims of this work.*

*Step-by-step derivation: [Derivation](DERIVATION.md) — the formula developed from first principles, operation by operation.*

*Geometric companion: [Geometric Structure](GEOMETRY.md) — Bloch trajectory, relational bundle, and the arrow as geometry.*

*Boundary analysis: [The Omniscient Observer](GOD_OBSERVER.md) — what happens when the observer has complete access.*

*Common objections: [FAQ](FAQ.md) — answers to questions from physicists: Occam's Razor, causality vs clock, and more.*
