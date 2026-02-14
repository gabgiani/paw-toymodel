# Main Claims

This work advances six explicit, testable claims — each supported by numerical validation and, where applicable, by algebraic proof.

**Claim 1 — Unified Operational Construction.**
The three pillars of the problem of time (dynamics, irreversibility, frame dependence) emerge simultaneously from a single operational pipeline — conditioning on a local clock and tracing out inaccessible degrees of freedom:

$$\rho_S(k) = \frac{\mathrm{Tr}_E[\langle k|_C\,|\Psi\rangle\langle\Psi|\,|k\rangle_C]}{p(k)}$$

Each operation is individually standard (Page–Wootters conditioning, partial trace); the claim is that their composition, applied to a single globally stationary state |Ψ⟩, yields all three pillars without separate mechanisms.
→ *Validated in:* [Three Pillars from One Formula](THEORY.md#three-pillars-from-one-formula) · [Step-by-Step Derivation](DERIVATION.md)

**Claim 2 — The arrow is informational.**
For generic product initial states and in the coarse-grained regime, the thermodynamic arrow of time emerges as an effective consequence of the conditioning-plus-partial-trace structure — from the observer's limited access to the global state — rather than from dynamical asymmetries of the Hamiltonian Ĥ. The arrow is present even when the underlying dynamics are time-reversal symmetric and vanishes when the observer has unlimited access (god-observer limit).
→ *Validated in:* [Pillar 2: Thermodynamic Arrow](THEORY.md#pillar-2-thermodynamic-arrow--from-tr_e-partial-trace) · [The Omniscient Observer](GOD_OBSERVER.md)

**Claim 3 — Clock Relabeling Covariance (Theorem).**
The conditional state transforms covariantly under arbitrary relabelling of the clock basis: if π is any permutation of the clock labels {0, …, N−1}, then the state ρ\_S evaluated at tick π(k) in the relabelled basis equals ρ\_S evaluated at tick k in the original basis. The physics is the same — only the labels change. This was proved algebraically and verified numerically for all 720 permutations of a 6-tick clock (error = 0 for every permutation).
→ *Validated in:* [Clock Orientation Covariance](THEORY.md#clock-orientation-covariance) · [Clock Reversal Validation](THEORY.md#clock-reversal-validation)

**Claim 4 — Continuity of the arrow.**
The arrow of time is not a binary (forward/backward) observable. When the clock basis is continuously rotated by an angle θ ∈ [0, π], the arrow strength — defined as A(θ) ≡ (S\_final − S\_initial)/(S\_final + S\_initial), where S are the von Neumann entropies of the conditioned reduced state — varies continuously from +1 (fully forward) to −1 (fully reversed), with a critical angle θ\* ≈ 0.365π at which A = 0. At intermediate angles the conditioned description exhibits an interference-like crossover between forward and reversed entropy profiles, a feature specific to the relational conditioning construction.
→ *Validated in:* [Angular Interpolation of Clock Orientation](THEORY.md#angular-interpolation-of-clock-orientation)

**Claim 5 — Necessity of every condition.**
Each condition in postulates P3 (good-clock regime) and P4 (informational arrow) is necessary within our operational definitions and chosen diagnostics: violating any single condition degrades or destroys the corresponding pillar. This was demonstrated via five contrapositiva tests.
→ *Validated in:* [Condition Necessity Tests (Contrapositiva)](THEORY.md#condition-necessity-tests-contrapositiva)

**Claim 6 — Hardware validation.**
The informational arrow survives real quantum hardware noise. Tested on IBM Quantum (ibm\_torino, 133 superconducting qubits), Pillar 1 (dynamics, max deviation 0.033) and Pillar 2 (arrow, S\_eff = 0.583 ± 0.005, 102.2% of exact) are confirmed with full noise characterisation.
→ *Validated in:* [Experimental Validation on IBM Quantum Hardware](THEORY.md#experimental-validation-on-ibm-quantum-hardware)

## Summary

| Claim | Type | Key evidence | Validation |
|-------|------|--------------|------------|
| 1. Unified Operational Construction | Structural | Three pillars from one ρ\_S(k); each operation standard, contribution is the unified pipeline | [Theory](THEORY.md#three-pillars-from-one-formula) · [Derivation](DERIVATION.md) |
| 2. Informational arrow | Conceptual + numerical | Arrow present with T-symmetric Ĥ; vanishes under unlimited access (god-observer) | [Theory](THEORY.md#pillar-2-thermodynamic-arrow--from-tr_e-partial-trace) · [God Observer](GOD_OBSERVER.md) |
| 3. Relabeling Covariance Theorem | Algebraic + numerical | 720/720 permutations exact (error = 0) | [Theory](THEORY.md#clock-orientation-covariance) |
| 4. Continuity | Numerical | A(θ) ≡ (S\_f − S\_i)/(S\_f + S\_i): +1 → 0 → −1; θ\* ≈ 0.365π | [Theory](THEORY.md#angular-interpolation-of-clock-orientation) |
| 5. Necessity | Contrapositiva | 5/5 conditions necessary within operational definitions | [Theory](THEORY.md#condition-necessity-tests-contrapositiva) |
| 6. Hardware | Experimental | IBM ibm\_torino: S\_eff = 0.583 ± 0.005 | [Theory](THEORY.md#experimental-validation-on-ibm-quantum-hardware) |
