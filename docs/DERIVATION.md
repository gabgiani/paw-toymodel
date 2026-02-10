# Step-by-Step Derivation of the Unified Relational Formula

## Companion Note

This document develops the unified relational formula from first principles, explaining each step, each symbol, and each operation. No prior familiarity with the Page–Wootters mechanism is assumed. The derivation follows the logical order of construction: from the problem, through the ingredients, to the formula and its consequences.

All numerical values are reproducible using the toy model in this repository.

---

## 1. The Problem: Where Does Time Come From?

In ordinary quantum mechanics, the Schrödinger equation

$$i\hbar\,\partial_t\,|\psi(t)\rangle = H\,|\psi(t)\rangle$$

contains an external time parameter t. This parameter is not an observable — it has no operator, no eigenvalues, no uncertainty relation. It is put in by hand as a classical background.

In a theory of the whole universe (quantum gravity, quantum cosmology), there is no "outside" to provide this parameter. The total Hamiltonian of a closed system satisfies a constraint:

$$\hat{C}\,|\Psi\rangle = 0$$

This is the **Wheeler–DeWitt equation** (or its finite-dimensional analog). It says: *the total energy of the universe is zero* (a consequence of general covariance). The state |Ψ⟩ does not evolve — it is **frozen**.

**The problem of time:** If nothing evolves, where does the time we experience come from? Where does the Schrödinger equation come from? Where does irreversibility come from?

The unified relational formula answers all three questions with a single expression.

---

## 2. The Setup: Three Subsystems

We begin with a Hilbert space that admits a tensor factorization into three parts:

$$\mathcal{H} = \mathcal{H}_C \otimes \mathcal{H}_S \otimes \mathcal{H}_E$$

| Subsystem | Symbol | Role | Toy model |
|-----------|--------|------|-----------|
| **Clock** | C | The degree of freedom the observer uses as a temporal reference | N = 30 orthogonal states \|k⟩, k = 0, …, 29 |
| **System** | S | What the observer is studying | Single qubit (dim = 2) |
| **Environment** | E | Degrees of freedom the observer cannot access | n\_env qubits (dim = 2^n\_env) |

This decomposition is an **operational choice** — not a law of nature. Different observers may choose different C, S, E partitions. That freedom is the content of Pillar 3.

### The global state

Inside this Hilbert space, there exists a pure state |Ψ⟩ satisfying the constraint:

$$\hat{C}\,|\Psi\rangle = 0$$

This state encodes all correlations between C, S, and E. It does not evolve. It has no time parameter. It is the timeless, eternal "block" of the universe.

### Toy model realization

In the toy model, the history state is constructed explicitly as:

$$|\Psi\rangle = \frac{1}{\sqrt{N}} \sum_{k} |k\rangle_C \otimes U_{SE}(t_k)\,|\psi_0\rangle_{SE}$$

where:
- |k⟩\_C is the k-th clock basis state
- t\_k = k · dt is the physical time associated with clock tick k
- U\_SE(t\_k) = exp(−i H\_tot t\_k) is the unitary evolution of S⊗E
- |ψ₀⟩\_SE = |0⟩\_S ⊗ |0⟩^⊗n\_env is the initial state of S and E

The factor 1/√N ensures equal probability of every clock reading.

**In code** (`validate_formula.py`, function `build_paw_history`):

```python
for k in range(N):
    t_k = k * dt
    U_SE = (-1j * H_tot * t_k).expm()
    comp = norm * qt.tensor(clock_basis[k], U_SE * initial_SE)
    psi += comp
```

---

## 3. The Hamiltonians

### System Hamiltonian

$$H_S = \frac{\omega}{2}\,\sigma_x$$

This generates rotations around the x-axis of the Bloch sphere. With |ψ₀⟩ = |0⟩ (eigenstate of σ\_z), the expectation value oscillates:

$$\langle\sigma_z\rangle(t) = \cos(\omega t)$$

| Parameter | Value |
|-----------|-------|
| ω (frequency) | 1.0 |
| dt (tick spacing) | 0.2 |
| Period | 2π/ω ≈ 6.28 |
| Ticks per period | 2π/(ω·dt) ≈ 31.4 |

### System–Environment Interaction

$$H_{SE} = g \sum_j \sigma_x^{(S)} \otimes \sigma_x^{(E_j)}$$

Each environment qubit j is coupled to the system via a σ\_x ⊗ σ\_x interaction with strength g = 0.1. This coupling:
- Creates entanglement between S and E
- Transfers coherence from S to S⊗E correlations
- Is what makes the partial trace Tr\_E non-trivial

### Total Hamiltonian

$$H_{\text{tot}} = H_S + H_{SE}$$

This acts on H\_S ⊗ H\_E only — **not** on the clock space H\_C. The clock provides the parametric label; the physics happens in S⊗E.

---

## 4. Step 1: Clock Projection — ⟨k|\_C

### What it does

Given the global state |Ψ⟩ living in H\_C ⊗ H\_S ⊗ H\_E, the projection ⟨k|\_C extracts the component correlated with clock reading k:

$$|\phi_k\rangle_{SE} = \big(\langle k|_C \otimes I_{SE}\big)\,|\Psi\rangle$$

This is a **partial inner product**: we fix the clock to reading k and ask "what is the state of everything else?"

### What it means

The result |φ\_k⟩\_SE is a (unnormalized) state in H\_S ⊗ H\_E. It represents: "the state of the system and environment **given that the clock shows k**."

The sequence k = 0, 1, 2, …, N−1 produces a **family** of conditional states:

$$k = 0 \to |\phi_0\rangle_{SE},\quad k = 1 \to |\phi_1\rangle_{SE},\quad \ldots,\quad k = 29 \to |\phi_{29}\rangle_{SE}$$

This family is the raw material for temporal evolution. **There is no external time parameter** — the ordering comes from the correlations between C and SE inside |Ψ⟩.

### The probability

The probability of finding clock reading k is:

$$p(k) = \langle\phi_k|\phi_k\rangle = \langle\Psi|\,\big(|k\rangle\langle k|_C \otimes I_{SE}\big)\,|\Psi\rangle$$

In the toy model with the uniform history state, p(k) = 1/N for all k (equidistributed clock).

### In the code

```python
# Decompose |Ψ⟩ into blocks: one per clock reading
psi_vec = psi.full().flatten()
blocks = psi_vec.reshape(N, d_SE)

# ⟨k|_C ⊗ I_SE |Ψ⟩ = k-th block
phi_k = blocks[k, :]
p_k = np.vdot(phi_k, phi_k).real   # probability
```

The vector `phi_k` is the state |φ\_k⟩\_SE expressed as a flat array of dimension d\_SE = dim(H\_S) × dim(H\_E).

### What this operation produces: Pillar 1

In the case n\_env = 0 (no environment), this projection alone is sufficient to recover Schrödinger dynamics:

$$|\phi_k\rangle_S = \langle k|_C\,|\Psi\rangle = \frac{1}{\sqrt{N}}\,U_S(t_k)\,|\psi_0\rangle$$

The conditional expectation value is:

$$\langle\sigma_z\rangle(k) = \frac{\langle\phi_k|\sigma_z|\phi_k\rangle}{p(k)} = \cos(\omega\,k\,dt)$$

**Numerical result:** max deviation from cos(ωkdt) = 4.44 × 10⁻¹⁶ (machine precision).

> **Pillar 1:** The projection ⟨k|\_C, applied to a timeless state |Ψ⟩, produces a parametric family of states that obeys the Schrödinger equation. Dynamics emerge from conditioning.

---

## 5. Step 2: Normalization — the Conditional State

### From unnormalized to density operator

The projected state |φ\_k⟩\_SE is not normalized (its norm squared is p(k), not 1). To obtain the proper conditional state, we form:

$$\rho_{SE}(k) = \frac{|\phi_k\rangle\langle\phi_k|_{SE}}{p(k)}$$

This is the **conditional density matrix** of the system and environment, given clock reading k. It is pure (rank 1) because |Ψ⟩ is pure and the projection ⟨k|\_C is a projective measurement on C.

### What it means

ρ\_SE(k) encodes the complete description of S and E at "time" k — but only from the perspective of an observer who has access to both S and E. This is the state of a **god with a clock** (Level 1 omniscience — see [GOD\_OBSERVER.md](GOD_OBSERVER.md)).

For this observer, ρ\_SE(k) is pure → S\_eff = 0 → no entropy → no arrow.

### In the code

```python
psi_SE_k = qt.Qobj(phi_k.reshape(-1, 1), dims=dims_ket)
rho_SE_k = (psi_SE_k * psi_SE_k.dag()) / p_k
```

---

## 6. Step 3: Partial Trace — Tr\_E

### What it does

The observer cannot access the environment E. The operation that discards E is the **partial trace**:

$$\rho_S(k) = \mathrm{Tr}_E\!\big[\rho_{SE}(k)\big]$$

This maps the pure state ρ\_SE(k) (which lives in the 2 × 2^n\_env dimensional space) down to a 2×2 density matrix ρ\_S(k) on the system alone.

### What it means physically

The partial trace is not a mathematical convenience — it is the **operational definition of limited access**. When the observer cannot distinguish environment microstates, the coherences between S and E become invisible. What remains is a mixture:

$$\rho_S(k) = \sum_e \langle e|_E\;\rho_{SE}(k)\;|e\rangle_E$$

where {|e⟩\_E} is any orthonormal basis of H\_E.

### What this operation produces: Pillar 2

The partial trace converts a pure state into a mixed one. This has three consequences:

**1. Purity loss.** The Bloch radius |r⃗| decreases:

$$\mathrm{Tr}\!\big[\rho_S(k)^2\big] \le 1 \quad \text{(equality only if }\rho_{SE}(k)\text{ is a product state)}$$

**2. Entropy growth.** The von Neumann entropy increases:

$$S_{\text{eff}}(k) = -\mathrm{Tr}\!\big[\rho_S(k)\ln\rho_S(k)\big] \ge 0 \quad \text{(0 only if no S-E entanglement)}$$

**3. Irreversibility.** The partial trace is a **CPTP (completely positive, trace-preserving) map** that is **contractive** for the relative entropy. By the data processing inequality, information about S that leaked into S⊗E correlations cannot be recovered from S alone.

**Numerical results (n\_env = 4):**

| k | ⟨σ\_z⟩ | S\_eff | \|r⃗\| |
|---|---------|--------|--------|
| 0 | 1.000 | 0.000 | 1.000 |
| 5 | 0.498 | 0.164 | 0.930 |
| 10 | −0.300 | 0.405 | 0.666 |
| 15 | −0.342 | 0.570 | 0.432 |
| 20 | −0.063 | 0.674 | 0.111 |
| 25 | 0.024 | 0.690 | 0.085 |
| 29 | 0.023 | 0.693 | 0.025 |

The oscillations damp. The entropy grows. The Bloch vector spirals inward. This is the **thermodynamic arrow of time**, emerging entirely from the partial trace.

> **Pillar 2:** The partial trace Tr\_E, applied to the conditioned pure state ρ\_SE(k), produces a mixed state ρ\_S(k) with growing entropy. Irreversibility emerges from limited access.

### In the code

```python
if n_env > 0:
    rho_S = rho_SE_k.ptrace(0)   # keep S (index 0), trace out E
else:
    rho_S = rho_SE_k              # no E to trace; ρ_S = ρ_SE
```

---

## 7. The Complete Formula — Assembled

Combining Steps 1, 2, and 3:

$$\rho_S(k) = \frac{\mathrm{Tr}_E\!\big[\langle k|_C\;|\Psi\rangle\langle\Psi|\;|k\rangle_C\big]}{p(k)}$$

In operator notation:

$$\rho_S(k) = \frac{\mathrm{Tr}_E\!\Big[\big(\langle k|_C \otimes I_{SE}\big)\,|\Psi\rangle\langle\Psi|\,\big(|k\rangle_C \otimes I_{SE}\big)\Big]}{p(k)}$$

### Reading the formula left to right

| Piece | Operation | Physical meaning |
|-------|-----------|-----------------|
| \|Ψ⟩⟨Ψ\| | Global density matrix | The timeless, atemporal object. S\_eff = 0. |
| ⟨k\|\_C … \|k⟩\_C | Clock projection | Fixes the temporal reference. Selects "when." |
| Tr\_E [ … ] | Partial trace | Discards inaccessible degrees. Creates mixture. |
| / p(k) | Normalization | Ensures Tr[ρ\_S(k)] = 1 (proper conditional state). |

### Reading the formula as three pillars

| Component | Pillar | What emerges |
|-----------|--------|-------------|
| ⟨k\|\_C | 1 | Quantum dynamics — the parametric family ρ\_S(k) obeys Schrödinger |
| Tr\_E | 2 | Thermodynamic arrow — entropy grows along k, purity decays |
| C is chosen by the observer | 3 | Observer-dependent time — different C → different temporal narrative |

---

## 8. Pillar 3: Clock Locality — the Observer's Freedom

### The structural content

Nowhere in the formula does the symbol t appear as a fundamental parameter. The "time" is the clock reading k, which depends on:

1. **Which subsystem** the observer designates as C.
2. **Which basis** {|k⟩} the observer measures in C.
3. **What spacing** dt the observer associates with consecutive readings.

A different observer choosing a different clock C′ (say, with a different dt′) will apply the same formula to the same |Ψ⟩ and obtain a **different** temporal description.

### Numerical demonstration

Two clocks with dt = 0.20 vs dt = 0.35 applied to the same |Ψ⟩ with n\_env = 4:

| k | ⟨σ\_z⟩ (C₁, dt=0.20) | S\_eff (C₁) | ⟨σ\_z⟩ (C₂, dt=0.35) | S\_eff (C₂) |
|---|------|------|------|------|
| 0 | 1.000 | 0.000 | 1.000 | 0.000 |
| 5 | 0.498 | 0.164 | −0.139 | 0.348 |
| 10 | −0.300 | 0.405 | −0.320 | 0.633 |
| 15 | −0.342 | 0.570 | 0.145 | 0.688 |
| 20 | −0.063 | 0.674 | 0.001 | 0.693 |
| 29 | 0.023 | 0.693 | −0.029 | 0.692 |

At k = 5: observer C₁ reports ⟨σ\_z⟩ ≈ 0.50 while C₂ reports ⟨σ\_z⟩ ≈ −0.14. **Neither is wrong.** Each is a valid conditional description relative to their own clock.

Both converge to the same asymptotic entropy (ln 2 ≈ 0.693), because the underlying |Ψ⟩ and the S-E coupling are the same — only the temporal parametrization differs.

> **Pillar 3:** Time is not a property of the universe. Time is a property of the question the observer asks. Different clocks → different answers → all equally valid.

---

## 9. What the Formula Does Not Contain

Understanding what is **absent** from the formula is as important as understanding what is present:

| Absent element | Why it matters |
|----------------|---------------|
| **External time t** | No background temporal parameter appears. Time is emergent, not input. |
| **Collapse postulate** | No wavefunction collapse occurs. The projection ⟨k\|\_C is a conditioning (Bayesian update), not a physical collapse. |
| **Initial conditions** | The state \|Ψ⟩ does not have a "special" initial moment. The arrow emerges structurally from Tr\_E, not from a low-entropy boundary condition. |
| **Non-unitary dynamics** | The global evolution is unitary (Ĉ\|Ψ⟩ = 0 is a constraint, not a dissipative equation). Apparent non-unitarity comes from Tr\_E. |
| **Preferred observer** | No subsystem is designated as the "right" clock. Any C works, yielding different descriptions. |

---

## 10. The Three Regimes — One Formula, Three Behaviors

The same formula produces qualitatively different physics depending on the access structure:

### Regime A: Full access, no environment (n\_env = 0)

$$\rho_S(k) = \frac{\langle k|_C\;|\Psi\rangle\langle\Psi|\;|k\rangle_C}{p(k)}$$

Tr\_E is absent (nothing to trace). Result:
- ρ\_S(k) is **pure** for all k
- ⟨σ\_z⟩(k) = cos(ωkdt) — **exact Schrödinger dynamics**
- S\_eff(k) = 0 — **no arrow**
- Bloch radius |r⃗| = 1 — trajectory on sphere surface

### Regime B: Partial access (n\_env > 0)

$$\rho_S(k) = \frac{\mathrm{Tr}_E\!\big[\langle k|_C\;|\Psi\rangle\langle\Psi|\;|k\rangle_C\big]}{p(k)}$$

Tr\_E is non-trivial. Result:
- ρ\_S(k) becomes **mixed** as k increases
- Oscillations **damp** (decoherence)
- S\_eff(k) **grows** toward ln 2 — **arrow emerges**
- Bloch radius |r⃗| → 0.025 — spiral toward center

### Regime C: Omniscient observer (no projection)

$$\rho_{\text{global}} = |\Psi\rangle\langle\Psi|$$

No ⟨k|\_C projection, no Tr\_E. Result:
- Single state, no temporal parametrization
- S\_eff = 0 — **no time, no arrow, no dynamics**
- The universe is "frozen"

**Summary:**

| What the observer does | What emerges |
|-----------------------|-------------|
| Projects onto ⟨k\|\_C | Dynamics |
| Traces out E | Arrow |
| Both | Full temporal experience |
| Neither | Frozen atemporality |

---

## 11. The Arrow Is Not Assumed — It Is Derived

A common objection: "Doesn't the partial trace just *assume* irreversibility by discarding information?"

The answer is subtle:

1. **Tr\_E is not a physical process.** It is a mathematical expression of the observer's limited access. No information is destroyed in the universe — it is merely inaccessible to this particular observer.

2. **The arrow is not postulated.** We do not add a special initial condition or a non-unitary term. The global state |Ψ⟩ satisfies a time-symmetric constraint Ĉ|Ψ⟩ = 0. The arrow appears because Tr\_E is a **contractive map**: once S-E correlations form, the restricted observer's description becomes irreversibly mixed.

3. **Remove the limitation, remove the arrow.** When n\_env = 0, Tr\_E is trivial, and S\_eff = 0 at all k. The arrow is not intrinsic to the universe — it is intrinsic to the **act of not seeing everything**.

This is the core insight: **the arrow of time is the cost of being a finite observer inside an atemporal whole**.

---

## 12. Step-by-Step Numerical Walkthrough

### The simplest case: k = 0

At k = 0, the unitary is U\_SE(0) = I (identity). The conditional state is:

$$|\phi_0\rangle_{SE} = \frac{1}{\sqrt{N}}\,|\psi_0\rangle_{SE} = \frac{1}{\sqrt{N}}\,|0\rangle_S \otimes |0\rangle^{\otimes n_{\text{env}}}$$

This is a product state — no S-E entanglement. Therefore:

$$\rho_S(0) = \mathrm{Tr}_E\!\big[|\phi_0\rangle\langle\phi_0|_{SE}\big] / p(0) = |0\rangle\langle 0| \quad (\text{pure}), \quad S_{\text{eff}}(0) = 0, \quad \langle\sigma_z\rangle(0) = 1, \quad |\vec{r}\,|(0) = 1$$

No mixture, no arrow, no entropy. The observer starts with full information about S.

### An intermediate step: k = 10

At k = 10 (t = 2.0), U\_SE has created entanglement between S and E:

$$U_{SE}(2.0)\,|\psi_0\rangle_{SE} = \alpha|{\uparrow}\rangle_S|e_1\rangle + \beta|{\downarrow}\rangle_S|e_2\rangle + \cdots \quad (\text{entangled})$$

The partial trace discards the environment labels:

$$\rho_S(10) = |\alpha|^2\,|{\uparrow}\rangle\langle{\uparrow}| + |\beta|^2\,|{\downarrow}\rangle\langle{\downarrow}| + (\text{off-diag terms decay})$$

Now the off-diagonal terms are suppressed because the environment states |e₁⟩, |e₂⟩, … become nearly orthogonal. Result:

$$S_{\text{eff}}(10) \approx 0.405, \quad \langle\sigma_z\rangle(10) \approx -0.300, \quad |\vec{r}\,|(10) \approx 0.666$$

### The asymptotic limit: k → 29

At late times, S-E entanglement is essentially maximal for a single qubit coupled to 4 environment qubits. The reduced state approaches:

$$\rho_S(29) \approx I/2 + \varepsilon \quad (\text{nearly maximally mixed})$$

$$S_{\text{eff}}(29) \approx 0.693 \approx \ln 2, \quad \langle\sigma_z\rangle(29) \approx 0.023, \quad |\vec{r}\,|(29) \approx 0.025$$

The observer's description has reached maximum uncertainty. The arrow has fully developed.

---

## 13. The Formula in One Sentence

> A timeless pure state |Ψ⟩, conditioned on a clock reading ⟨k|\_C and restricted by partial access Tr\_E, produces a mixed state ρ\_S(k) that exhibits Schrödinger dynamics, growing entropy, and observer-dependent parametrization — the three pillars of temporal experience — as a single, unified operation.

---

## 14. Attribution

| Ingredient | Origin | Year |
|-----------|--------|------|
| Clock projection ⟨t\|\_C | Page & Wootters | 1983 |
| Partial trace Tr\_E as origin of arrow | Shaari; this work | 2014; 2026 |
| Clock locality / temporal QRFs | Höhn, Smith & Lock | 2021 |
| **Unification into one formula** | **This work** | **2026** |

---

## Reproducibility

All numerical results are generated by:

```bash
python validate_formula.py              # Pillars 1 & 2, step-by-step
python run_all.py                       # Full pipeline + all metrics
python generate_pillar3_plot.py         # Pillar 3 (two-clock comparison)
python generate_geometry_plots.py       # Geometric interpretation (Bloch)
```

Repository: [github.com/gabgiani/paw-toymodel](https://github.com/gabgiani/paw-toymodel)

---

*Back to: [Theory](THEORY.md) | [Geometric Structure](GEOMETRY.md) | [Scripts & Outputs](SCRIPTS.md) | [The Omniscient Observer](GOD_OBSERVER.md)*
