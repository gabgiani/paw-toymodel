# Geometric Structure of the Unified Relational Formula

## Companion Note

This companion note identifies the geometric objects underlying the unified relational formula and validates them numerically using the toy model. The notation and parameters are consistent with the main paper.

---

## 1. What Is the Fundamental Geometric Object?

The fundamental object is a **stationary pure state** |Ψ⟩ satisfying the Wheeler–DeWitt-type constraint:

$$\hat{C}\,|\Psi\rangle = 0$$

This state lives in the **projective Hilbert space** CP(H), where kets are defined up to global phase. The constraint Ĉ = 0 defines a codimension-1 hypersurface (a submanifold) within this projective space.

At this level:
- There is no Lorentzian metric.
- There is no temporal foliation.
- There is no notion of "before" or "after."
- The effective entropy is exactly zero: S\_eff = 0.

The state |Ψ⟩ is a **fixed point** — a timeless, atemporal object from which all temporal structure will be extracted operationally.

**Numerical confirmation:** In the toy model, when we eliminate the partition and the projection (n\_env = 0 and no conditioning on k), only ρ\_global = |Ψ⟩⟨Ψ| remains — a pure state with S\_eff = 0 and no temporal curve. The base object is atemporal and stationary.

---

## 2. From State Space to Relational Bundle

### The operational partition

The Hilbert space admits a tensor factorization:

$$\mathcal{H} = \mathcal{H}_C \otimes \mathcal{H}_S \otimes \mathcal{H}_E$$

This is an **operational choice**, not a fundamental decomposition: the observer selects which degrees of freedom serve as clock (C), which as system (S), and which remain inaccessible (E).

### The bundle structure

Upon this factorization, the conditional map

$$k \;\mapsto\; \rho_S(k) = \frac{\mathrm{Tr}_E\!\big[\langle k|_C\;|\Psi\rangle\langle\Psi|\;|k\rangle_C\big]}{p(k)}$$

defines a **section** of a trivial quantum bundle over the discrete base space of clock readings {|k⟩\_C}:

| Element | Geometric role |
|---------|----------------|
| Base space | Spectrum of distinguishable clock readings (k = 0, 1, …, N−1) |
| Fiber at k | The space of density operators D(H\_S) |
| Section | The map k ↦ ρ\_S(k) |
| Structure group | CPTP maps connecting fibers |

The projection ⟨k|\_C **selects a fiber** in this bundle. The partial trace Tr\_E determines the **location within the fiber** (how mixed the state is).

**Numerical confirmation:** In Version A (n\_env = 0), the family k ↦ ρ\_S(k) parametrizes exactly a transversal section of the trivial bundle over the discrete clock space (k = 0..29). The coherent ⟨σ\_z⟩(k) = cos(ωk·dt) curve is the projection of this section onto the σ\_z observable.

---

## 3. Where Does the Observer's Projection Live?

The observer's projection lives **in the clock subsystem C**, specifically in the family of operators {|k⟩⟨k|\_C} (or the POVM defining distinguishable readings).

- It is an **operational action**: the observer "chooses" C and projects to "read" time.
- In the bundle: the projection ⟨k|\_C **selects the fiber** corresponding to reading k.
- Ontologically: there is no privileged location. Any subsystem with distinguishable states can serve as C (complete relationality — Pillar 3).

Without this projection, there is no temporal parameter, no curve, no dynamics:

$$\text{No }\langle k|_C\text{ projection} \;\Rightarrow\; \text{no }t \;\Rightarrow\; \text{no temporal narrative}$$

**Numerical confirmation:** In `get_conditioned_observables`, the projection is explicit: ⟨k|\_C ⊗ I\_SE |Ψ⟩ extracts block k of the state vector. Without this step, there is no parametric family ρ\_S(k) — only the global |Ψ⟩.

---

## 4. The Arrow of Time as Geometry

### The Bloch ball representation

For a single qubit system S, the space of density operators D(H\_S) is the **Bloch ball** — a solid sphere of radius 1 in ℝ³, parameterized by the Bloch vector:

$$\rho_S = \frac{I + \vec{r}\cdot\vec{\sigma}}{2}, \quad |\vec{r}\,| \le 1$$

- |r⃗| = 1 : pure state (surface of the sphere)
- |r⃗| = 0 : maximally mixed state I/2 (center)
- 0 < |r⃗| < 1 : partial mixture (interior)

### Two trajectories, one formula

The unified formula generates two qualitatively different trajectories in the Bloch ball depending on the access structure:

**Version A** (n\_env = 0, no Tr\_E):

The Bloch vector traces a **great circle** on the sphere surface. The radius |r⃗| = 1 at all times. The state remains pure. The dynamics are reversible. There is no arrow.

**Version B** (n\_env = 4, Tr\_E active):

The Bloch vector **spirals inward** from the surface toward the center I/2. The radius |r⃗| decays from 1.000 to 0.025. The entropy S\_eff grows from 0 to ln 2 ≈ 0.693. The dynamics are irreversible. The arrow emerges.

### The arrow is the spiral

The thermodynamic arrow of time is geometrically encoded as the **monotonic displacement** of the Bloch vector toward the center of the Bloch ball — equivalently, the monotonic growth of S\_eff along the curve ρ\_S(k) in the convex manifold D(H\_S).

The mechanism is the partial trace Tr\_E, which is a **contractive CPTP map**: it systematically pushes states toward the maximally mixed interior. The contraction is monotone for the von Neumann entropy (data processing inequality), guaranteeing:

$$S_{\text{eff}}(k+1) \ge S_{\text{eff}}(k) \quad \text{(in expectation)}$$

This is not a coincidence. It is a **geometric necessity** of partial trace over inaccessible degrees of freedom.

---

## 5. Numerical Validation

### Bloch vector trajectories

The following table shows the Bloch vector components, radius, and entropy at selected clock readings (full data: `output/table_bloch_trajectory.csv`):

| k | ⟨σ\_y⟩\_A | ⟨σ\_z⟩\_A | \|r⃗\|\_A | ⟨σ\_y⟩\_B | ⟨σ\_z⟩\_B | \|r⃗\|\_B | S\_eff\_B |
|---|---------|---------|---------|---------|---------|---------|---------|
| 0 | 0.000 | 1.000 | **1.000** | 0.000 | 1.000 | **1.000** | 0.000 |
| 3 | −0.565 | 0.825 | **1.000** | −0.549 | 0.802 | **0.972** | 0.075 |
| 6 | −0.932 | 0.362 | **1.000** | −0.844 | 0.327 | **0.905** | 0.198 |
| 10 | −0.912 | −0.411 | **1.000** | −0.607 | −0.274 | **0.666** | 0.430 |
| 15 | −0.288 | −0.958 | **1.000** | −0.114 | −0.380 | **0.397** | 0.601 |
| 20 | 0.756 | −0.655 | **1.000** | 0.084 | −0.073 | **0.111** | 0.674 |
| 25 | 0.959 | 0.284 | **1.000** | 0.082 | 0.024 | **0.085** | 0.690 |
| 29 | 0.465 | 0.886 | **1.000** | 0.012 | 0.023 | **0.025** | 0.693 |

Key observations:
- **Version A**: |r⃗| = 1.000 at all k — the state never leaves the Bloch sphere surface.
- **Version B**: |r⃗| decays from 1.000 → 0.025, approaching the maximally mixed center.
- **Duality**: |r⃗| decay and S\_eff growth are dual descriptions of the same geometric process.

### Summary statistics

| Quantity | Version A | Version B |
|----------|-----------|-----------|
| Initial Bloch radius | 1.000 | 1.000 |
| Final Bloch radius | 1.000 | 0.025 |
| Initial S\_eff | 0.000 | 0.000 |
| Final S\_eff | 0.000 | 0.693 |
| Trajectory type | Great circle | Inward spiral |
| Reversible | Yes | No |
| Arrow of time | Absent | Present |

---

## 6. Figures

### Figure 1: Geometric interpretation — three levels

![Geometric interpretation](../output/geometric_interpretation.png)

**Left:** The global state |Ψ⟩ as a fixed point on the constraint surface Ĉ = 0 within the projective Hilbert space CP(H). No temporal structure exists at this level.

**Center:** The relational bundle over clock readings. Each vertical line is a fiber (the space D(H\_S) at clock reading k). The colored dots show the section ρ\_S(k); color encodes entropy (blue = pure, red = mixed). The dashed purple line traces the section — the apparent "evolution."

**Right:** The trajectory in D(H\_S) projected onto the Bloch disk (y–z plane). Version B spirals from the sphere surface (k = 0, pure) toward the center I/2 (k = 29, mixed). The arrow of time follows the gradient of S\_eff.

### Figure 2: Bloch trajectory — purity decay ↔ entropy growth

![Bloch trajectory](../output/bloch_trajectory.png)

**Left:** Bloch disk (y–z plane). Version A traces a circle on the unit boundary (pure, reversible). Version B spirals inward, colored by S\_eff / ln 2 (blue → red). The diamond marks k = 0 (pure); the square marks k = 29 (mixed).

**Right:** Dual plot of Bloch radius |r⃗| (blue) and effective entropy S\_eff (red) vs clock reading k. The two curves are **mirror images**: as the state moves away from the surface (|r⃗| → 0), the entropy rises toward ln 2. The dashed blue line shows Version A remaining at |r⃗| = 1.

---

## 7. Connections

### To the three pillars

| Pillar | Geometric counterpart |
|--------|-----------------------|
| Pillar 1 (⟨t\|\_C → dynamics) | Defines the **base space** of the bundle and the parametric family k ↦ ρ\_S(k) |
| Pillar 2 (Tr\_E → arrow) | Determines the **location within each fiber** — drives the section toward I/2 |
| Pillar 3 (C local → frame dependence) | Different choices of C produce **different bundles** over the same |Ψ⟩ |

### To the omniscient observer

The geometric perspective illuminates the god observer analysis (see [GOD\_OBSERVER.md](GOD_OBSERVER.md)):

- **Level 1** (god with a clock): Tr\_E is trivial → section stays on the Bloch sphere surface → no spiral → no arrow.
- **Level 2** (god without a clock): no ⟨k|\_C projection → no base space → no bundle → no temporal curve.
- **Level 3** (progressive blindness): as Tr\_E becomes non-trivial, the section peels off the surface and spirals inward → the arrow strengthens.

The arrow of time is a geometric consequence of **not being omniscient**.

### To the category of relational frames

At a more abstract level, the framework can be cast as a category:
- **Objects:** subsystem partitions (choices of C, S, E)
- **Morphisms:** transformations between relational clock choices (temporal quantum reference frames)
- **Functors:** the conditional map ρ\_S(k) as a functor from clock readings to density operators

This categorical perspective is not required for the current analysis but provides a natural language for extensions involving multiple observers or dynamical frame changes (Höhn et al., 2021).

---

## 8. Formal Paragraph for the Paper

> **Geometric Structure Underlying the Framework.** The fundamental object is a stationary pure state |Ψ⟩ satisfying Ĉ|Ψ⟩ = 0, residing as a fixed point on the constraint hypersurface within the projective Hilbert space CP(H). Upon an operational tensor factorization H = H\_C ⊗ H\_S ⊗ H\_E, the conditional map k ↦ ρ\_S(k) = Tr\_E[⟨k|\_C |Ψ⟩⟨Ψ| |k⟩\_C] / p(k) defines a section of a trivial quantum bundle over the base space of distinguishable clock readings in C. The observer's projection ⟨k|\_C selects a particular fiber, generating apparent temporal evolution. The thermodynamic arrow is geometrically encoded as the monotonic increase of the von Neumann entropy S\_eff(k) along the curve ρ\_S(k) in the convex manifold of reduced density operators D(H\_S), driven by the partial trace Tr\_E — a contractive CPTP map that systematically displaces states toward the maximally mixed interior.
>
> Numerical validation confirms this picture: in the absence of Tr\_E (n\_env = 0), S\_eff remains at zero and coherent oscillations persist indefinitely (Bloch radius |r⃗| = 1); with finite inaccessible degrees (n\_env = 4), S\_eff grows to ln 2 ≈ 0.693 (|r⃗| → 0.025), tracing a spiral path toward maximal mixing. The arrow thus emerges not from fundamental asymmetry but from the geometry of partial access in the reduced state space.

---

## Reproducibility

All data and figures in this note are generated by a single script:

```bash
python generate_geometry_plots.py
```

Full Bloch trajectory data: `output/table_bloch_trajectory.csv`

Repository: [github.com/gabgiani/paw-toymodel](https://github.com/gabgiani/paw-toymodel)

---

*Back to: [Theory](THEORY.md) | [Derivation](DERIVATION.md) | [Scripts & Outputs](SCRIPTS.md) | [The Omniscient Observer](GOD_OBSERVER.md)*
