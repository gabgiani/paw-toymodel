# The Observer as a Local Breakdown of Atemporality

## What Is This?

In quantum gravity, there is a deep puzzle known as the **problem of time**: the fundamental equations of the universe (the Wheeler–DeWitt equation) describe a *timeless* state — yet we clearly experience time flowing, clocks ticking, and entropy growing. How does time emerge from a theory that has none?

In 1983, Don Page and William Wootters proposed an elegant answer: time is not fundamental but **relational**. A subsystem of the universe acts as a "clock," and the temporal evolution of everything else is recovered by asking: *"what does the system look like when the clock reads t?"* This is the **Page–Wootters (PaW) mechanism**.

This repository takes that idea and extends it. We show that a single mathematical expression — the **unified relational formula** — is enough to produce not just quantum dynamics, but also the **thermodynamic arrow of time** (why entropy grows) and **observer-dependent time** (why different observers can disagree about temporal descriptions). Three pillars of the problem of time, from one formula:

```
                    Tr_E [ ⟨t|_C  |Ψ⟩⟨Ψ|  |t⟩_C ]
    ρ_S(t)   =    ─────────────────────────────────
                                p(t)
```

The key insight is that the **observer** is not a passive spectator — it is a *local anomaly* in the global atemporality. By choosing a clock ($C$) and tracing over inaccessible degrees of freedom ($\mathrm{Tr}_E$), the observer *creates* the temporal experience. Remove the observer's limitations, and time dissolves back into the timeless whole.

### Three Pillars from One Formula

| Operation | What it produces | Pillar |
|-----------|-----------------|--------|
| ⟨t\|\_C (projection) | Schrödinger dynamics | Quantum mechanics |
| Tr\_E (partial trace) | Entropy growth, irreversibility | Thermodynamic arrow |
| C is local (not global) | Observer-dependent time | Relational time |

### What's in This Repository

Everything here is **numerically verifiable**. The repository contains:

- A **toy model** (1 qubit + up to 8 environment qubits) that demonstrates all three pillars
- **Robustness tests** showing the results survive gravitational corrections, different initial states, and Poincaré recurrences
- **IBM Quantum hardware validation** — the formula was tested on a real quantum processor (ibm\_torino), confirming the theoretical predictions with error bars
- A **boundary analysis** (the "omniscient observer") exploring what happens when you give the observer complete access — time disappears, as predicted
- **Geometric interpretations** showing how the arrow of time traces a path on the Bloch sphere
- An **independent extension** studying observational asymmetry between systems with different access structures

The mathematics draws on work by Page & Wootters (1983), Giovannetti, Lloyd & Maccone (2015), and Höhn, Smith & Lock (2021), among others. The synthesis — the three pillars from one formula, the observer as anomaly, and the numerical demonstrations — is original.

> Gabriel Giani Moreno (2026)

Implemented with [QuTiP](https://qutip.org/). Hardware validation on [IBM Quantum](https://quantum.ibm.com/).

**Repository:** [https://github.com/gabgiani/paw-toymodel](https://github.com/gabgiani/paw-toymodel)

---

## Documentation

| Document | Description |
|----------|-------------|
| **[Theory](docs/THEORY.md)** | The unified relational formula, three pillars, and the observer as anomaly |
| **[Scripts & Outputs](docs/SCRIPTS.md)** | Complete guide to every script, figure, and CSV |
| **[The Omniscient Observer](docs/GOD_OBSERVER.md)** | Boundary analysis: what happens when the observer has complete access |
| **[Geometric Structure](docs/GEOMETRY.md)** | Bloch trajectory, relational bundle, and the arrow as geometry |
| **[Step-by-Step Derivation](docs/DERIVATION.md)** | The formula developed from first principles, operation by operation |
| **[Glossary & Reading Guide](docs/GLOSSARY.md)** | Non-specialist dictionary: every term, symbol, and concept explained in plain language |
| **[Gravity Robustness](docs/THEORY.md#gravity-robustness)** | Three tests: backreaction, fuzzy boundaries, clock uncertainty |
| **[Structural Robustness](docs/THEORY.md#structural-robustness)** | Poincaré recurrences, initial states, partition independence |
| **[IBM Quantum Validation](docs/THEORY.md#experimental-validation-on-ibm-quantum-hardware)** | Pillars 1 & 2 confirmed on ibm_torino (real QPU), error bars, noise characterisation |

### Extensions

| Document | Description |
|----------|-------------|
| **[Access Asymmetry](extensions/access_asymmetry/ACCESS_ASYMMETRY.md)** | Observational asymmetry between systems with different access structures |

---

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```bash
python run_all.py                       # Pillars 1 & 2 + all metrics
python generate_pillar3_plot.py         # Pillar 3 (two-clock comparison)
python generate_god_observer_plots.py   # Boundary analysis (omniscient observer)
python generate_geometry_plots.py       # Geometric interpretation (Bloch trajectory)
python generate_gravity_robustness.py  # Gravity robustness (3 tests)
python generate_structural_robustness.py # Structural robustness (3 tests)

# Extension: Observational asymmetry (independent analysis)
python extensions/access_asymmetry/generate_access_asymmetry.py

# IBM Quantum hardware validation (requires API key)
pip install qiskit qiskit-ibm-runtime
python IBMquantum/run_ibm_validation.py --mode both
python IBMquantum/run_ibm_enhanced.py --mode all --n-runs 3  # error bars + noise
```

All figures (PNG) and tables (CSV) are saved to `output/`.

---

## Key Results at a Glance

### Pillar 1 — Projection yields dynamics

![Pillar 1](output/validation_pillar1.png)

The blue dots show ⟨σ\_z⟩ computed from the Page–Wootters formula at each clock reading *k*; the dashed black line is the analytic solution cos(ωk·dt). They overlap exactly (max deviation ~ 4×10⁻¹⁶), confirming that temporal dynamics emerge from projection alone — no external time parameter is needed.

### Pillar 2 — Partial trace yields the arrow

![Pillar 2](output/validation_unified.png)

Three panels, same formula with a 4-qubit environment added. **Left:** ⟨σ\_z⟩ now decays (green) instead of oscillating indefinitely — decoherence from the partial trace. **Center:** the von Neumann entropy S\_eff grows monotonically from 0 to ln 2 ≈ 0.693, defining the thermodynamic arrow. **Right:** direct comparison — Version A (blue, no environment, reversible) vs Version B (green, with environment, irreversible). The arrow emerges from the same formula by adding Tr\_E.

### Pillar 3 — Clock locality yields observer-dependent time

![Pillar 3](output/validation_pillar3_two_clocks.png)

Two observers choose different clocks (dt = 0.20 vs dt = 0.35) applied to the same global state |Ψ⟩. **Left panels:** ⟨σ\_z⟩ dynamics differ — faster clock (top) resolves more oscillations. **Right panels:** both observers see entropy grow to ln 2, but along different paths. Neither narrative is "correct" — both are valid relational descriptions.

---

## Structure

| File | Description |
|------|-------------|
| `paw_core.py` | Core simulation functions (reusable module) |
| `validate_formula.py` | Formula validation with step-by-step pillar verification |
| `run_all.py` | Full pipeline: all versions, metrics, plots, and CSVs |
| `generate_pillar3_plot.py` | Two-clock comparison (Pillar 3) |
| `generate_god_observer_plots.py` | Omniscient observer boundary analysis |
| `generate_geometry_plots.py` | Geometric interpretation (Bloch trajectory + bundle) |
| `generate_gravity_robustness.py` | Gravity robustness (backreaction, fuzzy boundaries, clock blur) |
| `generate_structural_robustness.py` | Structural robustness (Poincaré, initial states, partitions) |
| `IBMquantum/run_ibm_validation.py` | IBM Quantum hardware validation (Pillar 2 on real QPU) |
| `IBMquantum/run_ibm_enhanced.py` | Enhanced validation: error bars, noise characterisation, Pillar 1 on QPU |
| `extensions/access_asymmetry/` | Observational asymmetry analysis (separate from main paper) |
| `test_god_observer.py` | Console validation of three omniscience levels |
| `run_essay_validation.py` | All 3 pillars — clean ASCII output |
| `paw_toymodel.ipynb` | Interactive Jupyter notebook |
| `requirements.txt` | Python dependencies |
| `output/` | 21 figures (PNG) + 11 data tables (CSV) |
| `IBMquantum/output/` | 2 figures (PNG) + 2 data tables (CSV) + 1 noise JSON — hardware results |

## Reference Parameters

| Parameter | Value |
|-----------|-------|
| N (clock levels) | 30 |
| dt (time step) | 0.2 |
| ω (system frequency) | 1.0 |
| g (coupling strength) | 0.1 |
| n\_env (environment qubits) | 2, 4, 6, 8 |
| \|ψ₀⟩\_S (initial state) | \|0⟩ |
