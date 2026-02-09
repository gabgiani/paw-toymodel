# PaW Toy Model — Page–Wootters Mechanism Demonstrator

Minimal numerical demonstrator for the paper:

> **"The Observer as a Local Breakdown of Atemporality: Relational Time and an Informational Arrow from Quantum Clocks"**
> Gabriel Giani Moreno (2026)

Implements Sections 5–6 and Appendix A using [QuTiP](https://qutip.org/).

**Repository:** [https://github.com/gabgiani/paw-toymodel](https://github.com/gabgiani/paw-toymodel)

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python run_all.py                    # Full pipeline: all versions + plots + CSVs
python validate_formula.py           # Formula validation (Pillars 1 & 2)
python run_essay_validation.py       # Essay validation (all 3 pillars, ASCII output)
python generate_pillar3_plot.py      # Two-clock comparison (Pillar 3 plot + CSV)
```

All figures (PNG) and tables (CSV) are saved to `output/`.

## Structure

| File | Description |
|------|-------------|
| `paw_core.py` | Core simulation functions (reusable module) |
| `run_all.py` | Full pipeline: runs both versions, generates all plots and CSVs |
| `validate_formula.py` | Formula validation with step-by-step pillar verification |
| `run_essay_validation.py` | All 3 pillars validation with clean ASCII output |
| `generate_pillar3_plot.py` | Two-clock comparison (Pillar 3) — plot and CSV |
| `paw_toymodel.ipynb` | Interactive Jupyter notebook |
| `paper3.tex` | LaTeX source of the paper |
| `essay_v2` | Philosophical essay (English) |
| `requirements.txt` | Python dependencies |
| `output/` | Generated figures and tables |

## What it computes

### Version A — No Environment (Pillar 1: Quantum Dynamics)
- PaW history state with N=30 clock levels
- Conditional ⟨σ_z⟩(k) → clean sinusoidal oscillation
- Comparison with theoretical cos(ωkdt)
- Machine-precision agreement (max deviation ~ 4×10⁻¹⁶)

### Version B — With Environment (Pillar 2: Thermodynamic Arrow)
- PaW history state with system–environment coupling
- Damped oscillations in ⟨σ_z⟩ (effective decoherence)
- Growing effective entropy S_eff(k) from 0 → ln 2 ≈ 0.693 (informational arrow)
- Fidelity vs ideal Schrödinger evolution
- Multi-environment comparison: n_env ∈ {2, 4, 6, 8}

### Version C — Two-Clock Comparison (Pillar 3: Observer-Dependent Time)
- Same global state |Ψ⟩, same formula, two different clock spacings (dt=0.2 vs dt=0.35)
- Different oscillation frequencies, damping rates, and entropy trajectories
- Demonstrates partition dependence: time is relational, not absolute

### Metrics (Sec. 4)
- **Back-action** ΔE_C(k) — clock disturbance metric
- **Fidelity** F²(k) — deviation from ideal dynamics
- **Entropy** S_eff(k) — informational arrow indicator

## Reference Parameters

| Parameter | Value |
|-----------|-------|
| N (clock levels) | 30 |
| dt (time step) | 0.2 |
| ω (system frequency) | 1.0 |
| g (coupling strength) | 0.1 |
| n_env (environment qubits) | 2, 4, 6, 8 |
| \|ψ₀⟩_S (initial state) | \|0⟩ |

## Important Correction

The paper specifies H_S = (ω/2)σ_z. However, σ_z generates only phase
rotations, leaving ⟨σ_z⟩ constant for any initial state. To reproduce the
claimed ⟨σ_z⟩(t) = cos(ωt) oscillations, we use **H_S = (ω/2)σ_x** with
initial state |0⟩. This is related by a π/2 basis rotation and is
physically equivalent.
