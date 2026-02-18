# IBM Quantum Hardware Validation

Validates the unified relational time formula on real quantum hardware via IBM Quantum.

## What it does

Runs 3-qubit experiments (1 system + 2 environment) that demonstrate all three pillars of the unified relational time formula on quantum circuits:

- **Pillar 1** — Pure Schrödinger dynamics (1 qubit, no environment)
- **Pillar 2** — Thermodynamic arrow of time (3 qubits, partial trace)
- **Pillar 3** — Observer-dependent time (3 qubits, two different clocks)

The Hamiltonian is Trotterized into quantum circuits:

$$U(dt) = R_x(\omega\,dt)_0 \cdot R_{XX}(2g\,dt)_{01} \cdot R_{XX}(2g\,dt)_{02}$$

At each step k, partial tomography of the system qubit reconstructs ρ\_S and computes S\_eff(k).

## Usage

### Step 1: Test locally (no QPU time)

```bash
cd /path/to/paw-toymodel
./venv/bin/python IBMquantum/run_ibm_validation.py --mode simulator
```

This verifies the Trotter circuits match QuTiP exactly. Run this first!

### Step 2: Pillar 3 — two-clock comparison (simulator)

```bash
./venv/bin/python IBMquantum/run_ibm_pillar3.py --mode simulator
```

### Step 3: Run on IBM Quantum hardware

```bash
./venv/bin/python IBMquantum/run_ibm_validation.py --mode hardware
./venv/bin/python IBMquantum/run_ibm_pillar3.py --mode hardware   # if QPU budget allows
```

Or both in one run:
```bash
./venv/bin/python IBMquantum/run_ibm_validation.py --mode both
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `simulator` | `simulator`, `hardware`, or `both` |
| `--shots` | `4096` | Shots per circuit (hardware only) |

## QPU Budget

- **Estimated QPU time:** ~30 seconds (21 circuits × 3 observables × 4096 shots)
- **Free tier:** 10 minutes/month — this uses ~5% of your monthly budget
- **Queue wait:** 5–30 minutes (not counted against QPU budget)

### Step 4: Stability & uniqueness validation (simulator)

```bash
./venv/bin/python IBMquantum/run_ibm_stability.py --mode simulator
```

This validates the stability theorems (Thm 3.1, 8.1) on quantum circuits — confirming that the tensor product structure is robust under perturbation.

### Step 5: Stability on IBM Quantum hardware

```bash
./venv/bin/python IBMquantum/run_ibm_stability.py --mode hardware
```

## Output

| File | Description |
|------|-------------|
| `output/ibm_quantum_validation.png` | 3-panel comparison: ⟨σ\_z⟩, S\_eff, and purity (Pillar 2) |
| `output/table_ibm_quantum_validation.csv` | Full numerical data (Pillar 2) |
| `output/ibm_pillar3_validation.png` | 4-panel two-clock comparison (Pillar 3) |
| `output/table_ibm_pillar3.csv` | Two-clock numerical data (Pillar 3) |
| `output/ibm_stability_validation.png` | 6-panel stability validation (purity deficit + MI scaling) |
| `output/stability_time_evolution.csv` | Time evolution data for stability analysis |
| `output/stability_eta_scaling.csv` | η-scaling fit data (purity deficit and MI vs λ) |

## Expected Results

- **Simulator:** Trotter circuits reproduce QuTiP exact evolution with error < 0.001
- **Hardware (Pillar 2):** S\_eff grows from 0 → 0.583 ± 0.005 (102.2% of exact). Hardware noise adds decoherence, which **strengthens** the arrow
- **Hardware (Pillar 3):** Two clocks differ by 0.69 in ⟨σ\_z⟩ and 0.14 in S\_eff — observer-dependence confirmed on ibm\_torino
- **Stability (Simulator):** Purity deficit scales as λ^2.03 (1.3% from theory); MI scales as λ^1.78 (expected log correction for pure states)

### Stability Validation Details

The stability script (`run_ibm_stability.py`) validates the mathematical theorems from the [stability analysis](../stability/README.md) on quantum circuits. It uses a 2-qubit Heisenberg model with Trotterized evolution and scans across 14 coupling strengths to verify:

- **Purity deficit** Δ = 1 − Tr(ρ²) scales as λ² — confirming Theorem 3.1
- **Mutual information** I(S:E) scales as λ² with a logarithmic correction — explained by Remark 3.3

![Stability Validation](output/ibm_stability_validation.png)

## API Key

The script reads the API key from `../apikey.json`:
```json
{
  "apikey": "YOUR_IBM_QUANTUM_API_KEY"
}
```

Get your key from [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com) → Account settings → API token.

**Important:** `apikey.json` is in `.gitignore` — never commit API keys.
