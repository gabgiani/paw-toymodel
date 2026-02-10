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

## Output

| File | Description |
|------|-------------|
| `output/ibm_quantum_validation.png` | 3-panel comparison: ⟨σ\_z⟩, S\_eff, and purity (Pillar 2) |
| `output/table_ibm_quantum_validation.csv` | Full numerical data (Pillar 2) |
| `output/ibm_pillar3_validation.png` | 4-panel two-clock comparison (Pillar 3) |
| `output/table_ibm_pillar3.csv` | Two-clock numerical data (Pillar 3) |

## Expected Results

- **Simulator:** Trotter circuits reproduce QuTiP exact evolution with error < 0.001
- **Hardware:** S\_eff grows from 0 → 0.3–0.5 with hardware noise creating additional decoherence (which actually **strengthens** the arrow — the formula is robust)

## API Key

The script reads the API key from `../apikey.json`:
```json
{
  "apikey": "YOUR_IBM_QUANTUM_API_KEY"
}
```

Get your key from [quantum.cloud.ibm.com](https://quantum.cloud.ibm.com) → Account settings → API token.

**Important:** `apikey.json` is in `.gitignore` — never commit API keys.
