# IBM Quantum Hardware Validation

Validates the unified relational formula on real quantum hardware via IBM Quantum.

## What it does

Runs a 3-qubit experiment (1 system + 2 environment) that demonstrates **Pillar 2** — the thermodynamic arrow of time emerging from the partial trace.

The Hamiltonian is Trotterized into quantum circuits:
```
U(dt) = Rx(ωdt)₀ · RXX(2gdt)₀₁ · RXX(2gdt)₀₂
```

At each step k, partial tomography of the system qubit reconstructs ρ\_S and computes S\_eff(k).

## Usage

### Step 1: Test locally (no QPU time)

```bash
cd /path/to/paw-toymodel
./venv/bin/python IBMquantum/run_ibm_validation.py --mode simulator
```

This verifies the Trotter circuits match QuTiP exactly. Run this first!

### Step 2: Run on IBM Quantum hardware

```bash
./venv/bin/python IBMquantum/run_ibm_validation.py --mode hardware
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
| `output/ibm_quantum_validation.png` | 3-panel comparison: ⟨σ\_z⟩, S\_eff, and purity |
| `output/table_ibm_quantum_validation.csv` | Full numerical data |

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
