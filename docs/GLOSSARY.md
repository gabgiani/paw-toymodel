# Glossary and Reading Guide

**For the curious reader who does not come from quantum physics.**

This document explains, in accessible language, all the technical terms, symbols, and concepts that appear in the repository. It is designed as a reference dictionary: you do not need to read it cover to cover, but it is worth keeping at hand while browsing the documentation.

---

## Table of Contents

1. [The Big Question](#1-the-big-question)
2. [Fundamental Concepts of Quantum Mechanics](#2-fundamental-concepts-of-quantum-mechanics)
3. [The Page–Wootters Mechanism and the Problem of Time](#3-the-pagewootters-mechanism-and-the-problem-of-time)
4. [Vocabulary of This Project ("the Three Pillars")](#4-vocabulary-of-this-project-the-three-pillars)
5. [Mathematical Symbols Dictionary](#5-mathematical-symbols-dictionary)
6. [Laboratory and Quantum Computing Terms](#6-laboratory-and-quantum-computing-terms)
7. [Acronyms](#7-acronyms)
8. [Entry-Level References](#8-entry-level-references)

---

## 1. The Big Question

### What is all of this about?

Physics has a serious problem: its two best theories — quantum mechanics (which describes the very small) and general relativity (which describes gravity and the very large) — say contradictory things about time.

- **Quantum mechanics**: time is an external parameter, an ideal clock that "sits out there" outside the system, which the theory does not explain but needs.
- **General relativity**: time is not absolute; it depends on the observer, on gravity, and on motion. There is no universal clock.

When we try to combine them into a theory of **quantum gravity**, the **Wheeler–DeWitt equation** appears, which describes the entire universe and says something disturbing: *the state of the universe does not change*. It is static, timeless. So where does the time we experience come from?

### The answer we explore

In 1983, Don Page and William Wootters proposed an elegant idea: time is not a property of the universe, but something that *emerges* when one subsystem (us) looks at another subsystem (a clock). The universe as a whole does not evolve, but the internal correlations between its parts *look like* temporal evolution to a limited observer.

This repository takes that idea, formulates it precisely with a single equation, and demonstrates numerically that three things emerge from it:

1. **Quantum dynamics** (things change with time)
2. **The thermodynamic arrow of time** (disorder grows)
3. **Observer-dependent time** (who looks determines what time they see)

---

## 2. Fundamental Concepts of Quantum Mechanics

These are the building blocks on which everything else is constructed. If any of these terms appears in the documentation and you do not remember it, come back here.

### Quantum State

The complete description of a quantum system at a given instant. It is the quantum equivalent of saying "the ball is at position X with velocity Y." But with a crucial difference: a quantum system can be in a *superposition* of several states at once.

### Qubit

The minimal unit of quantum information. Just as a classical bit is 0 or 1, a qubit can be |0⟩, |1⟩, or any combination (*superposition*) of both. In this project, the "system" we study is a qubit.

### Superposition

A quantum state that is a combination of several basis states. A qubit in superposition is not "neither 0 nor 1": it is genuinely in both at the same time, until it is measured.

### Entanglement

A quantum correlation between two or more systems that has no classical analogue. If two qubits are entangled, measuring one instantaneously affects the state of the other, regardless of the distance. In our model, the entanglement between the system and its environment is what generates the arrow of time.

### Operator / Observable

A mathematical object that represents a measurable quantity (position, energy, spin). The **Pauli** operators (σ_x, σ_y, σ_z) are the basic observables of a qubit — they measure spin in the three spatial directions.

### Expectation Value — ⟨σ_z⟩

The statistical average we would obtain if we measured σ_z many times on identical copies of the system. If ⟨σ_z⟩ = +1, the qubit is definitely in |0⟩; if it equals −1, it is in |1⟩; if it equals 0, it is in symmetric superposition.

### Hamiltonian (H)

The operator that encodes the total energy of a system and dictates how it evolves in time. It is the quantum "recipe for motion." In our model there are several:

| Hamiltonian | What it describes |
|---|---|
| H_S | The free energy of the system (a qubit rotating) |
| H_SE | The system–environment interaction |
| H_tot | The sum of both |

### Unitary Evolution — U(t) = exp(−iHt)

The rule for how a closed quantum state changes over time. "Unitary" means it is reversible and conserves total probability. It is the quantum equivalent of Newton's equations.

### Schrödinger Equation

The fundamental equation: i∂_t|ψ⟩ = H|ψ⟩. It says that the temporal change of a state is proportional to its energy. All of standard quantum mechanics is derived from here.

### Pure State vs. Mixed State

- **Pure state** (|ψ⟩): we know everything that can be known about the system. Maximum information.
- **Mixed state** (ρ): we have uncertainty, either because the system is entangled with something we do not control. Less information.

### Density Matrix (ρ)

The general mathematical representation of a quantum state, valid for both pure and mixed states. It is a square matrix containing all the statistical information about the system:

- The **diagonal** elements are probabilities (the chance of finding each state).
- The **off-diagonal** elements ("coherences") encode superposition.

### Partial Trace (Tr_E)

The key operation of this project. If we have a composite system (system + environment) and can only access the system, the partial trace *discards* the environment's information and gives us the description of the system alone.

**Analogy**: imagine you have a stereoscopic (3D) image composed of two layers. The partial trace is like covering one eye: you lose the depth (environment information) but still see an image (the reduced system). That loss of depth is precisely what generates the arrow of time.

### Decoherence

The process by which a quantum state loses its "quantum" properties (superposition, coherence) upon interacting with an environment. It is what makes the macroscopic world appear classical. In our model, decoherence is a byproduct of the partial trace applied over the environment.

### Von Neumann Entropy — S = −Tr[ρ ln ρ]

The quantum measure of disorder or ignorance. For a qubit:
- S = 0 → pure state, complete knowledge.
- S = ln 2 ≈ 0.693 → maximally mixed state, total ignorance.

The growth of this entropy *is* the thermodynamic arrow of time.

### Fidelity (F)

A number between 0 and 1 that measures how "similar" two quantum states are. F = 1 means identical; F = 0 means completely different. In this project, we compare the system state obtained by our formula with the one obtained by the standard Schrödinger equation.

### Purity — Tr[ρ²]

A number between 0 and 1 indicating how "pure" a state is:
- Tr[ρ²] = 1 → pure state.
- Tr[ρ²] = 1/d → maximally mixed (d = dimension).

Its decay is the flip side of entropy growth.

### Bloch Sphere

A unit sphere that geometrically represents all possible states of a qubit:
- **Surface**: pure states (Tr[ρ²] = 1).
- **Interior (ball)**: mixed states.
- **Center**: maximally mixed state.

The state is parameterized by a **Bloch vector** r⃗ = (⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩). The length |r⃗| is the Bloch radius (equivalent to purity). A system undergoing decoherence traces an **inward spiral** on the sphere: that *is* the arrow of time, geometrically.

### Hilbert Space (H)

The mathematical space (vector, complex, with inner product) where quantum states live. In our model, the total space is the tensor product of three subspaces:

$$\mathcal{H} = \mathcal{H}_C \otimes \mathcal{H}_S \otimes \mathcal{H}_E$$

corresponding to the clock (C), the system (S), and the environment (E).

### Tensor Product (⊗)

The mathematical operation that combines two Hilbert spaces into a larger one. If the clock has N states and the system has 2 (one qubit), the joint space has 2N states. It is the quantum way of saying "system A *and* system B simultaneously."

---

## 3. The Page–Wootters Mechanism and the Problem of Time

### Problem of Time

The conflict between quantum mechanics (which needs an external absolute time) and general relativity (which says such time does not exist). It is one of the most important open problems in theoretical physics.

### Wheeler–DeWitt Equation — Ĉ|Ψ⟩ = 0

The equation of canonical quantum gravity. The operator Ĉ (constraint) is essentially the total Hamiltonian of the universe. That it equals zero means the global state |Ψ⟩ **does not evolve**. The universe, seen from outside, is frozen.

### Page–Wootters Mechanism (PaW)

The 1983 proposal: if the universe is frozen, we can recover time by defining a subsystem as a "clock" and asking "what does the rest of the universe look like when the clock reads 3?" Mathematically:

$$\rho_S(t) = \frac{ \text{Tr}_E\big[\langle t|_C \,|\Psi\rangle\langle\Psi|\, |t\rangle_C \big]}{p(t)}$$

This is the **unified relational formula** that underpins the entire repository.

### History State — |Ψ⟩

The global state of the universe in the PaW framework. It encodes the *entire* temporal history at once: all configurations of the system at all clock times, coherently entangled. It is constructed as:

$$|\Psi\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} |k\rangle_C \otimes U(t_k)|\psi_0\rangle_{SE}$$

Each term reads "at clock hour k, the system+environment are in the state corresponding to having evolved for a time t_k."

### Clock Projection — ⟨k|_C

The operation of "asking what time it is." By projecting |Ψ⟩ onto the clock state |k⟩, we extract the system+environment state correlated with that clock reading. It is a kind of **Bayesian update**: given that the clock reads k, what do we know about the rest?

### p(k) — Clock Reading Probability

The probability that, upon measuring the clock, we obtain reading k. For an ideal clock with N equally spaced levels, p(k) = 1/N for all k.

### General Covariance

The principle from general relativity stating that the laws of physics do not depend on the coordinate system. In quantum gravity, it leads to Ĉ|Ψ⟩ = 0 as a consistency condition.

### Temporal Quantum Reference Frames (temporal QRF)

The modern theoretical framework (Höhn, Smith, Lock, 2021) that treats the clock as a genuine quantum system with its own dynamics, uncertainty, and backreaction. Our Pillar 3 implements this.

---

## 4. Vocabulary of This Project ("the Three Pillars")

### The Unified Relational Formula

The central equation:

$$\rho_S(t) = \frac{\text{Tr}_E[\langle t|_C\,|\Psi\rangle\langle\Psi|\,|t\rangle_C]}{p(t)}$$

which combines three operations: clock projection → partial trace → normalization. Everything that follows comes from here.

### The Three Pillars

The main result of the project: the formula above, by itself, produces three phenomena:

| Pillar | What emerges | Responsible operation |
|---|---|---|
| **Pillar 1 — Dynamics** | The Schrödinger equation | The projection ⟨t\|_C onto the clock |
| **Pillar 2 — Arrow of Time** | Entropy grows, time acquires direction | The partial trace Tr_E over the environment |
| **Pillar 3 — Observer Time** | Time depends on who is looking | The clock is an imperfect quantum system |

### Version A / Version B

Two model configurations:
- **Version A** (n_env = 0): system alone, no environment. Dynamics emerge perfectly (Pillar 1), but there is no arrow of time (entropy stays at zero).
- **Version B** (n_env ≥ 1): system + environment. The arrow appears (Pillar 2), at the cost of a small deviation in the dynamics.

### Clock Back-Action

The effect that the system+environment dynamics have on the clock. For an ideal clock, it is zero. For a real quantum clock, the clock is slightly perturbed. Our Pillar 3 quantifies this with the metric ΔE_C(k).

### The Observer as Anomaly

The central philosophical thesis: the observer is not a passive spectator "outside" the universe, but a subsystem *within* the universe whose access limitations (it cannot see everything) are precisely what creates temporal experience. Time is not a property of the universe; it is a property of ignorance.

### Omniscient Observer / "God Observer"

A thought experiment: what happens if a hypothetical observer has access to *all* degrees of freedom? It does not need to perform a partial trace, so it loses no information, and therefore experiences no arrow of time — it sees a frozen universe. This scenario is analyzed at three levels:

| Level | What it can do | What it experiences |
|---|---|---|
| **Level 1** | Has a clock but sees the entire environment | Dynamics without an arrow (ρ pure, S = 0) |
| **Level 2** | Does not even use a clock | Sees the global density matrix (frozen) |
| **Level 3** | Access to the pure state \|Ψ⟩ | Absolute atemporality |

### Access Structure

Which degrees of freedom a subsystem can and cannot observe. This determines the specific partial trace it applies, and therefore what arrow of time it sees. Two observers with different access structures live, literally, in different times.

### Progressive Blindness

A procedure that interpolates between the god observer and the finite observer: one starts seeing the entire environment and then "switches off" degrees of freedom one by one. The arrow of time appears gradually as access is lost.

### Arrow Strength

Quantitative metric: S_final / ln 2. Measures how completely the arrow of time develops. A value of 1.0 means maximum entropy is reached; a value near 0 means the arrow barely appeared.

### Monotonicity Score

The fraction of time steps in which entropy actually grows. A score of 1.0 means entropy increased at *every* step without exception.

### Observational Asymmetry

The result from the `access_asymmetry` extension: two subsystems sharing an environment do not see each other symmetrically. One can detect the other; the other cannot detect the first. Mutual visibility depends on each one's access structure.

### Detection Signal — Δ_det(k)

How much a subsystem's state changes when another subsystem is coupled vs. when it is not. If Δ_det ≈ 0, the second subsystem is undetectable.

### Mutual Information — I(A:B)

A measure of all correlations (classical and quantum) between two subsystems A and B:

$$I(A:B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})$$

If I = 0, the subsystems are completely uncorrelated.

---

## 5. Mathematical Symbols Dictionary

For quick reference when they appear in equations or in the code.

| Symbol | Read as | Meaning |
|---|---|---|
| \|ψ⟩ | "ket psi" | Quantum state vector (Dirac notation) |
| ⟨ψ\| | "bra psi" | Dual vector (conjugate transpose of \|ψ⟩) |
| \|Ψ⟩⟨Ψ\| | "projector onto Psi" | Density matrix of the pure state \|Ψ⟩ |
| ρ_S(k) | "rho sub S of k" | Reduced density matrix of the system at clock hour k |
| Tr_E[...] | "partial trace over E" | Discards the environment's degrees of freedom |
| ⟨k\|_C | "bra k of the clock" | Projection onto clock reading k |
| σ_x, σ_y, σ_z | "sigma x, y, z" | Pauli matrices — qubit observables |
| ⟨σ_z⟩(k) | "expectation value of sigma z at k" | Average of the spin-z measurement at clock hour k |
| H | "H" or "Hamiltonian" | Energy operator of the system |
| U(t) | "U of t" | Time evolution operator |
| S(ρ) | "entropy of rho" | Von Neumann entropy |
| F(k) | "fidelity at k" | Overlap between the PaW state and the Schrödinger state |
| ⊗ | "tensor" | Tensor product of spaces or states |
| Ĉ | "C hat" or "constraint" | Constraint operator (total energy = 0) |
| N | "N" | Number of clock readings (time steps) |
| n_env | "n sub env" | Number of environment qubits |
| ω | "omega" | Qubit frequency (rotation speed) |
| g | "g" | System–environment coupling constant |
| dt, Δt | "delta t" | Time step (spacing between clock readings) |
| \|0⟩, \|1⟩ | "ket zero, ket one" | Qubit basis states (spin up / spin down) |
| I, I/2 | "identity" / "identity over two" | Identity operator / maximally mixed state |
| ln 2 ≈ 0.693 | "natural logarithm of 2" | Maximum entropy of a qubit |
| r⃗ | "vector r" | Bloch vector (state position on the sphere) |
| \|r⃗\| | "magnitude of r" | Bloch radius (= geometric purity) |

---

## 6. Laboratory and Quantum Computing Terms

These appear in the sections about validation on IBM Quantum hardware.

| Term | Meaning |
|---|---|
| **QPU** | Quantum Processing Unit — the actual quantum hardware chip |
| **IBM Quantum / ibm_torino** | IBM's cloud quantum computing platform; `ibm_torino` is the specific processor used |
| **Superconducting qubits** | A type of physical qubit based on superconducting circuits cooled to −273°C |
| **Quantum gate** | An elementary operation on one or two qubits (equivalent to a classical logic gate) |
| **SX gate** | The √X gate: half rotation around the X axis |
| **CZ gate** | Controlled-Z gate: a two-qubit gate |
| **RXX gate** | Two-qubit rotation around X⊗X |
| **Shots** | Repetitions of a quantum experiment to accumulate statistics |
| **Readout error** | Probability of reading 0 when the qubit was in 1 (or vice versa) |
| **Gate error** | Imprecision when applying a quantum operation |
| **T₁, T₂** | Qubit coherence times: T₁ = relaxation (energy loss), T₂ = dephasing (phase loss) |
| **Partial tomography** | Reconstruction of a quantum state from measurement statistics |
| **Error bars** | Statistical uncertainty in the results (typically ±1 standard deviation) |
| **Trotter decomposition** | Technique to approximate evolution under a complex Hamiltonian as a sequence of simple operations |

---

## 7. Acronyms

| Acronym | Meaning |
|---|---|
| **PaW** | Page–Wootters (mechanism) |
| **QRF** | Quantum Reference Frame |
| **CPTP** | Completely Positive, Trace-Preserving — a type of allowed quantum operation |
| **POVM** | Positive Operator-Valued Measure — a generalized quantum measurement |
| **QuTiP** | Quantum Toolbox in Python — the quantum simulation library used in this project |
| **QPU** | Quantum Processing Unit |
| **IBM** | International Business Machines (here: IBM Quantum) |

---

## 8. Entry-Level References

For those who want to go deeper, these are the primary sources organized by accessibility level.

### Popular Level (no equations)

- **Sean Carroll**, *Something Deeply Hidden* (2019): excellent introduction to quantum mechanics and foundations, including the measurement problem.
- **Carlo Rovelli**, *The Order of Time* (2018): the nature of time from the perspective of quantum gravity, written for a general audience.
- **Lee Smolin**, *Time Reborn* (2013): arguments for why time should be fundamental — an opposing perspective, but very clearly presented.

### Intermediate Level (some equations)

- **Giovannetti, Lloyd & Maccone**, "Quantum time" (Physical Review D, 2015): the modern formalization of the PaW mechanism using quantum information theory language.
- **W. H. Zurek**, "Decoherence, einselection, and the quantum origins of the classical" (Reviews of Modern Physics, 2003): the standard reference on decoherence.

### Specialist Level (original paper and developments)

- **Page & Wootters**, "Evolution without evolution" (Physical Review D, 1983): the foundational paper.
- **Höhn, Smith & Lock**, "Equivalence of approaches to relational quantum dynamics" (Physical Review D, 2021): the quantum reference frame formulation we use for Pillar 3.
- **Shaari**, "Entanglement, decoherence and arrow of time" (2014): the connection between partial trace and the thermodynamic arrow that underpins our Pillar 2.

### Online Resources

- [Qiskit Textbook — Quantum States and Qubits](https://learning.quantum.ibm.com/): free interactive tutorial from IBM on quantum computing.
- [QuTiP documentation](https://qutip.org/docs/latest/): documentation for the simulation library used in the code.
- [Stanford Encyclopedia of Philosophy — The Problem of Time in Quantum Gravity](https://plato.stanford.edu/entries/quantum-gravity/): a rigorous philosophical treatment of the problem of time.

---

*This glossary covers the ~200 technical terms, symbols, and concepts that appear in the repository documentation. For any term not included here, open an issue on GitHub.*
