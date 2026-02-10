# Glosario y Guía de Lectura

**Para el lector curioso que no viene de la física cuántica.**

Este documento explica, en lenguaje accesible, todos los términos técnicos, símbolos y conceptos que aparecen en el repositorio. Está pensado como un diccionario de consulta: no hace falta leerlo de corrido, pero conviene tenerlo a mano mientras se recorre la documentación.

---

## Tabla de contenidos

1. [La pregunta de fondo](#1-la-pregunta-de-fondo)
2. [Conceptos fundamentales de mecánica cuántica](#2-conceptos-fundamentales-de-mecánica-cuántica)
3. [El mecanismo Page–Wootters y el problema del tiempo](#3-el-mecanismo-pagewootters-y-el-problema-del-tiempo)
4. [Vocabulario de este proyecto ("los tres pilares")](#4-vocabulario-de-este-proyecto-los-tres-pilares)
5. [Diccionario de símbolos matemáticos](#5-diccionario-de-símbolos-matemáticos)
6. [Términos de laboratorio y computación cuántica](#6-términos-de-laboratorio-y-computación-cuántica)
7. [Acrónimos](#7-acrónimos)
8. [Referencias de entrada](#8-referencias-de-entrada)

---

## 1. La pregunta de fondo

### ¿De qué trata todo esto?

La física tiene un problema serio: sus dos mejores teorías —la mecánica cuántica (que describe lo muy pequeño) y la relatividad general (que describe la gravedad y lo muy grande)— dicen cosas contradictorias sobre el tiempo.

- **Mecánica cuántica**: el tiempo es un parámetro externo, un reloj ideal que "está ahí" afuera del sistema, que la teoria no explica pero necesita.
- **Relatividad general**: el tiempo no es absoluto, depende del observador, de la gravedad y del movimiento. No hay un reloj universal.

Cuando intentamos combinarlas en una teoría de **gravedad cuántica**, aparece la **ecuación de Wheeler–DeWitt**, que describe el universo entero y dice algo perturbador: *el estado del universo no cambia*. Es estático, atemporal. Entonces, ¿de dónde sale el tiempo que experimentamos?

### La respuesta que exploramos

En 1983, Don Page y William Wootters propusieron una idea elegante: el tiempo no es una propiedad del universo, sino algo que *emerge* cuando un subsistema (nosotros) mira a otro subsistema (un reloj). El universo como un todo no evoluciona, pero las correlaciones internas entre sus partes *parecen* evolución temporal para un observador limitado.

Este repositorio toma esa idea, la formula de manera precisa con una sola ecuación, y demuestra numéricamente que de ella emergen tres cosas:

1. **La dinámica cuántica** (las cosas cambian con el tiempo)
2. **La flecha termodinámica del tiempo** (el desorden crece)
3. **Un tiempo que depende del observador** (quién mira determina qué tiempo ve)

---

## 2. Conceptos fundamentales de mecánica cuántica

Estos son los ladrillos con los que se construye todo lo demás. Si alguno de estos términos aparece en la documentación y no lo recuerda, vuelva aquí.

### Estado cuántico

La descripción completa de un sistema cuántico en un instante dado. Es el equivalente cuántico de decir "la pelota está en la posición X con velocidad Y". Pero con una diferencia crucial: un sistema cuántico puede estar en *superposición* de varios estados a la vez.

### Qubit

La unidad mínima de información cuántica. Así como un bit clásico es 0 o 1, un qubit puede ser |0⟩, |1⟩, o cualquier combinación (*superposición*) de ambos. En este proyecto, el "sistema" que estudiamos es un qubit.

### Superposición

Un estado cuántico que es combinación de varios estados base. Un qubit en superposición no está "ni en 0 ni en 1": está genuinamente en ambos a la vez, hasta que se lo mide.

### Entrelazamiento (entanglement)

Correlación cuántica entre dos o más sistemas que no tiene análogo clásico. Si dos qubits están entrelazados, medir uno afecta instantáneamente el estado del otro, sin importar la distancia. En nuestro modelo, el entrelazamiento entre el sistema y su entorno es lo que genera la flecha del tiempo.

### Operador / Observable

Un objeto matemático que representa una cantidad medible (posición, energía, spin). Los operadores de **Pauli** (σ_x, σ_y, σ_z) son los observables básicos de un qubit — miden el spin en las tres direcciones del espacio.

### Valor esperado — ⟨σ_z⟩

El promedio estadístico que obtendríamos si midiéramos σ_z muchas veces sobre copias idénticas del sistema. Si ⟨σ_z⟩ = +1, el qubit está definitivamente en |0⟩; si vale −1, está en |1⟩; si vale 0, está en superposición simétrica.

### Hamiltoniano (H)

El operador que codifica la energía total de un sistema y dicta cómo evoluciona en el tiempo. Es la "receta de movimiento" cuántica. En nuestro modelo hay varios:

| Hamiltoniano | Qué describe |
|---|---|
| H_S | La energía libre del sistema (un qubit rotando) |
| H_SE | La interacción sistema–entorno |
| H_tot | La suma de ambos |

### Evolución unitaria — U(t) = exp(−iHt)

La regla de cómo cambia un estado cuántico cerrado con el tiempo. "Unitaria" significa que es reversible y conserva la probabilidad total. Es el equivalente cuántico de las ecuaciones de Newton.

### Ecuación de Schrödinger

La ecuación fundamental: i∂_t|ψ⟩ = H|ψ⟩. Dice que el cambio temporal de un estado es proporcional a su energía. Toda la mecánica cuántica estándar se deriva de aquí.

### Estado puro vs. estado mixto

- **Estado puro** (|ψ⟩): conocemos todo lo que se puede saber del sistema. Máxima información.
- **Estado mixto** (ρ): tenemos incertidumbre, o bien porque el sistema está entrelazado con algo que no controlamos. Menos información.

### Matriz de densidad (ρ)

La representación matemática general de un estado cuántico, que sirve tanto para estados puros como mixtos. Es una matriz cuadrada que contiene toda la información estadística del sistema:

- Los elementos **diagonales** son probabilidades (cuánta chance de encontrar cada estado).
- Los elementos **fuera de la diagonal** ("coherencias") codifican la superposición.

### Traza parcial (Tr_E)

La operación clave de este proyecto. Si tenemos un sistema compuesto (sistema + entorno) y solo podemos acceder al sistema, la traza parcial *descarta* la información del entorno y nos da la descripción del sistema solo.

**Analogía**: imagine que tiene una imagen estereoscópica (3D) compuesta por dos capas. La traza parcial es como taparse un ojo: pierde la profundidad (información del entorno) pero sigue viendo una imagen (el sistema reducido). Esa pérdida de profundidad es precisamente lo que genera la flecha del tiempo.

### Decoherencia

El proceso por el cual un estado cuántico pierde sus propiedades "cuánticas" (superposición, coherencia) al interactuar con un entorno. Es lo que hace que el mundo macroscópico parezca clásico. En nuestro modelo, la decoherencia es un subproducto de la traza parcial aplicada al entorno.

### Entropía de Von Neumann — S = −Tr[ρ ln ρ]

La medida cuántica del desorden o la ignorancia. Para un qubit:
- S = 0 → estado puro, conocimiento completo.
- S = ln 2 ≈ 0.693 → estado maximalmente mixto, ignorancia total.

El crecimiento de esta entropía *es* la flecha termodinámica del tiempo.

### Fidelidad (F)

Un número entre 0 y 1 que mide cuán "parecidos" son dos estados cuánticos. F = 1 significa idénticos; F = 0 significa completamente distintos. En este proyecto, comparamos el estado del sistema obtenido por nuestra fórmula con el obtenido por la ecuación de Schrödinger estándar.

### Pureza — Tr[ρ²]

Un número entre 0 y 1 que indica cuán "puro" es un estado:
- Tr[ρ²] = 1 → estado puro.
- Tr[ρ²] = 1/d → maximalmente mixto (d = dimensión).

Su decaimiento es la otra cara de la moneda del crecimiento de entropía.

### Esfera de Bloch

Una esfera unitaria que representa geométricamente todos los estados posibles de un qubit:
- **Superficie**: estados puros (Tr[ρ²] = 1).
- **Interior (bola)**: estados mixtos.
- **Centro**: estado maximalmente mixto.

El estado se parametriza con un **vector de Bloch** r⃗ = (⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩). La longitud |r⃗| es el radio de Bloch (equivale a la pureza). Un sistema que se decoherece traza una **espiral hacia adentro** de la esfera: eso *es* la flecha del tiempo, geométricamente.

### Espacio de Hilbert (H)

El espacio matemático (vectorial, complejo, con producto interno) donde viven los estados cuánticos. En nuestro modelo, el espacio total es el producto tensorial de tres subespacios:

$$\mathcal{H} = \mathcal{H}_C \otimes \mathcal{H}_S \otimes \mathcal{H}_E$$

que corresponden al reloj (C), el sistema (S) y el entorno (E).

### Producto tensorial (⊗)

La operación matemática que combina dos espacios de Hilbert en uno mayor. Si el reloj tiene N estados y el sistema tiene 2 (un qubit), el espacio conjunto tiene 2N estados. Es la forma cuántica de decir "sistema A *y* sistema B simultáneamente."

---

## 3. El mecanismo Page–Wootters y el problema del tiempo

### Problema del tiempo

El conflicto entre mecánica cuántica (que necesita un tiempo externo absoluto) y relatividad general (que dice que ese tiempo no existe). Es uno de los problemas abiertos más importantes de la física teórica.

### Ecuación de Wheeler–DeWitt — Ĉ|Ψ⟩ = 0

La ecuación de la gravedad cuántica canónica. El operador Ĉ (constraint, restricción) es esencialmente el hamiltoniano total del universo. Que sea cero significa que el estado global |Ψ⟩ **no evoluciona**. El universo, visto desde afuera, está congelado.

### Mecanismo Page–Wootters (PaW)

La propuesta de 1983: si el universo está congelado, podemos recuperar el tiempo definiendo un subsistema como "reloj" y preguntando "¿cómo se ve el resto del universo cuando el reloj marca las 3?". Matemáticamente:

$$\rho_S(t) = \frac{ \text{Tr}_E\big[\langle t|_C \,|\Psi\rangle\langle\Psi|\, |t\rangle_C \big]}{p(t)}$$

Esta es la **fórmula relacional unificada** que vertebra todo el repositorio.

### Estado historia (history state) — |Ψ⟩

El estado global del universo en el marco PaW. Codifica *toda* la historia temporal de una vez: todas las configuraciones del sistema a todos los tiempos del reloj, entrelazadas coherentemente. Se construye como:

$$|\Psi\rangle = \frac{1}{\sqrt{N}} \sum_{k=0}^{N-1} |k\rangle_C \otimes U(t_k)|\psi_0\rangle_{SE}$$

Cada término es "a la hora k del reloj, el sistema+entorno están en el estado que corresponde tras evolucionar un tiempo t_k".

### Proyección del reloj — ⟨k|_C

La operación de "preguntar qué hora es". Al proyectar |Ψ⟩ sobre el estado de reloj |k⟩, extraemos el estado del sistema+entorno correlacionado con esa lectura del reloj. Es una especie de **actualización bayesiana**: dado que el reloj marca k, ¿qué sabemos del resto?

### p(k) — Probabilidad de lectura del reloj

La probabilidad de que, al medir el reloj, obtengamos la lectura k. En un reloj ideal con N niveles equiespaciados, p(k) = 1/N para todo k.

### Covarianza general

El principio de la relatividad general que dice que las leyes de la física no dependen del sistema de coordenadas. En gravedad cuántica, lleva a Ĉ|Ψ⟩ = 0 como condición de consistencia.

### Marcos de referencia cuántico-temporales (temporal QRF)

El marco teórico moderno (Höhn, Smith, Lock, 2021) que trata al reloj como un sistema cuántico genuino con su propia dinámica, incertidumbre y retroacción. Nuestro Pilar 3 implementa esto.

---

## 4. Vocabulario de este proyecto ("los tres pilares")

### La fórmula relacional unificada

La ecuación central:

$$\rho_S(t) = \frac{\text{Tr}_E[\langle t|_C\,|\Psi\rangle\langle\Psi|\,|t\rangle_C]}{p(t)}$$

que combina tres operaciones: proyección del reloj → traza parcial → normalización. Todo lo que sigue sale de aquí.

### Los tres pilares

El resultado principal del proyecto: la fórmula anterior, por sí sola, produce tres fenómenos:

| Pilar | Qué emerge | Operación responsable |
|---|---|---|
| **Pilar 1 — Dinámica** | La ecuación de Schrödinger | La proyección ⟨t\|_C sobre el reloj |
| **Pilar 2 — Flecha del tiempo** | La entropía crece, el tiempo tiene dirección | La traza parcial Tr_E sobre el entorno |
| **Pilar 3 — Tiempo del observador** | El tiempo depende de quién mira | El reloj es un sistema cuántico imperfecto |

### Versión A / Versión B

Dos configuraciones del modelo:
- **Versión A** (n_env = 0): sistema solo, sin entorno. La dinámica emerge perfecta (Pilar 1), pero no hay flecha del tiempo (la entropía se queda en cero).
- **Versión B** (n_env ≥ 1): sistema + entorno. Aparece la flecha (Pilar 2), con costo de una pequeña desviación en la dinámica.

### Retroacción del reloj (clock back-action)

El efecto que la dinámica del sistema+entorno tiene sobre el reloj. En un reloj ideal, es cero. En un reloj cuántico real, el reloj se perturba ligeramente. Nuestro Pilar 3 cuantifica esto con la métrica ΔE_C(k).

### El observador como anomalía

La tesis filosófica central: el observador no es un espectador pasivo "fuera" del universo, sino un subsistema *dentro* del universo cuyas limitaciones de acceso (no puede ver todo) son precisamente lo que crea la experiencia temporal. El tiempo no es una propiedad del universo; es una propiedad de la ignorancia.

### Observador omnisciente / "observador dios"

Un experimento mental: ¿qué pasa si un observador hipotético tiene acceso a *todos* los grados de libertad? No necesita hacer traza parcial, así que no pierde información, y por lo tanto no experimenta flecha del tiempo — ve un universo congelado. Este escenario se analiza en tres niveles:

| Nivel | Qué puede hacer | Qué experimenta |
|---|---|---|
| **Nivel 1** | Tiene reloj pero ve todo el entorno | Dinámica sin flecha (ρ puro, S = 0) |
| **Nivel 2** | Ni siquiera usa un reloj | Ve la matrix de densidad global (congelada) |
| **Nivel 3** | Acceso al estado puro \|Ψ⟩ | Atemporalidad absoluta |

### Estructura de acceso

Cuáles grados de libertad puede y cuáles no puede observar un subsistema. Es lo que determina la traza parcial concreta que aplica, y por lo tanto qué flecha del tiempo ve. Dos observadores con diferente estructura de acceso viven, literalmente, en tiempos distintos.

### Ceguera progresiva (progressive blindness)

Procedimiento que interpola entre el observador dios y el observador finito: se empieza viendo todo el entorno y se van "apagando" grados de libertad uno a uno. La flecha del tiempo aparece gradualmente a medida que uno pierde acceso.

### Fuerza de la flecha (arrow strength)

Métrica cuantitativa: S_final / ln 2. Mide cuán completamente se desarrolla la flecha del tiempo. Un valor de 1.0 significa entropía máxima alcanzada; un valor cercano a 0 significa que la flecha apenas apareció.

### Monotonicidad (monotonicity score)

Fracción de pasos temporales en los que la entropía efectivamente crece. Un puntaje de 1.0 significa que la entropía creció en *cada* paso sin excepción.

### Asimetría observacional (observational asymmetry)

El resultado de la extensión `access_asymmetry`: dos subsistemas que comparten un entorno no se ven simétricamente. Uno puede detectar al otro; el otro no puede detectar al primero. La visibilidad mutua depende de la estructura de acceso de cada uno.

### Señal de detección — Δ_det(k)

Cuánto cambia el estado de un subsistema cuando hay otro subsistema acoplado vs. cuando no lo hay. Si Δ_det ≈ 0, el segundo subsistema es indetectable.

### Información mutua — I(A:B)

Medida de todas las correlaciones (clásicas y cuánticas) entre dos subsistemas A y B:

$$I(A:B) = S(\rho_A) + S(\rho_B) - S(\rho_{AB})$$

Si I = 0, los subsistemas están completamente decorrelacionados.

---

## 5. Diccionario de símbolos matemáticos

Para consulta rápida cuando aparecen en ecuaciones o en el código.

| Símbolo | Se lee como | Significado |
|---|---|---|
| \|ψ⟩ | "ket psi" | Vector de estado cuántico (notación de Dirac) |
| ⟨ψ\| | "bra psi" | Vector dual (conjugado transpuesto de \|ψ⟩) |
| \|Ψ⟩⟨Ψ\| | "proyector sobre Psi" | Matriz de densidad del estado puro \|Ψ⟩ |
| ρ_S(k) | "ro sub S de k" | Matriz de densidad reducida del sistema a la hora k |
| Tr_E[...] | "traza parcial sobre E" | Descarta los grados de libertad del entorno |
| ⟨k\|_C | "bra k del reloj" | Proyección sobre la lectura k del reloj |
| σ_x, σ_y, σ_z | "sigma x, y, z" | Matrices de Pauli — observables del qubit |
| ⟨σ_z⟩(k) | "valor esperado de sigma z en k" | Promedio de la medición de spin-z a la hora k |
| H | "hache" o "hamiltoniano" | Operador de energía del sistema |
| U(t) | "u de t" | Operador de evolución temporal |
| S(ρ) | "entropía de ro" | Entropía de Von Neumann |
| F(k) | "fidelidad en k" | Overlap entre el estado PaW y el de Schrödinger |
| ⊗ | "tensor" | Producto tensorial de espacios o estados |
| Ĉ | "ce sombrero" o "constraint" | Operador de restricción (energia total = 0) |
| N | "ene" | Número de lecturas del reloj (pasos temporales) |
| n_env | "ene sub env" | Número de qubits del entorno |
| ω | "omega" | Frecuencia del qubit (velocidad de rotación) |
| g | "ge" | Constante de acoplamiento sistema–entorno |
| dt, Δt | "delta t" | Paso temporal (espaciado entre lecturas del reloj) |
| \|0⟩, \|1⟩ | "ket cero, ket uno" | Estados base del qubit (spin arriba / spin abajo) |
| I, I/2 | "identidad" / "identidad sobre dos" | Operador identidad / estado maximalmente mixto |
| ln 2 ≈ 0.693 | "logaritmo natural de 2" | Entropía máxima de un qubit |
| r⃗ | "vector r" | Vector de Bloch (posición del estado en la esfera) |
| \|r⃗\| | "módulo de r" | Radio de Bloch (= pureza geométrica) |

---

## 6. Términos de laboratorio y computación cuántica

Estos aparecen en las secciones sobre la validación en hardware IBM Quantum.

| Término | Significado |
|---|---|
| **QPU** | Unidad de Procesamiento Cuántico — el chip de hardware cuántico real |
| **IBM Quantum / ibm_torino** | Plataforma de computación cuántica en la nube de IBM; `ibm_torino` es el procesador específico usado |
| **Qubits superconductores** | Tipo de qubit físico basado en circuitos superconductores enfriados a −273°C |
| **Compuerta cuántica (gate)** | Operación elemental sobre uno o dos qubits (equivalente a una puerta lógica clásica) |
| **SX gate** | Compuerta √X: media rotación alrededor del eje X |
| **CZ gate** | Compuerta Controlled-Z: compuerta de dos qubits |
| **RXX gate** | Rotación de dos qubits alrededor de X⊗X |
| **Shots** | Repeticiones de un experimento cuántico para acumular estadística |
| **Error de lectura (readout error)** | Probabilidad de leer 0 cuando el qubit estaba en 1 (o viceversa) |
| **Error de compuerta (gate error)** | Imprecisión al aplicar una operación cuántica |
| **T₁, T₂** | Tiempos de coherencia del qubit: T₁ = relajación (pérdida de energía), T₂ = defasaje (pérdida de fase) |
| **Tomografía parcial** | Reconstrucción del estado cuántico a partir de estadística de mediciones |
| **Barras de error** | Incertidumbre estadística en los resultados (típicamente ±1 desviación estándar) |
| **Descomposición de Trotter** | Técnica para aproximar la evolución bajo un hamiltoniano complejo como secuencia de operaciones simples |

---

## 7. Acrónimos

| Sigla | Significado |
|---|---|
| **PaW** | Page–Wootters (mecanismo de) |
| **QRF** | Marco de Referencia Cuántico (Quantum Reference Frame) |
| **CPTP** | Completamente Positivo, Preserva la Traza — tipo de operación cuántica permitida |
| **POVM** | Medida con Operador de Valor Positivo — medición cuántica generalizada |
| **QuTiP** | Quantum Toolbox in Python — librería de simulación cuántica usada en este proyecto |
| **QPU** | Unidad de Procesamiento Cuántico |
| **IBM** | International Business Machines (aquí: IBM Quantum) |

---

## 8. Referencias de entrada

Para quien quiera profundizar, estas son las fuentes primarias organizadas por nivel de accesibilidad.

### Nivel divulgativo (sin ecuaciones)

- **Sean Carroll**, *Something Deeply Hidden* (2019): excelente introducción a la mecánica cuántica y los fundamentos, incluyendo el problema de la medición.
- **Carlo Rovelli**, *The Order of Time* (2018): la naturaleza del tiempo desde la perspectiva de la gravedad cuántica, escrito para público general.
- **Lee Smolin**, *Time Reborn* (2013): argumentos sobre por qué el tiempo debería ser fundamental, perspectiva contraria pero muy clara.

### Nivel intermedio (algunas ecuaciones)

- **Giovannetti, Lloyd & Maccone**, "Quantum time" (Physical Review D, 2015): la formalización moderna del mecanismo PaW con lenguaje de teoría de la información cuántica.
- **W. H. Zurek**, "Decoherence, einselection, and the quantum origins of the classical" (Reviews of Modern Physics, 2003): referencia estándar sobre decoherencia.

### Nivel especializado (paper original y desarrollos)

- **Page & Wootters**, "Evolution without evolution" (Physical Review D, 1983): el paper fundacional.
- **Höhn, Smith & Lock**, "Equivalence of approaches to relational quantum dynamics" (Physical Review D, 2021): la formulación en marcos de referencia cuánticos que usamos para el Pilar 3.
- **Shaari**, "Entanglement, decoherence and arrow of time" (2014): la conexión entre traza parcial y la flecha termodinámica que fundamenta nuestro Pilar 2.

### Recursos online

- [Qiskit Textbook — Quantum States and Qubits](https://learning.quantum.ibm.com/): tutorial interactivo gratuito de IBM sobre computación cuántica.
- [QuTiP documentation](https://qutip.org/docs/latest/): documentación de la librería de simulación usada en el código.
- [Stanford Encyclopedia of Philosophy — The Problem of Time in Quantum Gravity](https://plato.stanford.edu/entries/quantum-gravity/): tratamiento filosófico riguroso del problema del tiempo.

---

*Este glosario cubre los ~200 términos técnicos, símbolos y conceptos que aparecen en la documentación del repositorio. Ante cualquier término no incluido aquí, abrir un issue en GitHub.*
