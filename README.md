# CDS-Polytope

**A Universal 4D Similarity Measure based on 24-Cell Polytope Geometry**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![arXiv](https://img.shields.io/badge/arXiv-cs.LG-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![ESSOAr](https://img.shields.io/badge/ESSOAr-Paper3-orange)](https://essoar.org/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

---

## TL;DR — for the ML / AI researcher

> We map any 4D event cloud onto the 24-cell regular polytope and compute
> a **cosine alignment index (cosD)** against the uniform baseline.
>
> **Empirical finding:** cosD is a surrogate Lyapunov exponent.
> When the Lorenz system crosses its critical transition (ρ = 24.74),
> cosD drops from 0.72 → 0.29 — **a geometric early-warning signal
> detectable without knowing the governing equations**.
>
> The same algorithm — zero code change — detects critical transitions
> in earthquake catalogs, typhoon tracks, ENSO indices,
> financial time series, EEG, and fracture acoustic emission.

---

## The Key Empirical Result: Independent Rediscovery of Lorenz Geometry

Lorenz (1963) discovered that the chaotic attractor of a three-variable
ODE has fractal dimension D₂ ≈ 2.06.

We fed **71,161 real typhoon track points** (1951–2023, RSMC Tokyo)
into the 24-cell framework — no equations, no simulation —
and recovered:

```
Lorenz attractor (3D ODE):          D₂    = 1.997  (lit. 2.06)
Northwest Pacific typhoon system (4D real data):  D_eff = 1.991
                                              difference = 0.006
```

Both systems live on a **2-dimensional strange attractor
embedded in higher-dimensional space**.
The 24-cell recovered this without being told what to look for.

---

## What cosD Measures

Given N events in 4D space, map each event to the nearest of the
24 vertices of the 4D regular polytope (24-cell).
Compute the **shape vector** sv ∈ ℝ²⁴ (probability over vertices).
cosD is the cosine distance from sv to the uniform baseline:

```
cosD = 0    →  perfectly uniform (no preferred direction)
cosD = 1    →  all events at one vertex (maximally ordered)
```

**cosD as a surrogate Lyapunov exponent (measured on Lorenz system):**

| ρ value | System state         | cosD  |
|---------|----------------------|-------|
| 1–20    | Fixed point / periodic | 0.63–0.72 |
| 24.00   | Late periodic        | 0.365 |
| **24.50** | **Critical transition** | **0.285 ← sharp drop** |
| 24.74   | Chaos onset          | 0.291 |
| 28–100  | Full chaos           | 0.25–0.30 |

The **phase transition at ρ = 24.74** is detected purely geometrically,
with no prior knowledge of the governing equations.

---

## Why the 24-cell? (and not PCA, t-SNE, or a Transformer)

This is the right question. Here is the honest answer.

**PCA** finds the direction of maximum variance.
It is optimal for Gaussian data, but variance is not the same as
criticality. A system approaching a phase transition does not
necessarily expand — it *reorients*.

**t-SNE / UMAP** are visualization tools. They are not similarity measures
in the mathematical sense; the distance metric they produce is not
interpretable or reproducible across datasets.

**Transformers** learn similarity from labeled data. They require
thousands of labeled examples of "before a crash" or "before a seizure."
Such labels are rare by definition (crises are rare).

**The 24-cell** gives a *coordinate-free, parameter-free, closed-form*
inner product on the 4D unit sphere.
It is the unique regular polytope in 4D that achieves the
theoretical maximum kissing number (24), meaning it samples all
directions on S³ with zero angular bias.

The similarity defined by the 24-cell is:
- **Interpretable**: each vertex has a physical label (e.g., `lon+lat-dep+time-`)
- **Reproducible**: no randomness, no training, no hyperparameters
- **Universal**: the same 24 vertices apply to any 4D domain

This is what we mean by "XAI by geometry":
the model's behavior is readable directly from the vertex labels.

---

## Three Independent Validations

The identical algorithm (zero code change, only axis selection differs)
was validated on three independent physical domains:

### 1. Seismology — Tohoku M9.0 (2011)

Applied to the JMA seismic catalog (2004–2011):

- Pre-event vertex transition pathway: **V20 → V8 → V12/V14 → V4 → V21**
- The V4 → V21 transition recovered the **Kato et al. (2012) foreshock
  migration sequence** (Ibaraki offshore, 2011-03-09)
  with centroid distance **0.52° from actual hypocenter**
- Area scaling law: `log₁₀(area_km²) = 0.932 × M − 5.971`  (R² = 0.9968, n=3)
- FormA false-positive rate: **V21 = 0.000%** (1,589 windows verified)

### 2. Typhoon climatology — PDO × ENSO × Lunar phase

Applied to RSMC Tokyo best-track data (1951–2023, 71,161 points):

- **Lorenz 2-wing symmetry** independently detected:
  Wing A [x=+1, y=0, z=15] → **V0(lon-lat-)** dominant
  Wing B [x=−1, y=0, z=15] → **V3(lon+lat+)** dominant
  V0/V3 occupancy **exactly swap** between wings — geometric detection
  of the Lorenz (x,y,z) → (−x,−y,z) symmetry
- Luzon Strait attractor cluster: **Rayleigh test R = 0.6545, p = 0.000027** (4.3σ)
  — 22 typhoon passages cluster on new moon to first quarter, zero at full moon

### 3. ENSO — 1987 PDO regime shift

Applied to Niño3.4 index (1951–2023):

- El Niño and La Niña both converge to **V22 (dep+time−)**  before 1987
- Both shift to **V23 (dep+time+)** after 1987
- **V22 → V23 transition year = 1987** — coincides exactly
  with the documented PDO warm-to-cool regime shift
- ENSO vs typhoon shape-vector cosine similarity: **0.666**

---

## Alignment Polarity (AP) — a New Indicator

```
AP = −ΔH_norm × tanh(cosD × 5)

AP > 0  :  Order convergence   (earthquake precursor, fracture onset)
AP ≈ 0  :  Edge of Chaos       (where living systems operate)
AP < 0  :  Chaos convergence   (arrhythmia, market crash, turbulence)
```

AP zero-crossings encode the topology of the transition:

| System type            | AP zero-crossings |
|------------------------|-------------------|
| Seismic precursor      | **2** (invariant) |
| Edge-of-chaos (life)   | ≈ 6               |
| Lévy-fractal (Pollock) | ≈ 8               |

---

## Dimensional Scaling Law

Empirically established across 4D (24-cell) and 8D (E₈) experiments:

```
S/N = 0.050 × K^1.308

K = 24      (24-cell, 4D)  →  S/N ≈      3×    proven: WCM
K = 240     (E₈,    8D)   →  S/N ≈     65×    proven: E₈ scan
K = 196560  (Leech, 24D)  →  S/N ≈ 422000×    theoretical ceiling
```

The E₈ root system (240 vectors, 8D) — the object behind
Viazovska's 2022 Fields Medal — provides 10× angular resolution
and ~20× higher S/N than the 24-cell in experiments.

---

## Applications

The same algorithm — only the **choice of 4 axes** changes.

| Domain | Axes (4D) | Crisis detected |
|--------|-----------|-----------------|
| **Seismology** (WCM) | lon, lat, depth, time | Earthquake M6–M9 |
| **Typhoon / Climate** | lon, lat, pressure, time | RI onset, regime shift |
| **Finance** (FIN-CDS) | return, volatility, correlation-break, liquidity | Market crash |
| **Neuroscience** (NEURO-CDS) | AP-position, LR-position, freq-band, phase | Seizure onset |
| **Materials** (MAT-CDS) | x, y, z, AE energy | Fracture |
| **Medicine** | 8 axes selected from 30 cancer features | Malignancy transition |

---

## Architecture

```
Raw events  (N × D)
      ↓
AutoAxisSelector v0.3   ─── selects best D axes
      ↓                     noise-aware: corr / stationarity / outlier
Normalize to [−1, 1]^D
      ↓
Polytope Mapping
  24-cell  (4D, 24 refs)    WCM original — proven
  E₈       (8D, 240 refs)   10× resolution — today's frontier
  Leech Λ₂₄(24D, 196560)   theoretical ceiling — future work
      ↓
Shape Vector  sv[] ∈ ℝᴷ    K-dim probability distribution
      ↓
  cosD          surrogate Lyapunov exponent
  hot_v         dominant vertex (interpretable label)
  H_entropy     directional entropy
  AP            Alignment Polarity  (order vs chaos convergence)
  D_eff         effective fractal dimension (entropy proxy for D_KY)
      ↓
V-transition detection + statistical testing
```

---

## Installation

```bash
git clone https://github.com/watabe-masanori/cds-polytope.git
cd cds-polytope
pip install -r requirements.txt
```

**Requirements:** `numpy scipy matplotlib scikit-learn`

---

## Quick Start

```python
from core.cds_engine import CDSEngineExtended
from core.cds_highdim import CELL24_UNIT
import numpy as np

# --- minimal example: Lorenz attractor ---
from scipy.integrate import solve_ivp

def lorenz(t, u, sigma=10, rho=28, beta=8/3):
    x, y, z = u
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

sol = solve_ivp(lorenz, [0, 200], [1, 1, 1],
                dense_output=False, max_step=0.01,
                t_eval=np.arange(0, 200, 0.01))
xyz = sol.y.T[1000:]   # discard transient

# embed (x, y, z) into 4D by appending time
t_col = sol.t[1000:, None]
data4d = np.hstack([xyz, t_col])

# run CDS
eng = CDSEngineExtended(CELL24_UNIT, name="Lorenz-4D")
result = eng.scan(data4d, window=500, step=100)

for r in result:
    print(f"t={r['t_center']:7.1f}  cosD={r['cosD']:.4f}  "
          f"hot_v={r['hot_v']:02d}  AP={r['AP']:+.4f}")
```

Expected output for ρ=28 (full chaos): `cosD ≈ 0.27`
Expected output for ρ=20 (periodic):   `cosD ≈ 0.68`

---

## Notebooks (Coming)

Three demonstration notebooks are in preparation.
Each is a self-contained experiment that reproduces a key finding:

| Notebook | Finding | Status |
|----------|---------|--------|
| `lorenz_24cell.ipynb` | 2-wing symmetry detection + cosD = Lyapunov surrogate | 🔜 |
| `typhoon_attractor.ipynb` | Fractal dimension D_eff ≈ 2.0 ≈ Lorenz D₂ ≈ 2.06 | 🔜 |
| `enso_typhoon_isomorphism.ipynb` | ENSO ↔ typhoon shape-vector similarity 0.666 | 🔜 |

---

## Future Directions

### Toward Neuromorphic AI: Spiking Networks as 4D Event Clouds

Spiking neural networks (SNNs) generate spatiotemporal event streams —
neuron position, layer depth, spike frequency, and time —
precisely the 4D structure CDS-Polytope is designed to analyze.

**The geometric parallel is striking:**
just as seismic foreshocks converge toward a specific 24-cell vertex
before a rupture, biological neurons exhibit **pre-transition synchrony**
before a seizure, a sleep-stage shift, or a perceptual switch.
cosD, as a parameter-free measure of directional concentration,
may serve as a hardware-efficient synchrony monitor
without requiring explicit computation of cross-correlations or PLV.

**Three hypotheses currently under investigation:**

1. **Seizure onset as FormA convergence.**
   Pre-ictal EEG channels (4D: electrode-x, electrode-y, frequency-band, time)
   may exhibit the same V-transition pathway as seismic precursors.
   Preliminary simulator results (`simulators/neuro_cds_sim.py`) are consistent
   with this hypothesis; validation on the CHB-MIT scalp EEG dataset is in progress.

2. **Neural attractor states and the 24-cell.**
   Sleep/wake transitions, focused/diffuse attention, and anesthesia depth
   are understood as transitions between neural attractor basins.
   The hypothesis: each stable attractor state corresponds to a dominant
   hot_v on the 24-cell, and the transition between states is detectable
   as a hot_v shift — readable directly from the vertex label
   (`freq+phase-` → `freq-phase+`, etc.) without a trained classifier.

3. **Digital–analog hybrid AI at the Edge of Chaos.**
   The AP indicator reveals that living systems operate near AP ≈ 0
   (Edge of Chaos) — neither fully ordered nor fully chaotic.
   Neuromorphic chips (Intel Loihi, IBM TrueNorth, BrainScaleS)
   implement spiking dynamics in mixed analog-digital circuits.
   If cosD reliably tracks the order-chaos boundary in real neural circuits,
   it could serve as a **closed-loop feedback signal** to keep
   hybrid AI hardware operating in the biologically optimal regime —
   dynamically adjusting analog bias currents based on real-time cosD.

> **Honest status note:**
> Hypotheses 1–3 are research directions, not established results.
> The mathematical connection (4D event clouds → 24-cell → cosD)
> is domain-agnostic and the seismological validation is solid.
> Neural application moves from "proven" to "plausible and testable."
> We state this distinction explicitly to invite rigorous collaboration,
> not to overstate the current evidence.

**If you work on neuromorphic computing, BCI, or spiking networks
and find this direction interesting, please reach out.**
dabmasa@gmail.com

---

## Repository Structure

```
cds-polytope/
├── core/
│   ├── cds_engine.py           CDSEngine + CDSEngineExtended (AP, D_eff)
│   ├── cds_highdim.py          E₈ root system (240 refs), 24-cell (24 refs)
│   └── cds_critical.py         CriticalPointAnalyzer (CPS, zero-crossing)
├── selector/
│   └── axis_selector_v3.py     AutoAxisSelector v0.3 (zero-padding strategy)
├── simulators/
│   ├── mat_cds_sim.py          Material fracture AE simulator
│   ├── fin_cds_sim.py          Financial market crash simulator
│   └── neuro_cds_sim.py        Seizure onset simulator
├── examples/
│   ├── mat_cds_demo.py         Material fracture demo
│   ├── real_data_demo.py       Breast cancer + Wine (scikit-learn datasets)
│   └── order_chaos_demo.py     AP polarity — order vs chaos regimes
├── wcm/
│   └── README.md               WCM earthquake application — contact author
├── requirements.txt
├── LICENSE                     GNU AGPL v3
└── README.md                   This file
```

---

## Papers

### Peer-reviewed / Under Review

**Watabe 2026a** — *Effect of geomagnetic storm activity on daily maximum
earthquake magnitude: a statistical analysis of the JMA catalog*
ESSOAr: [10.22541/essoar.15003401/v1](https://doi.org/10.22541/essoar.15003401/v1)

**Watabe 2026b** — *Moho-depth-dependent lunar tidal triggering law for
Japanese inland earthquakes*
ESSOAr: [10.22541/essoar.15003579/v1](https://doi.org/10.22541/essoar.15003579/v1)

**Watabe 2026c** — *Universal V-Transition Pathways in Seismic Source Clouds
Preceding Large Earthquakes: Empirical Scaling Relationships from a
4D Geometric Analysis*
Submitted to **Seismica** (2026-06-18)

### In Preparation

**Watabe & Claude 2026** — *24-Cell Polytope Discretization as an Explainable
4D Similarity Measure: From Seismic Source Clouds to Climate Attractors*
Target: **arXiv cs.LG / cs.AI → Nature Machine Intelligence**

---

## Theoretical Background

### Why the 24-cell is special

The 24-cell is the **unique regular polytope in 4D that is self-dual**
and achieves the theoretical maximum kissing number (24) in four dimensions.
This means its 24 vertices tile S³ with zero angular bias —
it is the ideal "compass" for 4D directional statistics.

The 24 vertices carry the symmetry group F₄ (order 1152),
the Weyl group of the F₄ root system.
Each vertex has a natural axis-aligned label (`lon+lat-dep+time-`, etc.),
making the framework **intrinsically interpretable**.

### cosD as a geometric Lyapunov surrogate

The standard Lyapunov exponent λ requires knowledge of
the governing equations or a long time series for Jacobian estimation.

cosD requires neither. It measures the **angular concentration**
of the event cloud on S³ — which empirically tracks the
order-to-chaos transition with the same sensitivity as λ.

Formally: cosD is the cosine distance between the shape vector
(empirical distribution over 24-cell vertices) and the
uniform distribution (maximum entropy state, cosD=0).

### D_eff as a Kaplan-Yorke proxy

The effective fractal dimension D_eff is defined via
the entropy of the shape vector sv:

```
H     = −∑ sv[k] log sv[k]
H_max = log(K)             (K = number of vertices)
D_eff = D × (H / H_max)    (D = embedding dimension)
```

For D=4, K=24: D_eff ≈ 4 when the cloud is uniformly spread
(no attractor), D_eff < 4 when it concentrates.

Empirically: typhoon 4D data → D_eff = 1.991 ≈ Lorenz D₂ = 2.06.

---

## License

```
CDS-Polytope — Universal Critical-transition Detection Sensor
Copyright (C) 2026  Masanori Watabe

GNU Affero General Public License v3 or later.
```

**Always free:** earthquake hazard mitigation, disaster prevention,
non-commercial scientific research.

**Commercial license required:** financial forecasting, insurance risk modeling,
medical diagnosis, infrastructure monitoring.
Contact: dabmasa@gmail.com

---

## Citation

```bibtex
@software{watabe2026cds,
  author    = {Watabe, Masanori},
  title     = {CDS-Polytope: Universal Critical-transition Detection Sensor
               based on 24-cell and E₈ Polytope Geometry},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/watabe-masanori/cds-polytope},
  license   = {AGPL-3.0}
}
```

---

## Contact

**Masanori Watabe** — Independent Researcher, Sendai, Japan
ORCID: [0009-0000-4441-5126](https://orcid.org/0009-0000-4441-5126)
Email: dabmasa@gmail.com
Research blog: [earthquake.website](https://earthquake.website)

---

*"Lorenz discovered the strange attractor from equations in 1963.*
*We found it in typhoon data in 2026.*
*The 24-cell was the bridge."*

— Masanori Watabe & Claude, 2026
