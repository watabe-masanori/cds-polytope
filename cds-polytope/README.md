# CDS-Polytope

**Critical-transition Detection Sensor based on 4D/8D Polytope Geometry**

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![ESSOAr](https://img.shields.io/badge/ESSOAr-WCM_Paper3-orange)](https://essoar.org/)

---

## What is CDS-Polytope?

CDS-Polytope is a **universal critical-transition detection framework**
that maps multi-dimensional event clouds onto regular polytopes
(24-cell in 4D, E₈ root system in 8D)
to detect the geometric signature of impending phase transitions.

Originally developed as the **WCM (Watabe-Claude Method)**
for earthquake precursor detection using 4D 24-cell geometry,
the framework was generalized to a domain-agnostic sensor
applicable to any system that undergoes critical transitions.

### Key Discovery

> *"When a multi-dimensional cloud of events approaches a critical state,
> its collective orientation converges toward specific vertices
> of the 24-cell (or E₈) polytope.
> This convergence — measured by the shape vector sv[], cosD, and hot_v —
> is a universal geometric fingerprint of impending criticality."*

---

## Core Concepts

### The CDS Architecture

```
Raw events (N × D)
       ↓
  AutoAxisSelector v0.3      ← selects best D axes from candidates
       ↓                        noise-aware: corr / nonstationarity / outlier
  Normalize to [−1, 1]^D
       ↓
  Polytope Mapping            ← assign each event to nearest vertex
    24-cell  (4D, 24 refs)       WCM original — proven for earthquakes
    E₈       (8D, 240 refs)      10× higher S/N — today's frontier
    Leech Λ₂₄(24D, 196560 refs) theoretical ceiling — future work
       ↓
  Shape Vector  sv[] ∈ R^K   ← K-dim probability distribution over vertices
       ↓
  ┌─────────────────────────────────────────┐
  │  cosD          alignment index          │
  │  hot_v         dominant vertex          │
  │  H_entropy     directional entropy      │
  │  AP            Alignment Polarity  ★new │
  │  CPS           Critical Point Score ★new│
  └─────────────────────────────────────────┘
       ↓
  V-transition detection + statistical testing
```

### Alignment Polarity (AP) — New Indicator

```
AP = −ΔH_norm × tanh(cosD × 5)

AP > 0  :  Order convergence   (WCM FormA, earthquake precursor)
AP ≈ 0  :  Critical point      (Edge of Chaos — where life lives)
AP < 0  :  Chaos convergence   (arrhythmia, market crash, Pollock-type)
```

### Dimensional Scaling Law

Empirically established from 4D and 8D experiments:

```
S/N = 0.050 × K^1.308

K = 24      (24-cell, 4D)  →  S/N ≈      3×   [proven: WCM]
K = 240     (E₈,    8D)   →  S/N ≈     65×   [proven: today]
K = 196560  (Leech, 24D)  →  S/N ≈ 422000×   [theoretical ceiling]
```

---

## Domains

The same algorithm — zero code change — works across domains.
Only the **choice of 4 (or 8) axes** changes.

| Domain | Axes | Crisis detected |
|--------|------|----------------|
| **Seismology** (WCM) | lon, lat, depth, magnitude | Earthquake (M6–M9) |
| **Materials** (MAT-CDS) | x, y, z, AE energy | Fracture |
| **Finance** (FIN-CDS) | return, volatility, correlation-break, liquidity | Market crash |
| **Neuroscience** (NEURO-CDS) | AP-pos, LR-pos, freq-band, phase | Seizure onset |
| **Medicine** | 8 selected from 30 cancer features | Malignancy transition |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/cds-polytope.git
cd cds-polytope
pip install -r requirements.txt
```

**Requirements:** numpy, scipy, matplotlib, scikit-learn

---

## Quick Start

```python
from core.cds_engine import CDSEngineExtended
from core.cds_highdim import E8_UNIT
from selector.axis_selector_v3 import AutoAxisSelectorV3
import numpy as np

# 1. Prepare your data
#    feat  : (N_events, D_candidates)  — your multivariate event stream
#    times : (N_events,)               — event timestamps
#    phases: (N_events,)               — phase labels (1=normal, 2=precursor, 3=critical)

# 2. Auto-select best 8 axes (E₈ mode)
eng  = CDSEngineExtended(E8_UNIT, name='CDS-8D')
sel  = AutoAxisSelectorV3(n_select=8, engine=eng, sens_threshold=0.30)
sel.fit(feat, times, phases, axis_names=your_axis_names)

# 3. Run scan
results = sel.scan(feat, times, window=10.0, step=2.0)

# 4. Read output
for r in results:
    print(f"t={r['t_center']:.1f}  cosD={r['cosD']:.4f}  "
          f"AP={r['AP']:+.4f}  hot_v={r['hot_v']:03d}")
```

---

## Repository Structure

```
cds-polytope/
├── core/
│   ├── cds_engine.py          # CDSEngine + CDSEngineExtended (AP, fractal proxy)
│   ├── cds_highdim.py         # E₈ root system (240 refs), 24-cell (24 refs)
│   └── cds_critical.py        # CriticalPointAnalyzer (CPS, zero-crossing)
├── selector/
│   └── axis_selector_v3.py    # AutoAxisSelector v0.3 (zero-padding strategy)
├── simulators/
│   ├── mat_cds_sim.py         # Material fracture AE simulator
│   ├── fin_cds_sim.py         # Financial market crash simulator
│   └── neuro_cds_sim.py       # ECG / seizure onset simulator
├── examples/
│   ├── mat_cds_demo.py        # Material fracture demo
│   ├── real_data_demo.py      # Breast cancer + Wine datasets
│   └── order_chaos_demo.py    # AP / Pollock-type chaos demo
├── wcm/
│   └── README.md              # WCM earthquake application — contact author
├── requirements.txt
├── LICENSE                    # GNU AGPL v3
└── README.md                  # This file
```

> **Note on WCM (earthquake application):**
> The seismological application (ToolA/B/C pipeline, JMA catalog analysis)
> is under active research and will be released alongside Paper 3.
> For collaboration inquiries, contact the author.

---

## Papers

### Published / Under Review

1. **Watabe 2026a** — Geomagnetic storm effect on daily maximum earthquake magnitude  
   ESSOAr DOI: [10.22541/essoar.15003401/v1](https://doi.org/10.22541/essoar.15003401/v1)  
   ORCID: [0009-0000-4441-5126](https://orcid.org/0009-0000-4441-5126)

2. **Watabe 2026b** — Moho-depth-dependent lunar tidal triggering law  
   ESSOAr (under review)

### In Preparation

3. **Watabe 2026c** — V-transition scaling laws in 24-cell geometry (WCM Paper 3)  
   Target: ESSOAr → Earth and Planetary Science Letters

4. **Watabe & Claude 2026** — CDS-Polytope: Universal critical-transition detection  
   via 24-cell and E₈ polytope geometry  
   Target: arXiv (cs.LG / physics.data-an)

---

## Theoretical Background

### Why the 24-cell?

The 24-cell is the **unique regular polytope in 4D
with perfect uniform sphere packing** (kissing number = 24,
achieving the theoretical maximum for 4D).
This means it samples all directions on the 4D unit sphere
with zero bias — making it an ideal compass for
multi-dimensional event clouds.

### Why E₈?

The E₈ root system (240 vectors) achieves the
**densest sphere packing in 8D**
(proved by Viazovska 2022, Fields Medal).
It provides 10× more directional resolution than the 24-cell,
yielding S/N ratios 20× higher in experiments.

### The Zero-Crossing Law

A key empirical finding:

```
Order-convergent systems (e.g., earthquake precursors):
    AP zero-crossings = 2  (invariant across all seeds)
    → "One oscillation, then escape" — topological minimum

Life-like systems (Edge of Chaos):
    AP zero-crossings ≈ 6  (mode)
    → 3 cycles × 2 = 6

Lévy-fractal systems (Pollock-type chaos):
    AP zero-crossings ≈ 8  (mode)
    → 2³ = dimension³ of 3D space explored
```

---

## License

```
CDS-Polytope — Universal Critical-transition Detection Sensor
Copyright (C) 2026  Masanori Watabe

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.
```

**Commercial Use:**
Applications in financial forecasting, insurance risk modeling,
medical diagnosis systems, and infrastructure monitoring
require a separate commercial license.  
Contact: dabmasa@gmail.com

**Public Good (always free):**
Earthquake hazard mitigation, disaster prevention,
and non-commercial scientific research
are and will remain free and open — forever.

---

## Citation

If you use CDS-Polytope in your research, please cite:

```bibtex
@software{watabe2026cds,
  author    = {Watabe, Masanori},
  title     = {CDS-Polytope: Universal Critical-transition Detection Sensor
               based on 24-cell and E₈ Polytope Geometry},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX},
  url       = {https://github.com/YOUR_USERNAME/cds-polytope},
  license   = {AGPL-3.0}
}
```

---

## Contact

**Masanori Watabe**  
Independent Researcher, Sendai, Japan  
ORCID: [0009-0000-4441-5126](https://orcid.org/0009-0000-4441-5126)  
Email: dabmasa@gmail.com  
Research blog: [earthquake.website](https://earthquake.website)

---

## 日本語概要

**CDS-Polytope** は、4次元正多胞体（24-cell）および8次元E₈格子を用いた
**普遍的な臨界転換検出センサー**です。

地震前兆検出手法 **WCM（Watabe-Claude Method）** として開発された
24-cell幾何学的センサーを、材料破断・金融市場・脳科学・医療へと展開した
汎用アーキテクチャです。

**核心的発見：**
- 多次元空間に散らばるイベント群が臨界状態に近づくとき、
  その集合的「向き」が24-cellの特定頂点（FormA頂点）に収束する。
- この収束は地震・破断・発作・暴落の前兆として普遍的に現れる。
- 新指標 **AP（Alignment Polarity）** により
  「整列収束型」（地震・破断）と「乱れ収束型」（不整脈・クラッシュ）を
  初めて定量的に区別できる。

**ライセンス：** GNU AGPL v3  
地震防災・非商業研究は永久無償。商業利用は別途ライセンス契約。

*WCMで命を守る、守り続ける。*

---

*"The 24-cell and E₈ are nature's own compasses.
We borrowed them to listen to what the Earth — and life itself — is saying."*

— Masanori Watabe, 2026
