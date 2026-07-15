# Formal Verification of Decentralized Satellite Swarms for Safe Orbital Maneuvering

**Empirical Study — Full Reproducible Codebase**



---

## Overview

This repository contains the complete Python simulation framework, formal model fragments,
experimental scripts, and plotting code for the empirical study on formal verification of
decentralized autonomous satellite swarms performing collision avoidance at 550 km LEO.

The framework implements:

- **HCW orbital dynamics engine** — Hill–Clohessy–Wiltshire state transition matrix propagation, reachable-set computation, and an interval-arithmetic safety certificate oracle (SMT proxy)
- **Distributed swarm protocol simulator** — Byzantine-resilient consensus (PBFT-variant), LTL property monitor (P1–P4), configurable communication topology with link dropout
- **Formation initializers** — Projected Circular Orbit (PCO), string-of-pearls, and stacked tetrahedron geometries with Gaussian orbital determination error injection
- **Three experimental scenarios** — 24-sat PCO, 12-sat string-of-pearls with Byzantine fault, 6-sat tetrahedron with 30% link dropout
- **Sensitivity sweeps** — keep-out radius, OD noise, link dropout, oracle sample count, orbital altitude, Byzantine fault fraction
- **Statistical analysis** — bootstrap confidence intervals, KS test, Mann-Whitney U test vs RL baseline
- **10 publication-quality figures** — all generated from raw simulation data

---

## Repository Structure

```
.
├── src/
│   ├── hcw_dynamics.py               # HCW propagator + safety oracle
│   ├── swarm_protocol.py             # Distributed protocol + LTL monitor
│   ├── formation_initializer.py      # Formation geometry generators
│   ├── run_main_experiments.py       # Scenarios 1–3 (200 trials each)
│   ├── run_sensitivity_experiments.py# Sensitivity & robustness sweeps
│   ├── plot_figures.py               # Figures 1–6
│   └── plot_extended_figures.py      # Figures 7–10
├── tests/
│   ├── test_hcw_dynamics.py          # Unit tests: HCW engine (8 tests)
│   └── test_swarm_protocol.py        # Unit tests: swarm protocol (6 tests)
├── models/
│   ├── satellite_swarm.prism         # PRISM model (guarded command language)
│   ├── swarm_properties.props        # PRISM LTL/PCTL property specifications
│   └── hcw_separation.pvs            # NASA PVS separation invariance theorem
├── figures/                          # All 10 generated PNG figures
├── data/                             # Serialised experiment results (JSON)
├── docs/                             # Paper PDF and DOCX
├── results/                          # Raw Monte Carlo output arrays
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run main experiments (Scenarios 1–3)

```bash
python src/run_main_experiments.py
```

Outputs results to `data/results_final.json`. Runtime: ~3–5 min on a modern laptop.

### 3. Run sensitivity sweeps

```bash
python src/run_sensitivity_experiments.py
```

Outputs to `data/ext_results.json`. Runtime: ~2–4 min.

### 4. Generate all figures

```bash
python src/plot_figures.py           # Figures 1–6 → figures/
python src/plot_extended_figures.py  # Figures 7–10 → figures/
```

### 5. Run unit tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Expected: **14 tests pass**, 0 failures.

---

## Key Results Summary

| Scenario | N | f | Dropout | P1 [%] | P3 [%] | P4 [%] | Coverage [%] | Latency [ms] |
|---|---|---|---|---|---|---|---|---|
| S1: PCO-24 | 24 | 0 | 0% | **0.00** | **0.00** | **0.00** | 92.81 | 255.8 |
| S2: String-12 Byzantine | 12 | 1 | 5% | **0.00** | **0.00** | **0.00** | 100.00 | 118.2 |
| S3: Tetrahedron-6 Dropout | 6 | 1 | 30% | **0.00** | **0.00** | **0.00** | 88.10 | 51.4 |

| Method | Formal Coverage | Decision Latency | P1 Violations |
|---|---|---|---|
| **Proposed (LTL+SMT)** | **92.8%** | **255.8 ms** | **0.00%** |
| Centralized MPC | 100.0% | 600,000 ms | 0.00% |
| Distributed RL (PPO) | 0.0% | 850 ms | 4.30% |

---

## Formal Models

### PRISM Model Checker
The `models/satellite_swarm.prism` file contains the guarded-command-language encoding
of the swarm protocol. Verify with:

```bash
# Install PRISM 4.8: https://www.prismmodelchecker.org/
prism models/satellite_swarm.prism models/swarm_properties.props
```

### NASA PVS
The `models/hcw_separation.pvs` file contains the separation invariance theorem.
Requires PVS 7.1 + NASA PVS Library:

```bash
# https://github.com/nasa/pvslib
pvs -batch -l models/hcw_separation.pvs
```

---

## Reproducibility

All experiments use fixed NumPy random seeds. The seed strategy is:
- Main scenarios: `seed = trial_index * 7 + 13`
- Sensitivity sweeps: `seed = trial_index * 31 + 7`
- Statistical analysis: `seed = trial_index * 97 + 5`

Results are deterministic on any platform with NumPy ≥ 1.24 and SciPy ≥ 1.9.

---

## Citation

```bibtex
@article{satellite_swarm_fv_2025,
  title   = {Formal Verification of Decentralized Satellite Swarms
             for Safe Orbital Maneuvering: An Empirical Study},
  author  = {[Author names withheld for review]},
  journal = {IEEE Transactions on Aerospace and Electronic Systems},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT License — see `LICENSE` for details.
