# Project Review

| Area | Assessment | Evidence and caveat |
|---|---:|---|
| Problem framing | 8/10 | Clear cold-start scenario problem. |
| Architecture | 7/10 | Persona cache, fallback, simulator, and CLI are implemented. |
| Validation | 2/10 | No observed demand or backtest is retained. |
| Price modelling | 2/10 | Only 15 price observations are versioned. |
| Reproducibility | 5/10 | Seeded fallback and cache exist; APIs and dependency versions remain external. |

Priority: add historical launch cohorts for backtesting and calibrate scenario
parameters before treating outputs as forecasts.
