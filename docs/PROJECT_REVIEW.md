# Project Review

| Area | Assessment | Evidence and caveat |
|---|---:|---|
| Problem framing | 8/10 | Clear cold-start scenario problem. |
| Architecture | 7/10 | Persona cache, fallback, simulator, and CLI are implemented. |
| Validation | 2/10 | No observed demand or backtest is retained. |
| Price modelling | 2/10 | Only 15 price observations are versioned; the saved LightGBM log shows no splits (train R² 0.0), so the model returned one mean price for every product. |
| Reproducibility | 4/10 | Persona caching and `requirements.txt` exist, but reruns are not bit-identical: seeds and market sizes use Python's per-process string hash and NumPy's global generator is unseeded. |

Priority: add historical launch cohorts for backtesting and calibrate scenario
parameters before treating outputs as forecasts.
