# Cold-Start Demand Simulation for New Products

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## At a glance

Persona-based Monte Carlo simulator for 15 new food products without sales history; awareness, trial, repeat, and price elasticity are editable inputs. Built for the 2025 Dongwon x KAIST AI Competition (August to September 2025).

**Problem.** An unreleased product has no target for a conventional forecasting model. The only inputs were product descriptions (ingredients, advertising months, ad model), a category tree, and 15 list prices from Naver Shopping. Every quantity the plan depends on (awareness, trial, repeat, price response, market size) has to be assumed, so the work keeps those assumptions visible instead of hiding them inside a model.

**Result.** With seed 42, 400 rule-based personas, and observed list prices, a 900 g premium tuna-extract seasoning sauce totals 3,070,766 simulated units over 12 months (per-month median of 7 Monte Carlo runs), and the same seed reproduces the file exactly. The competition run (3 Gemini personas, one price for every product) gave 7,207,421 in a single Monte Carlo run. These are scenarios, not observed sales.

[README](README.md) · [Pipeline](forecast_pipeline.py) · [Final-run notebook](notebooks/02_final_demand_forecast.ipynb)

## Key decisions

**1. Simulate scenarios instead of fitting a forecast.**
Options: borrow sales curves from analogous products (no such data was provided), fit a diffusion model with guessed parameters, or simulate persona-level purchase decisions.
Choice: a persona-based Monte Carlo simulation with explicit parameters, writing the per-month median across runs to the submission.
Why: it fits the competition's focus on personas and generative AI, and every driver (price, discounts, GRPs (gross rating points), distribution ramp, seasonality) is a named input someone can argue with.
Cost: no ground truth; totals scale with hand-set defaults (awareness 0.18, trial 0.35, repeat 0.55, price elasticity -1.2) and a heuristic market size.

**2. LLM personas with a cache and a rule-based fallback.**
Options: LLM (large language model) personas only, rule-based personas only, or LLM personas cached to JSON with a rule-based fallback.
Choice: the third, with a count check.
Why: missing keys or malformed LLM output should not stop the pipeline, and caching lets repeated runs reuse the same personas.
Cost: in the competition run Gemini returned 3 personas instead of the 400 requested and the code used them without notice. The pipeline now logs the shortfall and records persona counts per product in `results/run_summary.csv`; the rule-based path always produces the requested number.

**3. Observed price first, model only if it earns its place.**
Options: use each product's observed price, a category median, a rule-based unit-price table, or a LightGBM model trained on collected prices.
Choice: observed price, then category median for unseen products; LightGBM only if it beats the median on leave-one-out.
Why: the competition-time LightGBM, trained on 12 rows, found no split (train R² 0.0000, test R² -0.1779) and returned KRW 14,093 for every product. On leave-one-out over the 15 prices, the category median has MAE KRW 4,920 against 7,722 and 9,474 for two LightGBM settings, so no model is saved.
Cost: observed prices are one listing each and do not track pack size (90 g at KRW 19,050, 135 g at KRW 16,275).

**4. Seeded, process-independent randomness.**
Options: Python's `hash()` and NumPy's global RNG, or an explicit seeded generator.
Choice: `--seed` (default 42) plus a CRC32 hash of the product name, passed as a NumPy generator to every random draw.
Why: `hash()` changes with `PYTHONHASHSEED`, so the same command gave different market sizes and seeds across processes.
Cost: none observed; two seed-42 runs, one under `PYTHONHASHSEED=123`, wrote identical submissions.

## Results

`results/submission.csv`, seed 42, 400 rule-based personas, 7 Monte Carlo runs:

| Premium seasoning sauce 900 g | Value |
|---|---:|
| Month 1 simulated units | 54,580 |
| Month 12 simulated units | 471,831 |
| 12-month total | 3,070,766 |
| Price input (observed) | KRW 22,400 |

Across the 15 products, 12-month totals range from 1,941,646 to 7,918,193. Growth comes from the assumed distribution ramp (35% to 100% coverage) and repeat purchases that scale with cumulative trial units. The competition-run files are kept in `results/competition_2025/`.

## Limitations

- Simulation, not forecast: no observed sales, backtest, baseline, or competition score, so the result says nothing about forecast accuracy or business impact.
- Observed prices carry listing noise, so the price effect is not a price-elasticity estimate.
- Rule-based personas are random draws, not fitted to survey or panel data.
- No sensitivity sweep, so which input moves the total most is unmeasured.

## Next steps

- Run one-at-a-time sweeps over awareness, trial, repeat, price elasticity, and distribution to show which assumption moves the 12-month total.
- Collect several price listings per product and use their median.
- With historical launch cohorts, calibrate the defaults and backtest.

## Evidence

- [`forecast_pipeline.py`](forecast_pipeline.py)
- [`results/submission.csv`](results/submission.csv) and [`results/run_summary.csv`](results/run_summary.csv) (seed-42 scenario, prices, persona counts)
- [`results/competition_2025/`](results/competition_2025/) and [`notebooks/02_final_demand_forecast.ipynb`](notebooks/02_final_demand_forecast.ipynb) (competition run, price-model fit, persona counts)
