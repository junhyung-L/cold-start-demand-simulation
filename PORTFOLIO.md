# Cold-Start Demand Simulation for New Products

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## At a glance

Persona-based Monte Carlo simulator for 15 new food products without sales history; awareness, trial, repeat, and price elasticity are editable inputs. Built for the 2025 Dongwon x KAIST AI Competition (August to September 2025).

**Problem.** An unreleased product has no target for a conventional forecasting model. The only inputs were product descriptions (ingredients, advertising months, ad model), a category tree, and 15 list prices from Naver Shopping. Every quantity the plan depends on (awareness, trial, repeat, price response, market size) has to be assumed, so the work keeps those assumptions visible instead of hiding them inside a model.

**Result.** In a single run with 3 personas, a 900 g premium tuna-extract seasoning sauce totals 7,207,421 simulated units over 12 months. This is a scenario, not observed sales.

[README](README.md) · [Pipeline](forecast_pipeline.py) · [Final-run notebook](notebooks/02_final_demand_forecast.ipynb)

## Key decisions

**1. Simulate scenarios instead of fitting a forecast.**
Options: borrow sales curves from analogous products (no such data was provided), fit a diffusion model with guessed parameters, or simulate persona-level purchase decisions.
Choice: a persona-based Monte Carlo simulation with explicit parameters, writing the per-month median across runs to the submission.
Why: it fits the competition's focus on personas and generative AI, and every driver (price, discounts, GRPs (gross rating points), distribution ramp, seasonality) is a named input someone can argue with.
Cost: no ground truth; totals scale with hand-set defaults (awareness 0.18, trial 0.35, repeat 0.55, price elasticity -1.2) and a heuristic market size.

**2. LLM personas with a cache and a rule-based fallback.**
Options: LLM (large language model) personas only, rule-based personas only, or LLM personas cached to JSON with a rule-based fallback.
Choice: the third.
Why: missing keys or malformed LLM output should not stop the pipeline, and caching lets repeated runs reuse the same personas.
Cost: the code accepts a short persona list without checking; in the final run Gemini returned 3 personas for this product instead of the 400 requested.

**3. Price input from a model trained on scraped prices.**
Options: use each product's observed price, apply a rule-based unit-price table, or train a LightGBM model on collected prices.
Choice: LightGBM with the rule table as fallback.
Why: to generalize to products without an observed price.
Cost: with 12 training rows the model found no split (train R² 0.0000, test R² -0.1779) and returned the training mean, KRW 14,093, for every product. The observed price or the rule table would have kept price differences between products.

## Results

| Single run, 3 personas (premium seasoning sauce 900 g) | Value |
|---|---:|
| Month 1 simulated units | 123,287 |
| Month 12 simulated units | 1,321,951 |
| 12-month total | 7,207,421 |
| Price input | KRW 14,093 |

Growth comes from the assumed distribution ramp (35% to 100% coverage) and repeat purchases that scale with cumulative trial units.

## Limitations

- Simulation, not forecast: no observed sales, backtest, baseline, or competition score, so the result says nothing about forecast accuracy or business impact.
- Only 3 personas drive this scenario, reused across all seven Monte Carlo runs.
- Price does not differentiate products: the price model is constant at KRW 14,093, so the result is not a price-elasticity estimate.
- No sensitivity sweep, so which input moves the total most is unmeasured.

## Next steps

- Validate LLM output (persona count and schema) before caching, and use observed prices directly.
- Run one-at-a-time sweeps over awareness, trial, repeat, price elasticity, and distribution to show which assumption moves the 12-month total.
- With historical launch cohorts, calibrate the defaults and backtest.

## Evidence

- [`forecast_pipeline.py`](forecast_pipeline.py)
- [`notebooks/02_final_demand_forecast.ipynb`](notebooks/02_final_demand_forecast.ipynb) (price-model fit and persona counts)
- [`results/solution_report.md`](results/solution_report.md) and [`results/monthly_forecast.csv`](results/monthly_forecast.csv) (scenario values)
