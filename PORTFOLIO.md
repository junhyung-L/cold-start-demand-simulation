# Cold-Start Demand Simulation: Planning When Sales History Does Not Exist

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## Summary

For the 2025 Dongwon x KAIST AI Competition (August to September 2025), this
project produces 12-month demand scenarios for 15 new food products with no
sales history. Instead of presenting a synthetic forecast as observed truth,
it builds assumption-driven scenarios: synthetic consumer personas make
monthly purchase decisions under price, promotion, advertising, distribution,
and seasonality inputs, repeated as a Monte Carlo simulation. One stored run
for a 900 g premium seasoning sauce totals 7,207,421 simulated units over 12
months. That is a scenario output, not observed sales or forecast accuracy.

## Challenge

An unreleased product has no target for a conventional forecasting model.
The only data in hand were product descriptions (ingredients, advertising
months, ad model), a category tree, and 15 list prices from Naver Shopping.
Every quantity the plan depends on (awareness, trial, repeat, price response,
market size) therefore has to be assumed, and the work has to keep those
assumptions visible rather than hide them inside a model.

## Decisions

**1. Simulate scenarios instead of fitting a forecast.**
Options: borrow sales curves from analogous products (no such data was
provided), fit a diffusion model with guessed parameters, or simulate
persona-level purchase decisions. Choice: a persona-based Monte Carlo
simulation whose parameters are explicit inputs, with the per-month median
across runs written to the submission. Why: it fits the competition's focus
on personas and generative AI, and every driver (price, discounts, GRPs,
distribution ramp, seasonality) is a named input someone can argue with.
Cost: there is no ground truth, and the totals scale with hand-set defaults
(awareness 0.18, trial 0.35, repeat 0.55, price elasticity -1.2) and a
heuristic market size.

**2. LLM personas with a cache and a rule-based fallback.**
Options: LLM-only personas, rule-only personas, or LLM personas cached to
JSON with a rule-based fallback. Choice: the third. Why: missing keys or
malformed LLM output should not stop the pipeline, and caching lets repeated
runs reuse the same personas. Cost: the fallback is not bit-reproducible
(seeds come from Python's per-process string hash and NumPy's global generator
is not seeded), and the code accepted a short persona list without checking.
In the final run the Gemini call returned 3 personas for this product instead
of the 400 requested.

**3. Price input from a model trained on scraped prices.**
Options: use each product's observed price directly, apply a rule-based
unit-price table, or train a LightGBM model on collected prices. Choice:
LightGBM with the rule table as fallback. Why: it was meant to generalise to
products without an observed price. Cost: with 12 training rows the model
found no split (train R² 0.0, test R² -0.18 in the notebook log), so it
returned the training mean, KRW 14,093, for every product. Using the observed
price or the rule table would have kept price differences between products.

## Results

| Retained run (premium seasoning sauce 900 g) | Value | Source |
|---|---:|---|
| Month 1 simulated units | 123,287 | `results/monthly_forecast.csv` |
| Month 12 simulated units | 1,321,951 | `results/monthly_forecast.csv` |
| 12-month total | 7,207,421 | `results/solution_report.md` |
| Price input | KRW 14,093 | `results/solution_report.md` |

These are the last single Monte Carlo run of the last product. The growth
comes from the assumed distribution ramp (35% to 100% coverage) and repeat
purchases that scale with cumulative trial units. There is no baseline or
competition score to compare against.

## What the result does not show

- Not observed sales, forecast accuracy, or business impact. No launch data or
  backtest exists.
- Not a price-elasticity estimate: only 15 price observations are retained and
  the price model was constant.
- Not a sensitivity analysis: no sweep over the assumptions is retained, so
  which input moves the total most is unmeasured.

## What I would do next

Validate the LLM output (persona count and schema) before caching, seed a
local NumPy generator and replace the built-in string hash with a stable one,
use observed prices directly, and run one-at-a-time sweeps over awareness,
trial, repeat, price elasticity, and distribution to show which assumption
moves the 12-month total. With historical launch cohorts, calibrate the
defaults and backtest.

## Evidence

- [`forecast_pipeline.py`](forecast_pipeline.py)
- [`notebooks/02_final_demand_forecast.ipynb`](notebooks/02_final_demand_forecast.ipynb)
- [`results/solution_report.md`](results/solution_report.md)
- [`docs/PROJECT_REVIEW.md`](docs/PROJECT_REVIEW.md)
