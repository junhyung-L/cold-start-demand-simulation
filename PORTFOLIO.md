# Cold-Start Demand Simulation: Planning When Sales History Does Not Exist

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## Overview

For an unreleased product, there is no sales history for a conventional
forecasting target. This project therefore builds assumption-driven demand
scenarios rather than presenting a synthetic forecast as observed truth.

## Approach

The pipeline combines product metadata, price, promotion, advertising,
distribution, seasonality, and synthetic consumer personas in a Monte Carlo
simulation. Gemini-generated personas are cached when credentials are present;
a seeded rule-based fallback keeps the same path runnable without an API key.

## Retained scenario

For a seasoning-sauce concept priced at KRW 14,093 per 900g with 400 personas,
the stored run totals 7,207,421 simulated units across 12 months. It starts at
123,287 units and ends at 1,321,951. These are scenario outputs, not observed
sales, forecast accuracy, or business impact.

## What this demonstrates

The work makes uncertainty explicit through assumptions and simulation inputs,
while keeping deterministic fallback and generated artifacts available for review.

## Limitations

Only 15 price observations are retained and no launch data or backtest exists.
Historical product cohorts are needed to calibrate assumptions before planning use.

## Evidence

- [`forecast_pipeline.py`](forecast_pipeline.py)
- [`results/solution_report.md`](results/solution_report.md)
- [`docs/PROJECT_REVIEW.md`](docs/PROJECT_REVIEW.md)

## Keeping uncertainty visible in the inputs

The primary input is not a known sales target but a bundle of assumptions: product concept and size, observed competitor prices, promotion, advertising, and distribution conditions. Personas repeatedly make monthly purchase decisions. An LLM can enrich the persona text and attributes, while a seeded rule-based fallback keeps the computational path executable when an API key is unavailable. This separates generative variation from the reproducible simulation mechanics.

The retained run uses 400 personas, a 900 g seasoning-sauce concept, and a price of KRW 14,093 over 12 months. Its total of 7,207,421 units, launch-month 123,287 units, and final-month 1,321,951 units are outputs of that scenario—not observed sales. With only 15 price observations, the project does not establish a general price elasticity either.

The value of this design is to expose which assumptions shift a pre-launch plan, not to certify a launch forecast. Once sales, repeat purchase, and inventory data exist, they should be used to calibrate the assumptions and produce explicit price, promotion, and distribution sensitivity ranges.
