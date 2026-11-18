# Architecture

`forecast_pipeline.py` is the end-to-end entry point: it selects LLM (cached)
or rule-based personas and checks their count; resolves each product's price
(given price, observed price, a LightGBM model only if it beat the category
median on leave-one-out, category median, rule table); then runs the demand
simulator with a NumPy generator seeded from `--seed` and a CRC32 hash of the
product name, and writes artifacts. `legacy/simulation_pipeline.py` is
an early simplified example and not the maintained scenario engine.

`project_config.py` keeps generated forecasts, prompts, source records, the
price model, and calibrated parameters under `results/`; persona caches use
the configured `persona_cache/` directory. The CLI passes its cache-path
option through to the scenario runner.
