# Architecture

`forecast_pipeline.py` is the end-to-end entry point: it selects cached, LLM,
or rule-based personas; optionally trains/loads a LightGBM price model; then
runs the demand simulator and writes artifacts. `simulation_pipeline.py` is a
separate simplified example and not the maintained scenario engine.

`project_config.py` keeps generated forecasts, prompts, source records, the
price model, and calibrated parameters under `results/`; persona caches use
the configured `persona_cache/` directory. The CLI passes its cache-path
option through to the scenario runner.
