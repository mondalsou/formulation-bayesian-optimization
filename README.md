# Formulation Bayesian Optimization

## The pharmaceutical problem
Oral solid dose formulation development is usually constrained by lab throughput, material cost, and formulation complexity. Classical one-factor-at-a-time (OFAT) and brute-force factorial screening are often too expensive when multiple excipients interact nonlinearly.

This project demonstrates a model-based Design of Experiments workflow using Bayesian Optimization (BO) to identify high-performing tablet formulations with fewer experiments.

## Why Bayesian Optimization instead of classical DoE
- Classical DoE is strong for linear/quadratic effects but can be sample-inefficient in nonlinear, constrained spaces.
- BO fits a Gaussian Process surrogate with uncertainty estimates and uses an acquisition function to recommend the next best experiment.
- The BO loop systematically balances exploration (high uncertainty) and exploitation (high expected performance).

## Formulation setup
- Fixed API: `30% w/w`
- Excipients (must sum to `70% w/w`):
  - `HPMC`: `0-20%`
  - `MCC`: `20-60%`
  - `CCS`: `1-8%`
  - `MgSt`: `0.25-2%`
  - `PVP K30`: derived from mass balance and constrained to `0-10%`
- Primary objective: maximize `Q45` dissolution (% released at 45 min)

## Notebook workflow
- `notebooks/01_doe_baseline.ipynb`
  - Synthetic physics-informed simulator
  - Full-factorial feasibility check
  - D-optimal proxy and CCD-style baseline designs
  - Quadratic RSM baseline and response-surface slice plots
- `notebooks/02_bayesian_optimization.ipynb`
  - Single-objective BO with `SingleTaskGP + LogExpectedImprovement`
  - 10-point Sobol initialization + 30 BO steps
  - Convergence, posterior uncertainty, and acquisition landscape plots
- `notebooks/03_comparison.ipynb`
  - 10-trial benchmark with equal 40-evaluation budget
  - BO vs random search vs budgeted grid search vs RSM-guided search
  - Mean convergence with confidence bands and threshold hit-time summary
- `notebooks/04_multiobj_bo.ipynb`
  - Stretch objective: optimize dissolution, hardness, and friability jointly
  - qNEHVI-based multi-objective BO and Pareto front visualization

## Streamlit pipeline
- `app/streamlit_app.py`
  - Design Space Explorer: predicted Q45 heatmap + uncertainty map
  - Run BO: one-click “Suggest Next Experiment” and auto-log
  - Experiment Log: all runs with manual experiment entry
  - Convergence: live best-so-far curve
  - Compare: BO vs random search with confidence intervals

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Then run notebooks in order from `01` to `04`.

Run Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

## Scientific assumptions and limitations
- Response data is synthetic (physics-informed simulator), not real lab data.
- Noise model is Gaussian and homoscedastic for simplicity.
- The 5-variable design space is moderately sized; real programs may include process parameters, categorical variables, and additional CQAs.

## Relevance to CMC/QbD development
- BO loop = model-based DoE under constrained formulation space.
- GP uncertainty = design space characterization and confidence-aware decisions.
- Acquisition-driven experiment selection = systematic prioritization of costly wet-lab runs.
- Pareto front = explicit multi-CQA trade-off analysis (e.g., dissolution vs hardness vs friability).
