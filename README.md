# Hourly Bike Rental Demand Forecasting

> Capital Bikeshare should fund this MLOps forecasting system because gated releases, drift monitoring and automatic fallback keep tomorrow's crew plans reliable, unlike one-off notebooks.

## Quick start
Setup, tests and the pipeline run: [section 2](#2-setup-test-and-run). Results and evidence: [section 3](#3-results-and-evidence).


## 1. Business case

### Context
Client: Capital Bikeshare, Operations Planning and Fleet Management business unit. This is a hypothetical scenario built on the public UCI Bike Sharing Dataset (`hour.csv`). It does not describe the company's actual technology, processes, organisation or forecasting capabilities.

Maturity:
- Data: Medium. 17,379 consistently structured hourly observations (2011 to 2012) with rental counts, weather, calendar and seasonal variables, and no reported missing values. The client's real data infrastructure and governance are not verified.
- Technology: Low (assumed). We start from CSV data and build the ML infrastructure in this project.
- Process: Low (assumed). Automated monitoring and retraining do not exist in this scenario.
- People: Medium (assumed). Operations and technical specialists collaborate.

Ownership: Operations Planning owns the business use of the predictions. Data Engineering / ML Engineering owns the technical system (deployment, monitoring, retraining).

### Problem and baseline
The Operations Planning and Fleet Management team struggles with potential bicycle shortages in high-demand periods because demand varies with weather, time of day, weekday and season. Too many bicycles in low-demand locations also create unnecessary rebalancing costs.

The team is assumed to plan from historical reports and operational experience. This process is not verified as Capital Bikeshare's real procedure.

Baseline: two simple rules, scored with `weighted_mae` in rentals per hour on the chronological test period (2012-10-01 to 2012-12-31), source: `hour.csv`.
- Weekday by hour average (uses only data before the test period): 134.98
- Same hour last week: 104.72 (the better simple rule)

Crew KPI baselines: To be measured (client rosters and payroll records).

### Goal and success metrics
The Operations Manager uses the system-wide hourly forecast to decide how many crew hours to schedule per shift, the evening before.

Business KPI: idle crew hours per week, guardrail: overtime and short-notice call-in crew hours per week must not increase, baselines To be measured, source: the client's rosters and payroll records.

ML metric: `weighted_mae` (under-forecast weighted 2x, assumption to validate) of at most 0.85 x the better simple rule on unseen data, because under-forecasting causes overtime and call-ins, which cost more than idle hours. Release decision checked at weights 1.5, 2 and 3.

| Type | Metric (exact name) | Baseline | Target or threshold |
| --- | --- | --- | --- |
| Business KPI | Idle crew hours per week | To be measured | Lower (target to agree with client) |
| Guardrail KPI | Overtime and short-notice call-in crew hours per week | To be measured | Must not increase |
| ML metric | `weighted_mae` (rentals per hour, 2x under-forecast, assumption) | 104.72 (same hour last week, test period) | At most 0.85 x the better simple rule, checked at weights 1.5, 2, 3 |
| Limit | Nightly batch completion | n/a | Run at 18:00, ready by 18:30 (to validate with client) |
| Limit | Inference cost | n/a | Under EUR 0.01 per prediction (initial estimate) |

Release criteria: no promotion to production unless the model meets the `weighted_mae` gate, the batch completes on time, and data quality and leakage checks (with the 17:00 cutoff) pass.

### Solution
The Operations Manager receives a system-wide forecast of total rentals for each hour of tomorrow to decide how many crew hours to schedule per shift. The Operations Manager approves the final roster.

The model predicts `cnt` for each hour of tomorrow from calendar variables and rentals up to 17:00 today, as a nightly batch run at 18:00 and ready by 18:30 (times to validate with the client).

If the batch fails, outputs are invalid or input data is older than the cutoff: same hour last week; if that data is also missing: the stored weekday by hour average.

Scope: the first version forecasts system-wide hourly demand. `hour.csv` contains no station identifier, so station-level decisions are out of scope.

### Data
Source: UCI Bike Sharing Dataset (`hour.csv`), created by Hadi Fanaee-T and distributed through the UCI Machine Learning Repository. The rental data originates from Capital Bikeshare. Licence: CC BY 4.0.

Size: 17,379 recorded hours, January 2011 to December 2012; 165 missing hours inserted as zero rentals and flagged (assumption: no rentals recorded).

Label: `cnt`, the hourly total of casual and registered rentals, known only after the hour ends.

Refresh: the UCI file is a fixed snapshot. In the scenario, new data enters at the nightly 18:00 batch, and a Data Engineering team owns ingestion, validation and versioning.

Inputs: calendar variables, `lag_48`, `lag_168`, the 7-day mean ending at the 17:00 cutoff, same hour yesterday (only for hours before 17:00) and the last hour before the cutoff (16:00). Weather, `casual`, `registered` and `is_imputed` are excluded.

Personal data: none. The data is aggregated hourly counts with no customer identifiers. Production access would still need security and governance controls.

### Value, cost and risks
Benefit over a non-AI approach: at least 15% lower `weighted_mae` than the better simple rule (same hour last week). Effects on idle crew hours and overtime must be validated in the pilot with client rosters.

Rough cost (illustrative estimates, excluding salaries and existing infrastructure):
- People: one Data Scientist / ML Engineer full time, one Data Engineer part time, one Operations representative part time
- Infrastructure: EUR 20 to 100 per month
- Time: 6 to 8 weeks for development, testing and first release, including pilot preparation. The six-month pilot is separate.

Scales to: station-level forecasting once station data exists, cutoff-based features for each forecast hour, electric scooter fleets, public transport capacity planning, and other bike-sharing operators or cities, reusing the MLOps architecture with new data and validation.

Limitation: training data covers 2011 to 2012, so recent operational data is needed before production.

| Risk | Mitigation |
| --- | --- |
| Data quality issues, missing or inconsistent records | Automated schema validation, missing-value checks, duplicate detection |
| Understaffing from under-forecasts | Under-forecasts weighted 2x in training and gate; guardrail KPI on overtime and call-in hours; Operations Manager approves the roster |
| Drift from changing demand | If 4-week `weighted_mae` is no longer 15% better than same hour last week, switch to fallback, retrain on recent data, release only through the gate |
| Batch not ready by 18:30 or input data older than the cutoff | Freshness check, then same hour last week, then stored weekday by hour average |
| Unauthorised access or misuse | Authentication, role-based access control, logging of roster decisions |

### Open gaps
- **Quality gate not met:** the model is 7 % better than same hour last week on validation (ratio 0.927), not the 15 % the gate requires (section 3). Next step: test a level-relative target, retraining on the most recent months, or a shorter validation window.
- **Gate reported, not enforced:** `main` logs whether the gate is met but always saves the model and writes the forecast; the fallback (same hour last week, then the weekday by hour average) and the input freshness check are not built yet.
- Problem statement still describes station shortages and rebalancing; rewrite it in the team's words to match the crew scheduling decision.
- Idle crew hour and overtime baselines: To be measured (need client rosters and payroll records); the idle hours target is still to be agreed with the client.
- The 2x under-forecast weight and the 15% threshold are assumptions to validate with the client (the sensitivity check at 1.5 and 3 shows the decision does not depend on the weight).
- Batch times (18:00 run, 18:30 ready) are assumptions to validate with the client's planning routine; the nightly schedule itself is not set up.
- The 4-week drift window is a starting choice, not yet tested.
- Cost per prediction can only be measured in the deployed system.

## 2. Setup, test and run

### Setup (once per laptop)
1. Clone the repo and open the folder in VS Code (**File > Open Folder**):
   ```bash
   git clone https://github.com/segobe/MBDS_April2026_MLOps_Group3.git
   ```
2. In the VS Code terminal, at the repo root, create and activate the environment:
   ```bash
   conda env create -f environment.yml
   conda activate bike-demand
   ```
   Check: the prompt starts with `(bike-demand)` and `python --version` shows 3.12. If `environment.yml` changes later: `conda env update -f environment.yml --prune`
3. In VS Code: `>Python: Select Interpreter`, pick `bike-demand`.
4. Download `hour.csv` (UCI Bike Sharing Dataset) and place it in `data/raw/`.
5. Build the demo inference file. There is no real "tomorrow" in a fixed 2012 dataset, so this takes the 8 days before 2012-12-20 (history ending at 16:00 the day before) plus that day's 24 hours with the demand removed:
   ```bash
   python -c "
   import pandas as pd
   # TEMP: demo only, rows may have appeared in training
   raw = pd.read_csv('data/raw/hour.csv')
   date = pd.Timestamp('2012-12-20')
   d = pd.to_datetime(raw['dteday'])
   keep = (d >= date - pd.Timedelta(days=8)) & (d <= date) & ~((d == date - pd.Timedelta(days=1)) & (raw['hr'] >= 17))
   demo = raw[keep].copy()
   demo.loc[pd.to_datetime(demo['dteday']) == date, ['casual', 'registered', 'cnt']] = None
   demo.to_csv('data/inference/inference_input.csv', index=False)
   "
   ```
   **TEMP: demo only.** These rows may have appeared in training, so the forecast is a demonstration of the pipeline, never out-of-sample evidence.

Data, models and reports are never committed (`.gitignore`).

### Test (repo root, `bike-demand` active)
```bash
pytest -vv                                                                   # all tests, none skipped
pytest --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=95  # branch coverage, fails below 95 %
flake8 --max-complexity 10 src tests                                         # style and complexity: no output = clean
```
The tests use small synthetic data only; they need neither `hour.csv` nor the demo file.

### Run
```bash
python -m src.main
```
One command trains, checks the quality gate and forecasts tomorrow. It writes:

| Output | What it is for |
| --- | --- |
| `data/processed/clean.csv` | The cleaned hourly calendar with the past-demand features, kept for inspection and audit |
| `models/model.joblib` | The fitted pipeline (feature recipe + model), reused for the forecast |
| `reports/predictions.csv` | Tomorrow's forecast: `ts` (hour) and `prediction` (rentals), 24 rows |

The log states whether the quality gate is met. Always start the pipeline with `python -m`, not the VS Code Run button. Every setting (paths, cutoff, split, model, gate) lives in `SETTINGS` at the top of `src/main.py`; change values there only.

### Team workflow
1. Read `TEAM_BRIEF.md`: owners, branch names and shared decisions
2. Switch to `dev`, run `>Git: Pull`, then `>Git: Create Branch From...`, pick `dev`, and use your branch from the brief
3. Tests first, then code; open a Pull Request into `dev`. Never commit directly to `dev` or `main`

## 3. Results and evidence

| Evidence | Value | Source |
| --- | --- | --- |
| Validation `weighted_mae`, model and same hour last week | 79.58 vs 85.88 rentals per hour (July to September 2012) | `main` log |
| `weighted_mae_ratio` and gate status (threshold 0.85) | 0.927: **not met** | `main` log |
| Sensitivity at `under_weight` 1.5 and 3 | 0.926 and 0.927: the decision does not change | `notebooks/max_iter_and_gate.ipynb` |
| Training time and time to forecast one day | 8.2 s training; the whole run, including the 24-hour forecast, about 9 s | `main` log, notebook |
| Size of the system: modules, tests, branch coverage | 9 modules, 112 tests, 99 % | pytest output |

`max_iter` was re-selected on the validation split in the exploratory notebook (300 instead of the provisional 622, which overfits the 2011 demand level), and two cutoff-safe features were added (same hour yesterday before 17:00, last hour before the cutoff). Together they moved the ratio from 1.296 to 0.927. The test split was scored only once, in the notebook, and was never used for choices.

**Value, cost and complexity.** The value today is mostly in the checks, not yet in the model: the model beats same hour last week by 7 %, short of the 15 % gate, and the pipeline says so in every run instead of hiding it. What already works is the safety net: every run validates the data, inserts and flags missing hours, enforces the 17:00 cutoff (a leakage test proves no feature reads later data), and reports the gate on hours the model never saw. The cost is small: a full run takes about 9 seconds, far inside the 18:00 to 18:30 window, so the EUR 20 to 100 per month infrastructure estimate is generous. The complexity is the real price: nine modules, 112 tests and the team rules to maintain, against one notebook. That is worth paying only if the model passes the gate; until then the honest recommendation is to keep planning with same hour last week and use the pipeline to test the next model ideas safely.

### README agrees with the settings
| Item | README | Setting (`SETTINGS` in `src/main.py`) | Agrees |
| --- | --- | --- | --- |
| Metric and gate | `weighted_mae`, ratio <= 0.85 | `quality_gate`: `weighted_mae_ratio`, `<=`, 0.85 | ✓ |
| Under-forecast weight | 2 (with sensitivity) | `evaluation.under_weight` = 2.0, `model.quantile` = 2/3 | ✓ |
| Baseline | same hour last week | `evaluation.baseline_column` = `lag_168` | ✓ |
| Cutoff | data up to 17:00 | `cleaning.cutoff_hour` = 17 | ✓ |
| Split | chronological, about Q3 validation, Q4 test | `split`: `time_ordered` = True, 0.126 / 0.126 | ✓ |
| Excluded inputs | weather, `casual`, `registered` | `features`: only calendar and past-demand columns | ✓ |


# MLOps Project Skeleton (reference only)

This repo is a **map, not a codebase**. Every file is a placeholder: no code, only a short note on
what the file does, where it sits in the pipeline and which files it talks to.

Create the same folders and files in **your own** project repo, then write the code for **your** use case.

## Stages = branches

| Branch | Stage | Sessions | What it adds |
|---|---|---|---|
| `main` (you are here) | 1. Kickoff | 1 to 3 | Folder structure, business case, one placeholder per pipeline step |
| `stage-2-modular` | 2. Modular code | 4 to 9 | Tests, style and test settings, first CI (Continuous Integration) workflow |
| `stage-3-cicd` | 3. CI/CD | 10 to 14 | Config loader, logging, experiment tracking, API, Docker, deployment |

Tip: on GitHub, open **Compare** (`main...stage-2-modular`) to see exactly what each stage adds.

## The pipeline at a glance

```text
                           main.py  (orchestrator, one command)
                              |
 data/raw/ -> load_data -> clean_data -> validate -> split -> features -> train -> evaluate
                                                (train/val/test) (recipe)   (fit)   (val split)
                                                                              |
                     reports/predictions.csv <- infer <- models/model.joblib (save)
```

## Stage 1 structure

```text
.
├── README.md                 # business case, success metrics, how to run
├── config.yaml               # non-secret settings (paths, target, split sizes)
├── environment.yml           # dependencies (conda)
├── .env.example              # names of secrets, never real values
├── .gitignore                # keeps data, secrets and generated files out of Git
├── LICENSE
├── .github/
│   └── pull_request_template.md
├── notebooks/                # the "lab": exploration only
├── src/                      # the "factory": one job per module
│   ├── load_data.py          # 1. read raw data
│   ├── clean_data.py         # 2. stateless cleaning
│   ├── validate.py           # 3. data quality gate
│   ├── features.py           # 4. unfitted feature recipe
│   ├── train.py              # 5. fit the pipeline
│   ├── evaluate.py           # 6. metrics on unseen data
│   ├── infer.py              # 7. predictions on new data
│   ├── utils.py              # shared read and write helpers
│   └── main.py               # 0. orchestrator: python -m src.main
├── data/  raw/ processed/ inference/
├── models/                   # saved model (generated)
├── reports/                  # predictions, plots (generated)
└── tests/                    # automated tests (stage 2)
```

## How to use this skeleton

1. Open a file and read its note
2. Create the equivalent file in your repo
3. Write your own code: the note describes the job, not the solution
