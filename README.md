# Hourly Bike Rental Demand Forecasting

> Capital Bikeshare should fund this MLOps forecasting system because gated releases, drift monitoring and automatic fallback keep tomorrow's crew plans reliable, unlike one-off notebooks.

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

Inputs: calendar variables, `lag_48`, `lag_168` and the 7-day mean ending at the 17:00 cutoff. Weather, `casual`, `registered` and `is_imputed` are excluded.

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
- Problem statement still describes station shortages and rebalancing; rewrite it in the team's words to match the crew scheduling decision.
- Idle crew hour and overtime baselines: To be measured (need client rosters and payroll records); the idle hours target is still to be agreed with the client.
- The 2x under-forecast weight and the 15% threshold are assumptions to validate; the sensitivity check at 1.5 and 3 has not been run yet.
- Batch times (18:00 run, 18:30 ready) are assumptions to validate with the client's planning routine.
- The next-evening model is not yet built: notebook features, `LAG_MIN` and the leakage test must be updated to the 17:00 cutoff, and the gate must be run against same hour last week.
- The 4-week drift window is a starting choice, not yet tested.
- Cost per prediction can only be measured in the deployed system.




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
