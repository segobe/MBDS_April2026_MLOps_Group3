# Team brief
Business case: README.md, section 1 (merged into dev on 2026-11-11)
Use case: the Operations Manager decides crew hours per shift the evening before, using a system-wide hourly forecast for tomorrow (README, Solution)
Notebook: notebooks/bike_demand_forecast.ipynb

## Data, features and model
Target: cnt | Problem type: regression | ID column: ts (timestamp from dteday + hr, built in clean_data)
Feature groups: calendar: hr, weekday, mnth, season, holiday, workingday -> passthrough (tree model; notebook used raw integers)
Feature groups: past_demand: lag_48, lag_168, mean_168_to_cutoff -> passthrough (built in clean_data with fixed past-only rules)
Excluded columns: instant (row number), dteday (replaced by ts), yr (not used in notebook), casual and registered (sum to the target), weathersit, temp, atemp, hum, windspeed (unknown the evening before), is_imputed (reveals the target)
Model: HistGradientBoostingRegressor, loss="quantile", quantile=2/3, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, early_stopping=False (default shuffles a hold-out and leaks the future; notebook cell 17), max_iter=622 (provisional, from the old 2-hour setup; re-select in an exploratory notebook)
Metric: weighted_mae (under-forecast weight 2.0), key in evaluate_model output: ["weighted_mae_ratio"] (model / same hour last week), quality gate: <= 0.85 (README, Goal and success metrics)
Split: chronological, no shuffling: test 0.126 and validation 0.126, both of the full dataset (about Q3 2012 validation, Q4 2012 test); random_state 42 (model only)
Validation rules: required columns present; no duplicate ts; hr in 0..23; cnt non-negative integer; cnt = casual + registered (training mode only); holiday and workingday in {0, 1} (notebook section 1; README risk: data quality)

## Contract details, decisions and owners
README follow-ups: rewrite problem statement for crew scheduling -> Sebastian Gollhofer; re-select max_iter and check the 0.85 gate at the 17:00 cutoff in an exploratory notebook, then run the gate at weights 1.5 and 3 -> Daniel Rodriguez; crew KPI baselines and batch times -> Sebastian Gollhofer with client
features.py signature: get_feature_preprocessor(calendar_columns, past_demand_columns), settings per group: none (passthrough)
Agreed extra parameters: split_data(time_ordered=False), clean_dataframe(cutoff_hour=17), evaluate_model(under_weight=2.0, baseline_column=None)
utils.py helpers: load_csv, save_csv, save_model, load_model
Environment: no packages beyond pandas, numpy, scikit-learn, joblib, pytest, pytest-cov, flake8, plus notebook tools matplotlib and ipykernel (notebooks only, never imported in src/)
Decisions: split -> chronological, SETTINGS passes time_ordered=True (shuffling leaks the future; notebook section 4)
Decisions: missing hours -> inserted in clean_data with cnt, casual and registered 0 and is_imputed 1 (notebook section 2; README Data)
Decisions: lags -> fixed rules relative to 17:00 the day before the target date, in clean_data; training mode drops rows without 7 days of history (to confirm)
Decisions: inference input -> history up to 17:00 today plus tomorrow's 24 calendar rows; main passes only tomorrow's rows to run_inference (to confirm)
Decisions: gate -> evaluate_model computes weighted_mae for the model and for lag_168 on the validation split and returns their ratio (to confirm)
Decisions: negative forecasts -> run_inference and evaluate_model clip predictions at 0 (notebook cells 17 and 20)
Decisions: split boundaries -> approximately Q3/Q4 2012, about a day off after dropping history rows (accepted; README baseline 104.72 is on exact Q4)
Owners (team): utils -> Sebastiano Cuniberti (@sebastianocuniberti), branch feature/utils
Owners (team): load_data -> Sebastiano Cuniberti (@sebastianocuniberti), branch feature/load-data
Owners (team): clean_data -> Sebastian Gollhofer (@segobe), branch feature/clean-data
Owners (team): validate -> Daniella Jimenez (@danijimenez96), branch feature/validate
Owners (team): features -> Daniella Jimenez (@danijimenez96), branch feature/features
Owners (team): train -> Ahsan Sheikh (@ahsanshaik-glitch), branch feature/train
Owners (team): evaluate -> Daniel Rodriguez (@Danielrodgo), branch feature/evaluate
Owners (team): infer -> Ahsan Sheikh (@ahsanshaik-glitch), branch feature/infer
Owners (team): main -> Sebastian Gollhofer (@segobe), branch feature/main