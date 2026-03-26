# TOMS pipeline with K regressors (one per class)

This document describes the code behaviour for **TOMS multi-regressor** mode, as used in `run_experiment.py` with `EXP_TYPES = ["TOMS"]`.

## 1. Single scalar time `t` per window

- Each **window** is one element of `ts_chunks[i]` (a DataFrame of all tweets in that time interval).
- For the time column (`TweetAt` on the Covid setup), all rows in the chunk are converted to **days since the UNIX epoch** (same rule as `quantifications._time_column_to_X`).
- The **single** `t` for window `i` is the **median** of those values: robust to noise and equal to the single value when every tweet in the chunk shares one date (typical day-based loader).
- The map `window_t[i]` is stored in `TOMSMultiRegressorBundle` and used for training and for matrices `M` in validation/test logging.

## 2. `_window_id` column on the validation set

- `val_set` is the concatenation of chunks `0 .. val_length-1` (from `utils.val_test_split`).
- We set `val_set["_window_id"]` so each row knows its window (`attach_window_ids`), matching the median `t` of that window.

## 3. Training: K `TimeSeriesMultinomialRegressor` models

- Using the HF classifier, we obtain soft probabilities `Y` on the full `val_set` (`Classifying.analyzer`).
- For each class `c_k` (index `k = 0..K-1` in `classes` order):
  - Keep only rows whose **true label** is `label == c_k`.
  - **Input**: vector `t` where each sample uses `window_t[line's _window_id]` (same `t` for all lines in a window).
  - **Target**: the corresponding rows of `Y`, i.e. **K-dimensional** classifier scores — each regressor still predicts a simplex over **all** classes.
- If a class has no samples in `val_set`, a **uniform** constant regressor is used (`1/K` for all classes).
- Models are trained via `methods.regression.trainingModel.trainer` with `REGRESSOR_NAME` (e.g. `"TSMN"`).

## 4. Matrix `M` (K x K) at time `t`

- For scalar `t_w` (window time):
  - For each `k`, regressor `k` returns a `K`-dimensional probability vector.
  - Stack these `K` rows into **`M`**, shape **(K, K)** — row `k` is the normalized output of the regressor for class `k`.

## 5. Quantification with TOMS (per-window calibration from **M**)

- The `val_set` with HF `Y` and `_window_id` is used **only** to **train** the K regressors (section 3). Regressor scores derived from `M` over every line of `val_set` are **not** used as the quantifier reference anymore.
- For **each** window (time chunk with global index `wi`):
  - Compute **one** matrix `M` with `score_matrix_at_t(bundle, window_t[wi])`.
  - Build a synthetic calibration set of **K rows**: row `k` = `M[k, :]` (regressor `k` output), synthetic true label = `classes[k]` (`quantifications.val_labels_scores_from_toms_matrix`).
  - **Document scores** for the window (the “test” side of DyS/ACC/etc.) always come from the **HF classifier** (`Classifying.analyzer`), aggregated per chunk as before.
- In the **test** phase (`qtfied_dists`), `base_wi = val_length`: for test chunk `j`, use `M` at time `window_t[val_length + j]`.
- In **validation MAE** (`getMAE_val_set`), chunks are validation windows and `base_wi = 0`: for chunk `j`, use `M` at `window_t[j]` (that validation window's time), keeping the metric time-consistent.

## 6. HF vs regressors

- **HF**: targets `Y` for regressor training; per-document scores for the “test” side of each quantifier.
- **Regressors** enter quantifier calibration only through the **K rows of `M(t)`** per window (not via row-mean scores per line of the original `val_set`).

## 7. Output files

- **`output_files/regressor/toms_regressor_log.txt`** (append): header per run (dataset, quantifier, seed), per-class training block (`n`, `t` min/max, sample `Y`), validation/test sections with `M` and row-mean vectors.
- **`output_files/regressor/*.csv`**: per window: `t_window`, `M_r{i}_c{j}`, and `score_mean_{class}` (row-mean of `M`, renormalized).

## 8. Notes

- Temporal min–max inside each `TimeSeriesMultinomialRegressor` is learned **only** from training `t` for that class; at prediction, times outside that range are still linearly extrapolated in the model's internal normalization.
- On-disk cache under `output_files/results` remains **skipped** when a TOMS regressor is active (existing behaviour in `qtfied_dists`).
