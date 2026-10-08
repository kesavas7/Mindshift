# HaulMark — Upgraded Pipeline (Fuel Consumption Prediction)

A machine learning pipeline designed for the **HaulMark Challenge**, predicting fuel consumption (`acons`) across haulage vehicles using multi-source telemetry, summary logs, and fleet metadata.

---

## Key Features & Pipeline Improvements

- **Blended Sequential Drift Correction:** Mitigates error compounding during iterative time-series inference by blending raw predictions with recent history and vehicle baselines.
- **Advanced Telemetry Feature Engineering:** Extracts speed-bin fractions, stop durations, route variability metrics, and jerk (acceleration derivative) from raw sensor streams.
- **Feature Processing:** Includes Exponential Moving Averages (EWM), trend differences, cold-start indicators, workload interactions, and nonlinear transformations.
- **Analog Dump Signal Integration:** Detects dumping events using orientation sensor dynamics (tilt/axis thresholding).
- **Per-Vehicle Target Normalization:** Models relative consumption behavior (`acons / vehicle_avg`) to decouple baseline variances before scaling back at inference.
- **Model Architecture:** CatBoost-centric ensemble (integrating LightGBM and XGBoost fallbacks) leveraging multi-seed training and dual-target blending (log, raw, and normalized spaces)[cite: 1].
- **Validation Scheme:** Time-series cross-validation using rolling temporal cutoffs[cite: 1].

---

## Data Architecture

The pipeline processes and merges three main datasets:
1. **Summary Logs:** Shift-level ground truth records containing operational variables[cite: 1].
2. **Telemetry Parquet Files:** High-frequency GPS and accelerometer telemetry aggregated per shift[cite: 1].
3. **Fleet Metadata:** Static vehicle attributes (e.g., tank capacities, mine mappings)[cite: 1].

---

## Requirements & Dependencies

To execute this notebook, install the following dependencies:

```bash
pip install numpy pandas scikit-learn catboost xgboost lightgbm pyarrow
