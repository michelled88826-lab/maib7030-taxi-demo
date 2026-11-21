"""Model and data layer for the MAIB7030 historical taxi-demand demo.

The demo trains only on 8 Jan–30 Apr 2025 and replays May 2025 validation rows.
June and July are deliberately excluded from this application.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

TRAIN_CUTOFF = pd.Timestamp("2025-05-01 00:00:00")
DEMO_END = pd.Timestamp("2025-06-01 00:00:00")
TARGET = "pickup_count"
NUMERIC_FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend", "hour_sin", "hour_cos",
    "day_sin", "day_cos", "is_holiday", "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_24h",
]


def load_demo_data(data_path: str | Path) -> pd.DataFrame:
    """Load and validate a sanitized Jan–May dataset (no June/July records)."""
    df = pd.read_csv(data_path, parse_dates=["hour"])
    required = {"PULocationID", "hour", TARGET, *NUMERIC_FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(df) != 34_550:
        raise ValueError(f"Unexpected dataset version: expected 34,550 rows, found {len(df):,}.")
    if df[list(required)].isna().any().any():
        raise ValueError("The prepared dataset contains missing values in required fields.")
    if df.duplicated(["PULocationID", "hour"]).any():
        raise ValueError("Duplicate zone-hour keys found.")
    if (df[TARGET] < 0).any():
        raise ValueError("Negative pickup counts found.")

    # Drop June/July before any model fitting or app-side evaluation.
    demo = df.loc[df["hour"] < DEMO_END].copy()
    demo = demo.sort_values(["hour", "PULocationID"]).reset_index(drop=True)
    if int((demo["hour"] < TRAIN_CUTOFF).sum()) != 27_110:
        raise ValueError("Training split does not match the frozen Jan–Apr sample count.")
    if int(((demo["hour"] >= TRAIN_CUTOFF) & (demo["hour"] < DEMO_END)).sum()) != 7_440:
        raise ValueError("May replay split does not match the frozen validation sample count.")
    return demo


def _new_hgb() -> HistGradientBoostingRegressor:
    # Exact hyperparameters from Member B's executed notebook.
    return HistGradientBoostingRegressor(
        max_iter=150,
        max_leaf_nodes=31,
        learning_rate=0.08,
        l2_regularization=1,
        random_state=42,
        early_stopping=False,
    )


def build_feature_matrices(df: pd.DataFrame):
    """Build training and May matrices with train-only zone vocabulary."""
    train_mask = df["hour"] < TRAIN_CUTOFF
    zones = sorted(df.loc[train_mask, "PULocationID"].unique().tolist())
    if len(zones) != 10:
        raise ValueError(f"Expected 10 training zones, found {len(zones)}.")

    zone_cat = pd.Categorical(df["PULocationID"], categories=zones)
    zone_dummies = pd.get_dummies(zone_cat, prefix="zone", dtype=float)
    augmented = pd.concat([df.reset_index(drop=True), zone_dummies.reset_index(drop=True)], axis=1)

    baseline_features = NUMERIC_FEATURES + list(zone_dummies.columns)
    augmented["recent_gap_3h"] = augmented["lag_1h"] - augmented["rolling_mean_3h"]
    augmented["relative_gap_3h"] = augmented["recent_gap_3h"] / (10 + augmented["rolling_mean_3h"])
    improved_features = baseline_features + ["recent_gap_3h", "relative_gap_3h"]

    if not np.isfinite(augmented[improved_features].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values found in model features.")
    return augmented, zones, baseline_features, improved_features, train_mask


def train_demo_models(df: pd.DataFrame) -> dict[str, Any]:
    """Load frozen JAN–APR models, NEVER re-fit them on the teacher's machine.

    The historical name is preserved to keep existing app imports compatible.
    Source artifacts were generated once with the Member B model specifications.
    """
    if sklearn.__version__ != "1.8.0":
        raise RuntimeError(
            f"Frozen models require scikit-learn 1.8.0, found {sklearn.__version__}. "
            "Run: python -m pip install 'scikit-learn==1.8.0'"
        )
    model_dir = Path(__file__).resolve().parent / "models"
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Missing models/manifest.json: use the v1.2 complete package.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_path = Path(__file__).resolve().parent / "data" / "taxi_demand_features.csv"
    checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if checksum != manifest["source_csv_sha256"]:
        raise RuntimeError("Frozen model/data mismatch. Do not replace or edit the prepared CSV.")

    augmented, zones, base_cols, improved_cols, train_mask = build_feature_matrices(df)
    may_mask = (augmented["hour"] >= TRAIN_CUTOFF) & (augmented["hour"] < DEMO_END)
    if zones != manifest["zones"]:
        raise RuntimeError("Unexpected taxi-zone vocabulary. The frozen model cannot be used.")
    y_may = augmented.loc[may_mask, TARGET].to_numpy(dtype=float)

    def load_frozen(key: str, columns):
        entry = manifest["models"][key]
        if entry["feature_columns"] != columns:
            raise RuntimeError(f"Feature ordering mismatch for {key}; cannot run safely.")
        artifact_path = model_dir / entry["filename"]
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing frozen artifact: {artifact_path.name}")
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Model checksum mismatch: {key}. Use untouched artifacts.")
        # SECURITY: joblib is pickle-backed. Only load the verified artifacts
        # from this trusted package; never accept an untrusted uploaded model.
        return joblib.load(artifact_path)

    baseline = load_frozen("hgb_baseline", base_cols)
    improved = load_frozen("hgb_transition", improved_cols)
    pred_base = np.maximum(0.0, baseline.predict(augmented.loc[may_mask, base_cols]))
    pred_improved = np.maximum(0.0, improved.predict(augmented.loc[may_mask, improved_cols]))

    def metrics(y, p):
        return {
            "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "WAPE": float(np.abs(y - p).sum() / y.sum()),
        }

    return {
        "baseline": baseline,
        "improved": improved,
        "zones": zones,
        "baseline_features": base_cols,
        "improved_features": improved_cols,
        "may_metrics": {
            "HGB baseline": metrics(y_may, pred_base),
            "HGB + transition": metrics(y_may, pred_improved),
        },
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(may_mask.sum()),
        "model_source": "Frozen Jan-Apr pre-trained artifacts with SHA-256 verification",
    }


def _single_row_features(row: pd.Series, model_bundle: dict[str, Any], improved: bool) -> pd.DataFrame:
    zones = model_bundle["zones"]
    values = {name: float(row[name]) for name in NUMERIC_FEATURES}
    for zone in zones:
        values[f"zone_{zone}"] = float(int(row["PULocationID"]) == int(zone))
    if improved:
        recent_gap = float(row["lag_1h"] - row["rolling_mean_3h"])
        values["recent_gap_3h"] = recent_gap
        values["relative_gap_3h"] = recent_gap / (10.0 + float(row["rolling_mean_3h"]))
        cols = model_bundle["improved_features"]
    else:
        cols = model_bundle["baseline_features"]
    return pd.DataFrame([[values[c] for c in cols]], columns=cols)


def predict_row(row: pd.Series, model_bundle: dict[str, Any]) -> dict[str, float]:
    """Return both model predictions for a target-hour feature row."""
    xb = _single_row_features(row, model_bundle, improved=False)
    xi = _single_row_features(row, model_bundle, improved=True)
    pb = max(0.0, float(model_bundle["baseline"].predict(xb)[0]))
    pi = max(0.0, float(model_bundle["improved"].predict(xi)[0]))
    return {"HGB baseline": pb, "HGB + transition": pi}


def get_case_row(df: pd.DataFrame, zone_id: int, target_hour: pd.Timestamp) -> pd.Series:
    match = df.loc[(df["PULocationID"] == int(zone_id)) & (df["hour"] == target_hour)]
    if len(match) != 1:
        raise ValueError(f"No unique May validation row for zone {zone_id} at {target_hour}.")
    return match.iloc[0]


def get_prior_history(df: pd.DataFrame, zone_id: int, target_hour: pd.Timestamp, hours: int = 24) -> pd.DataFrame:
    """Return only completed hours strictly before the target hour."""
    start = target_hour - pd.Timedelta(hours=hours)
    hist = df.loc[
        (df["PULocationID"] == int(zone_id))
        & (df["hour"] >= start)
        & (df["hour"] < target_hour),
        ["hour", TARGET],
    ].sort_values("hour")
    return hist.copy()


def reveal_actual(df: pd.DataFrame, zone_id: int, target_hour: pd.Timestamp) -> int:
    """Fetch the selected target only when the user explicitly reveals it."""
    match = df.loc[(df["PULocationID"] == int(zone_id)) & (df["hour"] == target_hour), TARGET]
    if len(match) != 1:
        raise ValueError("No unique actual value exists for the selected validation case.")
    return int(match.iloc[0])
