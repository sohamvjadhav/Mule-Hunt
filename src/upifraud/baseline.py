"""Non-graph baselines: RandomForest and gradient-boosted trees on flattened node features."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from torch_geometric.data import Data

from .train import standardize


def train_baseline(
    data: Data,
    model_name: str = "rf",
    seed: int = 42,
    out_dir: Path = Path("models"),
) -> dict:
    x, mean, std = standardize(data.x)
    X = x.numpy()
    y = data.y.numpy()

    X_train, y_train = X[data.train_mask.numpy()], y[data.train_mask.numpy()]

    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=200, max_depth=16, class_weight="balanced", n_jobs=-1, random_state=seed
        )
    elif model_name == "hgb":
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.1,
            max_depth=8,
            class_weight="balanced",
            random_state=seed,
        )
    elif model_name == "xgb":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            nthread=1,
            scale_pos_weight=max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1)),
            eval_metric="aucpr",
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown baseline {model_name!r}; choose from ['rf', 'hgb', 'xgb']")

    cv = cross_val_score(
        model, X_train, y_train, cv=3, scoring="average_precision", n_jobs=1
    )

    model.fit(X_train, y_train)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / f"{model_name}.joblib")
    (out_dir / f"{model_name}_args.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "cv_ap_mean": float(cv.mean()),
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            indent=2,
        )
    )

    scores = model.predict_proba(X)[:, 1]
    return {
        "model": model,
        "args": json.loads((out_dir / f"{model_name}_args.json").read_text()),
        "scores": scores,
    }


def evaluate_baseline(data: Data, scores: np.ndarray, split: str = "test") -> dict:
    mask = getattr(data, f"{split}_mask").numpy()
    y = data.y.numpy()[mask]
    s = scores[mask]
    return {
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "n_fraud": int(y.sum()),
        "n_total": len(y),
    }
