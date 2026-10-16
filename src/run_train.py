"""DVC stage 'train': train one model, log it to MLflow with provenance tags,
and write reports/pipeline_metrics.json (tracked by DVC as a metrics file).

The MLflow run id is written into the metrics file, so
`dvc metrics show` points to the MLflow run, and the MLflow run carries the
Git commit and the dataset md5 as tags -- each side can find the other.

Run from the repository root:  python src/run_train.py
This script is self-contained: it does not call train.py, so it does not
create extra registered model versions.
"""
import json
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

cfg = yaml.safe_load(open("configs/config.yaml"))


def git(*args):
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "not recorded"


def dataset_md5_from_dvc(pointer="data/dataset.csv.dvc"):
    """md5 stored in the committed .dvc pointer (the dataset version)."""
    try:
        return yaml.safe_load(open(pointer))["outs"][0]["md5"]
    except Exception:
        return "not recorded"


df = pd.read_csv(cfg["data_path"])
X = df.drop(columns=[cfg["target"]])
y = df[cfg["target"]]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=cfg["test_size"], random_state=cfg["seed"]
)

model = Ridge(alpha=1.0)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("dvc-pipeline")

with mlflow.start_run(run_name="pipeline_ridge") as run:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae = float(mean_absolute_error(y_te, pred))
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2 = float(r2_score(y_te, pred))

    status = git("status", "--porcelain")
    mlflow.set_tags({
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": str(bool(status)),
        "dataset_md5": dataset_md5_from_dvc(),
        "dataset_rows": str(len(df)),
        "dataset_pointer": "data/dataset.csv.dvc",
        "pipeline": "dvc repro (stats -> train)",
    })
    mlflow.log_params({"algorithm": "Ridge", "alpha": 1.0,
                       "test_size": cfg["test_size"], "seed": cfg["seed"]})
    mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
    mlflow.sklearn.log_model(
        model, name="model",
        signature=infer_signature(X_te, pred),
        input_example=X_te.head(3),
    )

    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "rows": int(len(df)),
        "mlflow_run_id": run.info.run_id,
    }

Path("reports").mkdir(exist_ok=True)
with open("reports/pipeline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("run_train:", metrics)
