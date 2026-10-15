import hashlib
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


cfg = yaml.safe_load(open("configs/config.yaml"))

# --- Code and data provenance ---
commit = git("rev-parse", "HEAD")
commit_short = git("rev-parse", "--short", "HEAD")
dirty = git("status", "--short", "--untracked-files=no") != ""

dvc_meta = yaml.safe_load(open(cfg["data_path"] + ".dvc"))
dataset_md5 = dvc_meta["outs"][0]["md5"]
actual_md5 = hashlib.md5(open(cfg["data_path"], "rb").read()).hexdigest()
assert actual_md5 == dataset_md5, "Dataset on disk does not match the .dvc pointer. Run: dvc checkout"

version_names = {"a16639ea": "v1", "8601f9b1": "v2"}
dataset_version = version_names.get(dataset_md5[:8], "unknown")

# --- Data and model ---
df = pd.read_csv(cfg["data_path"])
X = df.drop(columns=[cfg["target"]])
y = df[cfg["target"]]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=cfg["test_size"], random_state=cfg["seed"]
)
model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(cfg["experiment_name"])

with mlflow.start_run(run_name="integration_rf_large") as run:
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    mae = mean_absolute_error(y_te, pred)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    r2 = r2_score(y_te, pred)

    mlflow.set_tags({
        "git_commit": commit,
        "git_commit_short": commit_short,
        "git_dirty": str(dirty),
        "dataset_md5": dataset_md5,
        "dataset_version": dataset_version,
        "dataset_pointer": cfg["data_path"] + ".dvc",
    })
    mlflow.log_param("algorithm", "RandomForest")
    mlflow.log_params(model.get_params())
    mlflow.log_param("dataset_rows", len(df))
    mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

    plt.figure()
    plt.scatter(pred, y_te - pred, s=8)
    plt.axhline(0, color="red")
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("integration_rf_large")
    fig_path = "reports/figures/residuals_integration_rf_large.png"
    plt.savefig(fig_path)
    plt.close()
    mlflow.log_artifact(fig_path)
    mlflow.log_artifact("configs/config.yaml")
    mlflow.log_artifact(cfg["data_path"] + ".dvc")

    signature = infer_signature(X_te, pred)
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=X_te.head(3),
        registered_model_name=cfg["registered_model_name"],
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    client = MlflowClient()
    ver = info.registered_model_version
    name = cfg["registered_model_name"]
    client.set_model_version_tag(name, ver, "git_commit", commit)
    client.set_model_version_tag(name, ver, "dataset_md5", dataset_md5)
    client.set_model_version_tag(name, ver, "dataset_version", dataset_version)
    client.update_model_version(
        name, ver,
        description=f"RandomForest trained on dataset {dataset_version} "
                    f"(md5 {dataset_md5[:8]}) at Git commit {commit_short}.",
    )

    print(f"Run ID:          {run.info.run_id}")
    print(f"Git commit:      {commit_short} (dirty={dirty})")
    print(f"Dataset:         {dataset_version}, md5 {dataset_md5}, rows {len(df)}")
    print(f"Metrics:         MAE={mae:.0f} RMSE={rmse:.0f} R2={r2:.3f}")
    print(f"Registered:      {name} version {ver}")