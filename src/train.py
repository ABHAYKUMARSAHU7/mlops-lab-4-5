import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

cfg = yaml.safe_load(open("configs/config.yaml"))
df = pd.read_csv(cfg["data_path"])
X = df.drop(columns=[cfg["target"]])
y = df[cfg["target"]]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=cfg["test_size"], random_state=cfg["seed"]
)

candidates = {
    "ridge_a1": ("Ridge", Ridge(alpha=1.0)),
    "rf_small": ("RandomForest", RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)),
    "rf_large": ("RandomForest", RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)),
}

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(cfg["experiment_name"])

for run_name, (algo, model) in candidates.items():
    with mlflow.start_run(run_name=run_name):
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        mae = mean_absolute_error(y_te, pred)
        rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
        r2 = r2_score(y_te, pred)

        mlflow.log_param("algorithm", algo)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})

        plt.figure()
        plt.scatter(pred, y_te - pred, s=8)
        plt.axhline(0, color="red")
        plt.xlabel("Predicted")
        plt.ylabel("Residual")
        plt.title(run_name)
        fig_path = f"reports/figures/residuals_{run_name}.png"
        plt.savefig(fig_path)
        plt.close()
        mlflow.log_artifact(fig_path)
        mlflow.log_artifact("configs/config.yaml")

        signature = infer_signature(X_te, pred)
        mlflow.sklearn.log_model(
            model, name="model", signature=signature, input_example=X_te.head(3), skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        print(f"{run_name}: MAE={mae:.0f} RMSE={rmse:.0f} R2={r2:.3f}")

