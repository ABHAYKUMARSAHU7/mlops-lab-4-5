import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
NAME = "HousePriceModel"

info = {
    "1": ("ridge_a1", "ridge",
          "Ridge regression (alpha=1.0), linear baseline. Test MAE 14152.54, R2 0.9755. Best of the three candidates on both metrics."),
    "2": ("rf_small", "random_forest_small",
          "Random forest, small configuration (rf_small). Test MAE 17534.38, R2 0.9610. Weakest of the three."),
    "3": ("rf_large", "random_forest_large",
          "Random forest, large configuration (rf_large). Test MAE 16378.05, R2 0.9669. Better than v2 but behind v1."),
}

for version, (run_name, algo, desc) in info.items():
    client.set_model_version_tag(NAME, version, "source_run", run_name)
    client.set_model_version_tag(NAME, version, "algorithm", algo)
    client.update_model_version(NAME, version, description=desc)
    print("updated version", version)