import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
name = "HousePriceModel"

versions = client.search_model_versions(f"name='{name}'")
for v in sorted(versions, key=lambda v: int(v.version)):
    run = client.get_run(v.run_id)
    m = run.data.metrics
    print(f"--- v{v.version} | run: {run.info.run_name} | run_id: {v.run_id}")
    print(f"    MAE={m.get('mae')}  RMSE={m.get('rmse')}  R2={m.get('r2')}")
    print(f"    aliases: {list(v.aliases)}")
    print(f"    tags: {dict(v.tags)}")
    print(f"    description: {v.description}")