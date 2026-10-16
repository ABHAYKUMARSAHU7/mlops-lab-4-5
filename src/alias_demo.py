import mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri("sqlite:///mlflow.db")
c = MlflowClient()
N = "HousePriceModel"
print("registered-model aliases:", c.get_registered_model(N).aliases)
mv = c.get_model_version_by_alias(N, "production")
print("production alias -> version", mv.version, "| run", mv.run_id)
print("version tags:", mv.tags)
for v in sorted(c.search_model_versions(f"name='{N}'"), key=lambda x: int(x.version)):
    print(f"v{v.version} aliases attribute:", v.aliases)
