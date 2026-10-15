import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
NAME = "HousePriceModel"
MAX_MAE, MIN_R2 = 15000, 0.97

# cleanup of the stray tag from the failed UI dialog (ignore if already gone)
try:
    client.delete_model_version_tag(NAME, "3", "source run")
except Exception:
    pass

# 1. mark all three as candidates and evaluate against the rule
results = {}
for v in ["1", "2", "3"]:
    mv = client.get_model_version(NAME, v)
    m = client.get_run(mv.run_id).data.metrics
    ok = m["mae"] <= MAX_MAE and m["r2"] >= MIN_R2
    results[v] = ok
    client.set_model_version_tag(NAME, v, "lifecycle", "candidate")
    client.set_model_version_tag(NAME, v, "validation",
        f"{'PASS' if ok else 'FAIL'} (mae={m['mae']:.2f}, r2={m['r2']:.4f}; rule mae<={MAX_MAE}, r2>={MIN_R2})")
    print(f"v{v}: mae={m['mae']:.2f} r2={m['r2']:.4f} -> {'PASS' if ok else 'FAIL'}")

# 2. promote the passing, best model (v1) to production
client.set_registered_model_alias(NAME, "production", "1")
client.set_model_version_tag(NAME, "1", "lifecycle", "production")

# 3. retire the weakest model (v2)
client.set_model_version_tag(NAME, "2", "lifecycle", "archived")
client.set_model_version_tag(NAME, "2", "archive_reason", "weakest of three candidates; superseded by v1")

# 4. what a teammate would do to find production
prod = client.get_model_version_by_alias(NAME, "production")
print("PRODUCTION -> version", prod.version, "| source run", prod.run_id, "| tags", prod.tags)