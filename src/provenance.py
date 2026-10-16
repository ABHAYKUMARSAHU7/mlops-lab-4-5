import mlflow
from mlflow import MlflowClient
mlflow.set_tracking_uri("sqlite:///mlflow.db")
c = MlflowClient()
NAME = "HousePriceModel"
V1_MD5 = "a16639ea8a90759346d794391e78caea"
NOTE = "backfilled after training; dataset v1 inferred from timeline (file unchanged until v2 commit 8be08a4)"
for v in sorted(c.search_model_versions(f"name='{NAME}'"), key=lambda x: int(x.version)):
    if int(v.version) > 3:
        continue
    run = c.get_run(v.run_id)
    git = run.data.tags.get("mlflow.source.git.commit", "not recorded")
    tags = {"dataset_version": "v1", "dataset_md5": V1_MD5, "git_commit": git, "provenance_note": NOTE}
    for k, val in tags.items():
        c.set_model_version_tag(NAME, v.version, k, val)
        c.set_tag(v.run_id, k, val)
    print(f"v{v.version} ({run.info.run_name}) run={v.run_id} git_commit={git}")
print()
for v in sorted(c.search_model_versions(f"name='{NAME}'"), key=lambda x: int(x.version)):
    print(f"v{v.version}:", {k: val for k, val in v.tags.items() if k in ("dataset_version", "dataset_md5", "git_commit", "lifecycle")})
