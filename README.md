# MLOps Lab 4-5: MLflow Model Registry and DVC Data Versioning

House-price regression project that links **Git** (code), **DVC** (data) and **MLflow** (experiments and models), so another engineer can identify a deployed model and recover the code and dataset used to create it.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Tool versions: MLflow 3.17.1 (aliases and tags are used instead of stages), DVC, scikit-learn, pandas, matplotlib.

## Project layout

```
mlops-lab-4-5/
├── .dvc/                      DVC configuration (remote, settings)
├── data/
│   └── dataset.csv.dvc        DVC pointer to the dataset (CSV itself is not in Git)
├── src/
│   ├── make_data.py           generates the synthetic dataset
│   ├── train.py               trains 3 candidates, logs to MLflow
│   ├── autolog_demo.py        MLflow autologging demo
│   ├── annotate_versions.py   adds descriptions/tags to registered versions
│   ├── lifecycle.py           candidate / production / archived workflow
│   └── integrate.py           Git + DVC + MLflow integration run
├── models/
├── reports/figures/           residual plots
├── configs/config.yaml        experiment name, target, seed, registered model name
├── requirements.txt
└── README.md
```

## Data versioning (DVC)

Git tracks only the small pointer file `data/dataset.csv.dvc`. The CSV lives in the DVC cache and the DVC remote.

| Version | Git commit | Rows | Columns | Size (bytes) | MD5 |
|---|---|---|---|---|---|
| Dataset v1 | `37271e4` | 500 | 5 | 22416 | `a16639ea8a90759346d794391e78caea` |
| Dataset v2 | `8be08a4` | 550 | 5 | 24556 | `8601f9b10b2111f08ca0f5315a182ddd` |

v2 adds 50 rows to v1.

**Remote:** the DVC remote is a local folder outside the repository (`../dvc-storage`), which is useful for learning the commands on one machine but is not a shared backup. Because of this, the CSV is **not** on GitHub. To get the data:

```powershell
# Option 1: with access to the DVC remote
dvc pull

# Option 2: regenerate the data, then track it
python src/make_data.py
dvc add data/dataset.csv
```

### Recover an earlier dataset version

Use a temporary branch so `main` stays untouched.

```powershell
git switch -c recover-v1 37271e4
dvc checkout
python -c "import pandas as pd; print(pd.read_csv('data/dataset.csv').shape)"
Get-FileHash data\dataset.csv -Algorithm MD5     # expect A16639EA...

# return to the latest version
git switch main
dvc checkout
Get-FileHash data\dataset.csv -Algorithm MD5     # expect 8601F9B1...
git branch -d recover-v1
```

Verification: row count (500 vs 550), file size, and MD5 compared with the values recorded in the `.dvc` file for each commit. A filename alone is not enough, because `dataset.csv` is the same name in every version.

## Training and MLflow

```powershell
python src/train.py        # runs ridge_a1, rf_small, rf_large and logs them
python src/integrate.py    # tagged run + registers a new HousePriceModel version
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000. Tracking store: `sqlite:///mlflow.db`, experiment: `house-price`.

What each run logs: parameters, MAE / RMSE / R², a residual plot, `config.yaml`, and the model with a signature and input example.

### Registered model: `HousePriceModel`

| Version | Source run | Notes |
|---|---|---|
| v1 | `ridge_a1` | Ridge regression baseline |
| v2 | `rf_small` | RandomForest, 50 trees, depth 5 |
| v3 | `rf_large` | RandomForest, 300 trees, depth 12 |
| v4 | `integration_rf_large` | Same model as v3, trained at Git commit `4f6aec4` on dataset v2, tagged for traceability |

Lifecycle is managed with aliases and tags (MLflow 3.x): a candidate tag/alias for validation, a `production` alias for the deployed version, and an archived/retired tag for superseded versions. The acceptance rule and evidence are documented in the report.

Retrieve the production model:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/HousePriceModel@production")
```

## Git + DVC + MLflow integration

`src/integrate.py` checks that the dataset on disk matches its `.dvc` pointer, trains the model, and records the following on the MLflow run and on the registered model version:

| Tag | Meaning |
|---|---|
| `git_commit` | Full Git commit ID of the training code |
| `git_dirty` | Whether tracked files had uncommitted changes |
| `dataset_md5` | MD5 of the dataset from `data/dataset.csv.dvc` |
| `dataset_version` | Human-readable dataset version (v1, v2) |

### Investigating a prediction

1. Open `HousePriceModel` in the MLflow registry and read the version's description and tags.
2. Recover the code: `git switch -c investigate <git_commit>`.
3. Recover the data: `dvc checkout` (or `dvc pull`), then compare its MD5 with `dataset_md5`.
4. Load the model: `mlflow.sklearn.load_model("models:/HousePriceModel/<version>")`.

| Tool | Records | Lives in |
|---|---|---|
| Git | Code, config, `.dvc` pointer files | Git history / GitHub |
| DVC | Dataset versions (by hash) | `.dvc` files in Git, data in DVC cache and remote |
| MLflow | Parameters, metrics, artifacts, model versions, aliases, tags | `mlflow.db`, `mlruns/` |

## Not included in the repository

`.venv/`, `mlruns/`, `mlflow.db`, `mlartifacts/` and `*.pkl` are excluded through `.gitignore`. The dataset CSV is excluded through DVC.
