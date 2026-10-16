"""DVC stage 'stats': summarise the current dataset version.

Reads configs/config.yaml and the dataset it points to, then writes
reports/data_stats.json (tracked by DVC as a metrics file).
Run from the repository root:  python src/data_stats.py
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

cfg = yaml.safe_load(open("configs/config.yaml"))
data_path = Path(cfg["data_path"])
target = cfg["target"]

df = pd.read_csv(data_path)
md5 = hashlib.md5(data_path.read_bytes()).hexdigest()

stats = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "dataset_md5": md5,
    "dataset_bytes": int(data_path.stat().st_size),
    "missing_values": int(df.isna().sum().sum()),
    "target_mean": round(float(df[target].mean()), 2),
    "target_min": round(float(df[target].min()), 2),
    "target_max": round(float(df[target].max()), 2),
}

Path("reports").mkdir(exist_ok=True)
with open("reports/data_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("data_stats:", stats)
