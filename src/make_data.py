import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500

df = pd.DataFrame({
    "area_sqft": rng.integers(500, 3500, n).astype("float64"),
    "bedrooms": rng.integers(1, 6, n).astype("float64"),
    "age_years": rng.integers(0, 50, n).astype("float64"),
    "distance_to_city_km": rng.uniform(1, 40, n),
})

df["price"] = (
    df.area_sqft * 120
    + df.bedrooms * 8000
    - df.age_years * 700
    - df.distance_to_city_km * 1500
    + rng.normal(0, 15000, n)
).round(2)

df.to_csv("data/dataset.csv", index=False)
print("Saved data/dataset.csv with shape:", df.shape)
