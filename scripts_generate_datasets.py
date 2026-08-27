"""
One-off script used to generate the built-in datasets shipped with ML Lab.
Run once: python scripts_generate_datasets.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris

DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

# --------------------------------------------------------------------------- #
# IRIS
# --------------------------------------------------------------------------- #
iris = load_iris(as_frame=True)
df_iris = iris.frame.copy()
df_iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
species_map = {i: name for i, name in enumerate(iris.target_names)}
df_iris["species"] = df_iris["target"].map(species_map)
df_iris = df_iris.drop(columns=["target"])
df_iris.to_csv(DATASETS_DIR / "iris.csv", index=False)
print("iris.csv written:", df_iris.shape)

# --------------------------------------------------------------------------- #
# HOUSE PRICES (synthetic, realistic relationships)
# --------------------------------------------------------------------------- #
n = 800
area = rng.normal(1500, 500, n).clip(300, 6000)
bedrooms = rng.integers(1, 6, n)
bathrooms = rng.integers(1, 4, n)
age = rng.integers(0, 60, n)
parking = rng.integers(0, 4, n)
location_score = rng.uniform(1, 10, n)
floor = rng.integers(0, 20, n)
distance_to_city = rng.exponential(8, n).clip(0.2, 60)

noise = rng.normal(0, 15000, n)
price = (
    area * 120
    + bedrooms * 8000
    + bathrooms * 6000
    - age * 700
    + parking * 2500
    + location_score * 9000
    - distance_to_city * 1200
    + 20000
    + noise
).clip(15000, None)

df_house = pd.DataFrame({
    "area": area.round(1),
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age": age,
    "parking": parking,
    "location_score": location_score.round(2),
    "floor": floor,
    "distance_to_city": distance_to_city.round(2),
    "price": price.round(0),
})
df_house.to_csv(DATASETS_DIR / "house_prices.csv", index=False)
print("house_prices.csv written:", df_house.shape)

# --------------------------------------------------------------------------- #
# CUSTOMER CHURN (synthetic, realistic relationships)
# --------------------------------------------------------------------------- #
n = 1000
tenure = rng.integers(0, 72, n)
monthly_charges = rng.normal(70, 25, n).clip(15, 150)
total_charges = (monthly_charges * tenure * rng.uniform(0.85, 1.05, n)).clip(0, None)
contract_type = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
support_calls = rng.poisson(1.5, n)
satisfaction = rng.integers(1, 6, n)
internet_service = rng.choice(["DSL", "Fiber", "No_internet"], n, p=[0.35, 0.45, 0.20])
payment_method = rng.choice(
    ["electronic_check", "mailed_check", "bank_transfer", "credit_card"], n
)

# Build churn probability from a logistic combination of realistic drivers
contract_risk = np.where(contract_type == "month-to-month", 1.0,
                 np.where(contract_type == "one-year", 0.4, 0.1))
logit = (
    -1.2
    + contract_risk * 1.8
    + (support_calls > 2) * 0.9
    + (satisfaction <= 2) * 1.3
    - (tenure / 72) * 1.6
    + (monthly_charges > 90) * 0.5
    + rng.normal(0, 0.6, n)
)
prob_churn = 1 / (1 + np.exp(-logit))
churn = (rng.uniform(0, 1, n) < prob_churn).astype(int)
churn_label = np.where(churn == 1, "Yes", "No")

df_churn = pd.DataFrame({
    "tenure": tenure,
    "monthly_charges": monthly_charges.round(2),
    "total_charges": total_charges.round(2),
    "contract_type": contract_type,
    "support_calls": support_calls,
    "satisfaction": satisfaction,
    "internet_service": internet_service,
    "payment_method": payment_method,
    "churn": churn_label,
})

# Inject a small, controlled fraction of genuine missing values so the
# Data Quality Analyzer has something realistic to detect and report.
missing_idx = rng.choice(n, size=int(0.015 * n), replace=False)
df_churn.loc[missing_idx, "monthly_charges"] = np.nan
missing_idx2 = rng.choice(n, size=int(0.01 * n), replace=False)
df_churn.loc[missing_idx2, "satisfaction"] = np.nan

df_churn.to_csv(DATASETS_DIR / "customer_churn.csv", index=False)
print("customer_churn.csv written:", df_churn.shape)
print("Churn rate:", df_churn["churn"].value_counts(normalize=True).to_dict())
