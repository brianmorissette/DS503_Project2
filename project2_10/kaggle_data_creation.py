import kagglehub
import pandas as pd
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")

# Find the CSV file in the dataset
csv_files = list(Path(path).glob("*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV file found in {path}")

df = pd.read_csv(csv_files[0])

# Select only the specified columns
columns = ["LIMIT_BAL", "AGE", "BILL_AMT1", "PAY_AMT1"]
df_subset = df[columns]

# Save to kaggle_credit_dataset.csv with no headers
df_subset.to_csv("kaggle_credit_dataset.csv", index=False, header=False)

print(f"Created kaggle_credit_dataset.csv with {len(df_subset)} rows")
