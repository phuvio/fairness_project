import numpy as np
import pandas as pd
import sys
from pathlib import Path
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric

np.random.seed(42)

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin1")
    
def load_data():
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir.parent / "data" / "Loan_approval_data_2025.csv"
    if not data_file.exists():
        print(f"CSV not found at: {data_file}")
        sys.exit(1)

    df = load_csv(str(data_file))

    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
   
    # Fill missing numeric values with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill missing categorical values with mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

if __name__ == "__main__":
    df = load_data()

    # Prepare label and protected attribute columns
    df['loan_status'] = df['loan_status'].astype(int)

    # Discretize income: top 20% = privileged
    df['income_top20'] = (df['annual_income'] >= df['annual_income'].quantile(0.80)).astype(int)

    # Discretize age: >40 = privileged
    df['age>40'] = (df['age'] > 40).astype(int)

    # Discretize years employed: top 20% = privileged
    df['years_employed_top20'] = (df['years_employed'] >= df['years_employed'].quantile(0.80)).astype(int)

    # Define protected attributes and privileged groups
    protected_attributes = {
        'age>40': 1,
        'income_top20': 1,
        'years_employed_top20': 1,
    }

    # Loop over each protected attribute and compute dataset fairness metrics
    for attr, privileged in protected_attributes.items():
        # Select only protected + label
        dataset = BinaryLabelDataset(
            df=df[[attr, 'loan_status']],
            label_names=['loan_status'],
            protected_attribute_names=[attr]
        )

        metric = BinaryLabelDatasetMetric(
            dataset,
            privileged_groups=[{attr: privileged}],
            unprivileged_groups=[{attr: 0}]
        )

        print(f"====== {attr.upper()} ======")
        print("SPD:", round(metric.statistical_parity_difference(), 4))
        print("DI:", round(metric.disparate_impact(), 4))
        print("Mean Diff:", round(metric.mean_difference(), 4))
        print()
