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
    data_file = script_dir.parent / "data" / "medical_insurance.csv"
    if not data_file.exists():
        print(f"CSV not found at: {data_file}")
        sys.exit(1)

    df = load_csv(str(data_file))
    
    if 'person_id' in df.columns:
        df = df.drop(columns=['person_id'])
    
    # Fill missing numeric values with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill missing categorical values with mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

if __name__ == "__main__":
    df = load_data()

    categorical_cols = ['education', 'urban_rural']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # Prepare label and protected attribute columns
    df['is_high_risk'] = df['is_high_risk'].astype(int)

    # Convert sex to binary: Male=1 (privileged), Female/Other=0
    df['sex'] = df['sex'].replace({'Male': 1, 'Female': 0, 'Other':0}).astype(int)

    # Discretize income: top 20% = privileged
    df['income_top20'] = (df['income'] >= df['income'].quantile(0.80)).astype(int)

    # Discretize age: >40 = privileged
    df['age>40'] = (df['age'] > 40).astype(int)

    # Define protected attributes and privileged groups
    protected_attributes = {
        'sex': 1,
        'age>40': 1,
        'income_top20': 1,
        'education_Doctorate': 1,
        'education_Masters': 1,
        'urban_rural_Urban': 1
    }

    # Loop over each protected attribute and compute dataset fairness metrics
    for attr, privileged in protected_attributes.items():
        # Select only protected + label
        dataset = BinaryLabelDataset(
            df=df[[attr, 'is_high_risk']],
            label_names=['is_high_risk'],
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
