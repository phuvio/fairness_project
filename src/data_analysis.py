import sys
import glob
from pathlib import Path
import pandas as pd

def find_bangladesh_csv():
    # Resolve path relative to this script so behavior doesn't depend on cwd
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir.parent / "data" / "bangladesh_renowned_university_student_Mental_health.csv"
    return str(data_file) if data_file.exists() else None

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin1")

def find_panic_column(df):
    for c in df.columns:
        lc = c.lower()
        if "panic" in lc:
            return c
    return None

def find_depression_column(df):
    for c in df.columns:
        lc = c.lower()
        if "depress" in lc:
            return c
    return None

def find_treatment_column(df):
    for c in df.columns:
        lc = c.lower()
        # match on common keywords around seeking treatment
        if "treat" in lc or "specialist" in lc or "seek" in lc:
            return c
    return None




def main():
    path = find_bangladesh_csv()
    if not path:
        print("No 'bangladesh' CSV found in the current directory tree.")
        sys.exit(1)

    df = load_csv(path)
    col = find_panic_column(df)
    if not col:
        print("No column containing 'panic' found. Available columns:", ", ".join(df.columns))
        sys.exit(1)

    props = df[col].value_counts(dropna=False, normalize=True) * 100
    print(f"File: {path}")
    print(f"Column: {col}")
    print("Proportions (percent):")
    print(props.map(lambda x: f"{x:.2f}%").to_string())
    
    col = find_depression_column(df)
    if not col:
        print("No column containing 'depression' found. Available columns:", ", ".join(df.columns))
        sys.exit(1)

    props = df[col].value_counts(dropna=False, normalize=True) * 100
    print(f"File: {path}")
    print(f"Column: {col}")
    print("Proportions (percent):")
    print(props.map(lambda x: f"{x:.2f}%").to_string())

    col = find_treatment_column(df)
    if not col:
        print("No column containing 'treatment' found. Available columns:", ", ".join(df.columns))
        sys.exit(1)

    props = df[col].value_counts(dropna=False, normalize=True) * 100
    print(f"File: {path}")
    print(f"Column: {col}")
    print("Proportions (percent):")
    print(props.map(lambda x: f"{x:.2f}%").to_string())


if __name__ == "__main__":
    main()