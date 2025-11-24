import sys
import glob
import pandas as pd

def find_bangladesh_csv():
    matches = glob.glob("../data/bangladesh_renowned_university_student_Mental_health.csv", recursive=True)
    return matches[0] if matches else None

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin1")

def find_panic_column(df):
    keys = [c for c in df.columns if "do you have panic attack?" in c.lower()]
    return keys[0] if keys else None

def find_depression_column(df):
    keys = [c for c in df.columns if "do you have depression?" in c.lower()]

    print(df.value_counts())

    return keys[0] if keys else None

def find_treatment_column(df):
    keys = [c for c in df.columns if "did you seek any specialist for a treatment?" in c.lower()]
    return keys[0] if keys else None




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