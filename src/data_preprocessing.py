import sys
import glob
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin1")

def main():
    # Resolve data and image paths relative to this script's location so the
    # script works regardless of the current working directory.
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir.parent / "data" / "bangladesh_renowned_university_student_Mental_health.csv"
    if not data_file.exists():
        print(f"CSV not found at: {data_file}")
        sys.exit(1)

    df = load_csv(str(data_file))
    # drop the timestamp column
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    numerical_cols = df.select_dtypes(include=['number']).columns
    print("Numerical columns and their basic statistics:")
    print(df[numerical_cols].describe().to_string())

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    print("\nCategorical columns and their value counts:")
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts(dropna=False).to_string())


    # Combine all plots in a 3-column grid
    all_cols = list(numerical_cols) + list(categorical_cols)
    n_cols = len(all_cols)
    n_rows = (n_cols + 2) // 3  # Calculate rows needed for 3 columns
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten()  # Flatten to easily iterate
    
    # Plot numerical columns as histograms
    for i, col in enumerate(numerical_cols):
        axes[i].hist(df[col].dropna(), bins=30)
        axes[i].set_title(f'Histogram of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(False)
    
    # Plot categorical columns as bar plots
    for i, col in enumerate(categorical_cols, start=len(numerical_cols)):
        df[col].value_counts().plot(kind='bar', ax=axes[i])
        axes[i].set_title(f'Bar plot of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].grid(False)
        axes[i].tick_params(axis='x', rotation=45)
    
    # Hide any unused subplots
    for i in range(n_cols, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    # ensure images directory exists (relative to project root)
    images_dir = script_dir.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    # plt.show()
    plt.savefig(str(images_dir / "data_overview.pdf"))

if __name__ == "__main__":
    main()