import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin1")

def plot_age_threshold(df, ax, threshold_min=19, threshold_max=30):
    """
    Plot percentage of datapoints as threshold increases from threshold_min to threshold_max.
    For each threshold value t, calculate percentage of datapoints >= t.
    """
    age_col = None
    for c in df.columns:
        if c.lower() == 'age':
            age_col = c
            break
    
    if age_col is None:
        print("Warning: 'age' column not found. Skipping age threshold plot.")
        return
    
    # Filter out NaN values
    age_data = df[age_col].dropna()
    total = len(age_data)
    
    if total == 0:
        print("Warning: No age data available. Skipping age threshold plot.")
        return
    
    # Calculate percentage of datapoints >= threshold for each threshold value
    thresholds = list(range(threshold_min, threshold_max + 1))
    percentages = []
    
    for threshold in thresholds:
        count_above = (age_data >= threshold).sum()
        pct = (count_above / total * 100)
        percentages.append(pct)
    
    # Create line plot
    ax.plot(thresholds, percentages, marker='o', linewidth=2, markersize=6, color='#3498db', markerfacecolor='#2ecc71')
    ax.set_xlabel('Age Threshold', fontsize=11)
    ax.set_ylabel('Percentage of Datapoints (%)', fontsize=11)
    ax.set_title(f'Datapoints >= Age Threshold (Range {threshold_min}–{threshold_max})', fontsize=12, fontweight='bold')
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on points
    for t, pct in zip(thresholds, percentages):
        ax.text(t, pct + 2, f'{pct:.1f}%', ha='center', fontsize=9)

def main():
    # Resolve data and image paths relative to this script's location so the
    # script works regardless of the current working directory.
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir.parent / "data" / "Loan_approval_data_2025.csv"
    if not data_file.exists():
        print(f"CSV not found at: {data_file}")
        sys.exit(1)

    df = load_csv(str(data_file))
    # drop the timestamp column
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    
    numerical_cols = df.select_dtypes(include=['number']).columns
    print("Numerical columns and their basic statistics:")
    print(df[numerical_cols].describe().to_string())

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    print("\nCategorical columns and their value counts:")
    for col in categorical_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts(dropna=False).to_string())


    # Combine all plots in a 3-column grid
    # +1 for the age threshold plot
    all_cols = list(numerical_cols) + list(categorical_cols)
    n_cols = len(all_cols) + 1  # +1 for age threshold plot
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
    
    # Plot age threshold distribution
    age_threshold_idx = len(numerical_cols) + len(categorical_cols)
    plot_age_threshold(df, axes[age_threshold_idx], threshold_min=19, threshold_max=30)
    
    # Hide any unused subplots
    for i in range(n_cols, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    # ensure images directory exists (relative to project root)
    images_dir = script_dir.parent / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    # plt.show()
    plt.savefig(str(images_dir / "data_overview.pdf"))
    
    # Save the age threshold plot separately as PNG
    fig_age, ax_age = plt.subplots(figsize=(10, 6))
    plot_age_threshold(df, ax_age, threshold_min=19, threshold_max=30)
    plt.tight_layout()
    plt.savefig(str(images_dir / "age_threshold_distribution.png"), dpi=150, bbox_inches='tight')
    plt.close(fig_age)

if __name__ == "__main__":
    main()