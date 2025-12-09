import numpy as np
import pandas as pd
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, roc_curve, auc, accuracy_score
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    filename='fair_model_training_ROC-curves.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("\n\nFair Model Training Script Started:\n")


np.random.seed(42)
torch.manual_seed(42)

def load_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, encoding="latin1")
    
def load_data():
    script_dir = Path(__file__).resolve().parent
    data_file = script_dir.parent / "data" / "Loan_approval_data_2025.csv"
    if not data_file.exists():
        logger.error(f"CSV not found at: {data_file}")
        sys.exit(1)

    df = load_csv(str(data_file))
    # drop the timestamp column
    if 'customer_id' in df.columns:
        df = df.drop(columns=['customer_id'])
    logger.info(f"Data loaded from {data_file}, shape: {df.shape}")
    return df

def equal_opportunity(df, protected_col, y_pred):
    # True Positive Rate for privileged vs unprivileged
    priv = df[df[protected_col] == 1]
    unpriv = df[df[protected_col] == 0]

    tpr_priv = recall_score(priv['y_true'], priv['y_pred'])
    tpr_unpriv = recall_score(unpriv['y_true'], unpriv['y_pred'])

    return tpr_priv, tpr_unpriv, tpr_priv - tpr_unpriv

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def EO_loss_fn(actual_loss, y_pred_probs, sensitive_attr, labels, lambda_coef=0.1, epsilon=1e-7):
    """
    Computes Equal Opportunity loss using a soft differentiable approximation.
    """
    # 1. Filter only positive ground truth labels (y=1)
    # EO only cares about the True Positive Rate
    pos_mask = (labels == 1).squeeze()

    # Filter predictions and sensitive attributes for y=1 samples
    y_pred_pos = y_pred_probs[pos_mask]
    sens_attr_pos = sensitive_attr[pos_mask]
    
    # Privileged group (sensitive == 1)
    priv_mask = (sens_attr_pos == 1)
    # Soft TPR: Average probability of predicting 1 for the privileged group
    tpr_priv = (y_pred_pos[priv_mask].sum()) / (priv_mask.sum() + epsilon)

    # Unprivileged group (sensitive == 0)
    unpriv_mask = (sens_attr_pos == 0)
    # Soft TPR: Average probability of predicting 1 for the unprivileged group
    tpr_unpriv = (y_pred_pos[unpriv_mask].sum()) / (unpriv_mask.sum() + epsilon)

    # 3. Calculate Penalty (Difference in Soft TPR)
    eo_penalty = torch.abs(tpr_priv - tpr_unpriv)

    return actual_loss + (eo_penalty * lambda_coef)


def plot_group_roc(y_true, y_pred_proba, groups, title, out_path=None):
    """Plot ROC curves for privileged and unprivileged groups.

    y_true, y_pred_proba, and groups should be 1D numpy arrays of the same length.
    If out_path is provided the plot will be saved there, otherwise it will be shown.
    """
    priv = (groups == 1)
    unpriv = (groups == 0)

    # guard against groups with no positive/negative samples
    if priv.sum() == 0 or unpriv.sum() == 0:
        logger.warning("One of the groups has no samples; skipping ROC plot.")
        return

    fpr_p, tpr_p, _ = roc_curve(y_true[priv], y_pred_proba[priv])
    fpr_u, tpr_u, _ = roc_curve(y_true[unpriv], y_pred_proba[unpriv])

    auc_p = auc(fpr_p, tpr_p)
    auc_u = auc(fpr_u, tpr_u)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_p, tpr_p, label=f"Privileged AUC={auc_p:.3f}")
    ax.plot(fpr_u, tpr_u, label=f"Unprivileged AUC={auc_u:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)

    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if out_path:
        # ensure directory exists
        out_dir = Path(out_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved ROC plot to {out_path}")
    else:
        plt.show()



if __name__ == "__main__":


    # sensitive attribute to make fair with respect to
    sensitive_attribute = 'age>25'
    lambda_coef = 0.5
    
    df = load_data()

    # Discretize age: >40 = privileged
    df['age>25'] = (df['age'] > 25).astype(int)

    X_df = pd.get_dummies(df.drop(columns=['loan_status', 'age>25']), drop_first=True)

    # Detect any non-numeric columns left after get_dummies (object dtype)
    obj_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        logger.warning(f'Found non-numeric columns in feature matrix after get_dummies: {obj_cols}')
        # Try to coerce them to numeric; if coercion fails values become NaN
        for c in obj_cols:
            X_df[c] = pd.to_numeric(X_df[c], errors='coerce')

    # Fill NaNs introduced by coercion (or existing) with 0 and ensure float32 dtype
    X_df = X_df.fillna(0)
    try:
        X = X_df.values.astype(np.float32)
    except Exception as e:
        logger.error(f'Failed to cast feature matrix to float32: {e}')
        logger.error('Dtypes of X_df:')
        logger.error(X_df.dtypes.value_counts())
        raise

    X = pd.DataFrame(X, columns=X_df.columns)

    protected_attributes = ['age>25']

    # Save the protected columns for train/test split
    X_protected = df[protected_attributes]
    y = df['loan_status'].values.astype(np.float32)

    # Split both features and protected columns
    X_train_full, X_test_full, X_protected_train, X_protected_test, y_train, y_test = train_test_split(
        X, X_protected, y, test_size=0.25, random_state=42
    )

    feature_names = X_train_full.columns

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_full)
    X_test = scaler.transform(X_test_full)

    X_train_final = pd.DataFrame(X_train, columns=feature_names)
    X_test_final = pd.DataFrame(X_test, columns=feature_names)

    # Convert numpy arrays to torch tensors
    X_train = torch.from_numpy(X_train_final.values.astype(np.float32))
    y_train = torch.from_numpy(y_train).float().view(-1, 1)
    X_test = torch.from_numpy(X_test_final.values.astype(np.float32))
    y_test = torch.from_numpy(y_test).float().view(-1, 1)

    num_epochs = 30
    results = {}
   
    logger.info(f"Training with Fairness Constraint on: {sensitive_attribute}")

    # 1. Prepare the sensitive attribute tensor for training
    # We explicitly grab the column defined in 'sensitive_attribute' variable
    sensitive_train = X_protected_train[sensitive_attribute].values.astype(np.float32)
    sensitive_train = torch.from_numpy(sensitive_train)

    # 2. Update Dataset to include features (X), labels (y), AND sensitive attributes (z)
    train_dataset = TensorDataset(X_train, y_train, sensitive_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = NeuralNetwork(input_dim=X_train.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Increase epochs slightly to allow fairness constraint to converge

    logger.info(f"Training with Fairness Constraint on: {sensitive_attribute}")

    for epoch in range(num_epochs):
        epoch_loss = 0
        # 3. Unpack 3 values now: inputs, labels, AND sensitive_batch
        for inputs, labels, sens_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate standard BCE loss
            actual_loss = criterion(outputs, labels)

            # Calculate Fairness loss
            # We pass 'sens_batch' (the tensor), NOT the string name
            loss = EO_loss_fn(
                actual_loss, 
                torch.sigmoid(outputs), # Pass probabilities, not logits
                sens_batch, 
                labels,
                lambda_coef=lambda_coef # Increased slightly to force effect
            )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch+1) % 5 == 0:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
        
    with torch.no_grad():
        y_pred_nn = model(X_test)
        y_pred_cls = (torch.sigmoid(y_pred_nn) >= 0.5).float()
        accuracy_nn = (y_pred_cls.eq(y_test).sum().item()) / y_test.size(0)
        logger.info(f'Accuracy on test set using NN: {accuracy_nn * 100:.2f}%')

    logger.info("\nFairness Evaluation Results:\n")


    # Fairness evaluation
    for attr in protected_attributes:
        df_eval = X_protected_test.copy()
        df_eval['y_true'] = y_test.numpy()
        df_eval['y_pred'] = y_pred_cls

        tpr_priv, tpr_unpriv, diff = equal_opportunity(df_eval, attr, y_pred_cls)
        tpr_per_dif = (1- (tpr_unpriv/tpr_priv)) * 100
        
    
        logger.info("Neural Network Equal Opportunity Results:")
        logger.info(f"Percentage: {1 - tpr_per_dif}")
        logger.info(f'Equal Opportunity for {attr}:')
        logger.info(f'  TPR Privileged: {tpr_priv:.4f}')
        logger.info(f'  TPR Unprivileged: {tpr_unpriv:.4f}')
        logger.info(f'  Difference (Priv - Unpriv): {diff:.4f}')
        logger.info('-----------------------------------')

    results[num_epochs] = {
        'accuracy': accuracy_nn,
        'tpr_priv': tpr_priv,
        'tpr_unpriv': tpr_unpriv,
        'tpr_diff': diff
    }

    # --- Plot ROC curves by protected group for the neural network ---
    try:
        y_true_np = y_test.numpy().ravel()
        y_pred_proba_np = torch.sigmoid(y_pred_nn).numpy().ravel()
        groups_np = X_protected_test[sensitive_attribute].values

        images_dir = Path(__file__).resolve().parent.parent / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        out_path = images_dir / 'roc_by_group_nn.png'

        plot_group_roc(y_true_np, y_pred_proba_np, groups_np, f'ROC by Group - NN (age>{25})', str(out_path))
    except Exception as e:
        logger.exception(f'Failed to generate ROC plot: {e}')
