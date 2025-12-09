import numpy as np
import pandas as pd
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import pickle
import os.path
import logging

np.random.seed(42)
torch.manual_seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='model_training.log'
)
logger = logging.getLogger(__name__)

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


if __name__ == "__main__":


    df = load_data()

    # Discretize income: top 20% = privileged
    df['income_top20'] = (df['annual_income'] >= df['annual_income'].quantile(0.80)).astype(int)

    # Discretize years employed: top 20% = privileged
    df['years_employed_top20'] = (df['years_employed'] >= df['years_employed'].quantile(0.80)).astype(int)

    # Discretize occupation status
    df['student'] = (df['occupation_status'] == 'Student').astype(int)

    threshold_age = range(20, 70)

    results = {}

    for age in threshold_age:
        print(f"Processing for age threshold: {age}")

        # Discretize age: >40 = privileged
        df['age>threshold'] = (df['age'] > age).astype(int)

        # Build feature matrix with one-hot encoding
        X_df = pd.get_dummies(df.drop(columns=['loan_status', 'income_top20', 'age>threshold', 'years_employed_top20', 'student']), drop_first=True)

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

        protected_attributes = ['age>threshold'] # ['income_top20', 'years_employed_top20', 'student']

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


        logger.info("Models need to be trained and saved.")



        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        model = NeuralNetwork(input_dim=X_train.shape[1])
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                
                
                loss = criterion(outputs, labels)


                loss.backward()
                
                optimizer.step()
            if (epoch+1) % 10 == 0:
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        with torch.no_grad():
            y_pred_nn = model(X_test)
            y_pred_cls = (torch.sigmoid(y_pred_nn) >= 0.5).float()
            accuracy_nn = (y_pred_cls.eq(y_test).sum().item()) / y_test.size(0)
            logger.info(f'Accuracy on test set using NN: {accuracy_nn * 100:.2f}%')

        # model saving
        torch.save(model.state_dict(), 'saved_model_nn.pth')

        # Random Forest Classifier
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train_final.values, y_train.numpy().ravel())
        y_pred_rf = classifier.predict(X_test_final.values)

        # model saving
        pickle.dump(classifier, open('saved_model_rf.pkl', 'wb'))

        accuracy_rf = accuracy_score(y_test.numpy(), y_pred_rf)
        logger.info(f'Accuracy on test set using Random Forest: {accuracy_rf * 100:.2f}%')

        logger.info("\nFairness Evaluation Results:\n")
        logger.info(f"Threshold Age: {age}")


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
            
            df_eval['y_pred'] = y_pred_rf

            tpr_priv, tpr_unpriv, diff = equal_opportunity(df_eval, attr, y_pred_rf)
            tpr_per_dif = (1- (tpr_unpriv/tpr_priv)) * 100

            logger.info("Random Forest Equal Opportunity Results:")
            logger.info(f"Percentage: {1 - tpr_per_dif}")
            logger.info(f'Equal Opportunity for {attr}:')
            logger.info(f'  TPR Privileged: {tpr_priv:.4f}')
            logger.info(f'  TPR Unprivileged: {tpr_unpriv:.4f}')
            logger.info(f'  Difference (Priv - Unpriv): {diff:.4f}')
            logger.info('-----------------------------------')

            results[age] = {
                'nn_accuracy': accuracy_nn,
                'rf_accuracy': accuracy_rf,
                'nn_equal_opportunity': (tpr_priv, tpr_unpriv, diff),
                'rf_equal_opportunity': (tpr_priv, tpr_unpriv, diff)
            }
    
    plt.figure(figsize=(10, 6))
    ages = list(results.keys())
    nn_diffs = [results[age]['nn_equal_opportunity'][2] for age in ages]
    nn_accs = [results[age]['nn_accuracy'] for age in ages]

    plt.subplot(1, 2, 1)
    plt.plot(ages, nn_diffs, marker='o')
    plt.xlabel('Age Threshold')
    plt.ylabel('TPR Difference (Priv - Unpriv)')
    plt.title('Age vs Equal Opportunity Difference')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(ages, nn_accs, marker='o', color='orange')
    plt.xlabel('Age Threshold')
    plt.ylabel('NN Accuracy')
    plt.title('Age vs Neural Network Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('images/fairness_results.png')
    plt.show()
                     