import numpy as np
import pandas as pd
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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
    # drop the timestamp column
    if 'person_id' in df.columns:
        df = df.drop(columns=['person_id'])
    return df
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LogisticRegressionModel, self).__init__()
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

    # Build feature matrix with one-hot encoding
    X_df = pd.get_dummies(df.drop(columns=['is_high_risk', 'risk_score']), drop_first=True)

    # Detect any non-numeric columns left after get_dummies (object dtype)
    obj_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        print('Found non-numeric columns in feature matrix after get_dummies:', obj_cols)
        # Try to coerce them to numeric; if coercion fails values become NaN
        for c in obj_cols:
            X_df[c] = pd.to_numeric(X_df[c], errors='coerce')

    # Fill NaNs introduced by coercion (or existing) with 0 and ensure float32 dtype
    X_df = X_df.fillna(0)
    try:
        X = X_df.values.astype(np.float32)
    except Exception as e:
        print('Failed to cast feature matrix to float32:', e)
        print('Dtypes of X_df:')
        print(X_df.dtypes.value_counts())
        raise

    X = pd.DataFrame(X, columns=X_df.columns)
    y = df['is_high_risk'].values.astype(np.float32)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    feature_names = X_train.columns

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.linear_model import RidgeCV

    model_feature_selection = RidgeCV()
    model_feature_selection.fit(X_train, y_train)

    print(f"model score on training data: {model_feature_selection.score(X_train, y_train)}")

    coefs = pd.DataFrame(
        model_feature_selection.coef_.T, columns=["Coefficients"]
    )

    coefs.plot(kind="barh", figsize=(12, 16))
    plt.title("Ridge model")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)
    plt.show()

    # select features with coefficients more than |0.005|
    abs_coefs = np.abs(model_feature_selection.coef_).ravel()
    mask = abs_coefs >= 0.005
    selected_features = feature_names[mask]
    X_train_final = pd.DataFrame(X_train, columns=feature_names) # [selected_features]
    X_test_final = pd.DataFrame(X_test, columns=feature_names) # [selected_features]

    print(f"Selected {len(selected_features)} features out of {len(feature_names)}")
    print(f"Selected features: {selected_features.tolist()}")

    correlations = pd.concat([X, pd.Series(y, name='is_high_risk')], axis=1) \
                   .corr()['is_high_risk'] \
                   .abs() \
                   .sort_values(ascending=False)
    print("\nTop positive correlations:")
    print(correlations[correlations > 0].head(15))

    # Convert numpy arrays to torch tensors
    X_train = torch.from_numpy(X_train_final.values.astype(np.float32))
    y_train = torch.from_numpy(y_train).float().view(-1, 1)
    X_test = torch.from_numpy(X_test_final.values.astype(np.float32))
    y_test = torch.from_numpy(y_test).float().view(-1, 1)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    model = LogisticRegressionModel(input_dim=X_train.shape[1])
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
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred_cls = (torch.sigmoid(y_pred) >= 0.5).float()
        accuracy = (y_pred_cls.eq(y_test).sum().item()) / y_test.size(0)
        print(f'Accuracy on test set using NN: {accuracy * 100:.2f}%')

    # Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train_final.values, y_train.numpy().ravel())
    y_pred = classifier.predict(X_test_final.values)

    accuracy = accuracy_score(y_test.numpy(), y_pred)
    print(f'Accuracy on test set using Random Forest: {accuracy * 100:.2f}%')
