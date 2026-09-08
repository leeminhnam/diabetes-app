import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Import model from current package
from model import DeeperMLP

# Cố định seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# backend/ is 2 levels up
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_PATH = os.path.join(BASE_DIR, "diabetes.csv")

# 1. Load data
data = pd.read_csv(DATA_PATH)

# 2. Preprocess exactly like XGBoost
X_df = data.drop('diabetes', axis=1)
y = data['diabetes'].values.astype(np.int64)

X_df = pd.get_dummies(X_df, columns=['gender', 'smoking_history'], drop_first=True)
model_columns = list(X_df.columns)
X_df = X_df.astype(float)

# Load existing scaler to ensure EXACT match with XGBoost (now from CURRENT_DIR)
scaler = joblib.load(os.path.join(CURRENT_DIR, "scaler_diabetes.pkl"))
X_scaled = scaler.transform(X_df)

# Train/Val/Test split
N = len(X_scaled)
indices = np.random.permutation(N)
train_end = int(0.70 * N)
val_end = int(0.85 * N)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]

X_train, y_train = X_scaled[train_idx], y[train_idx]
X_val, y_val = X_scaled[val_idx], y[val_idx]

# To Tensor
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Init Model
model = DeeperMLP(input_dim=len(model_columns))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train briefly
for epoch in range(20):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

# Save model locally in app/mlp/
torch.save(model.state_dict(), os.path.join(CURRENT_DIR, "mlp_model.pth"))
print(f"Model trained and saved with {len(model_columns)} features to {os.path.join(CURRENT_DIR, 'mlp_model.pth')}.")
