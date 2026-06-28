import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from loguru import logger
import sys
sys.path.append("..")
from config import FORECAST_HORIZONS, SEQ_LEN
from features.engineer import load_and_engineer, get_feature_columns

logger.add("logs/lstm_model.log", rotation="1 MB")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class RadiationDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_len],
            self.y[idx + self.seq_len]
        )

class RadiationLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

def train_lstm(df: pd.DataFrame, horizon: int, epochs: int = 50, batch_size: int = 64):
    feature_cols = get_feature_columns(df)
    target_col = f"target_{horizon}h"

    X = df[feature_cols].values
    y = df[target_col].values

    split = int(len(X) * 0.8)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train = scaler_X.fit_transform(X[:split])
    X_test  = scaler_X.transform(X[split:])
    y_train = scaler_y.fit_transform(y[:split].reshape(-1, 1)).flatten()
    y_test  = scaler_y.transform(y[split:].reshape(-1, 1)).flatten()

    train_ds = RadiationDataset(X_train, y_train)
    test_ds  = RadiationDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = RadiationLSTM(input_size=X.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                val_loss += criterion(pred, yb).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss / len(test_loader)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict().copy()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    # Load best weights
    model.load_state_dict(best_state)

    # Evaluate
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy()
            all_preds.extend(pred)
            all_true.extend(yb.numpy())

    # Inverse transform
    preds_orig = scaler_y.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()
    true_orig  = scaler_y.inverse_transform(np.array(all_true).reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(true_orig, preds_orig))
    mae  = mean_absolute_error(true_orig, preds_orig)

    print(f"\n--- LSTM {horizon}h Forecast ---")
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f}")
    logger.info(f"LSTM {horizon}h | RMSE: {rmse:.4f} | MAE: {mae:.4f}")

    # Save
    torch.save(model.state_dict(), f"models/lstm_{horizon}h.pt")
    with open(f"models/lstm_{horizon}h_scalers.pkl", "wb") as f:
        pickle.dump({
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "features": feature_cols,
            "input_size": X.shape[1]
        }, f)

    logger.info(f"Saved lstm_{horizon}h.pt")
    return model, scaler_X, scaler_y, feature_cols

def train_all(epochs: int = 50):
    print("Loading features...")
    df = load_and_engineer()
    if df.empty:
        print("No data. Run omniweb_fetcher.py first.")
        return

    for h in FORECAST_HORIZONS:
        print(f"\nTraining LSTM for {h}h horizon...")
        train_lstm(df, h, epochs=epochs)

def load_lstm(horizon: int):
    with open(f"models/lstm_{horizon}h_scalers.pkl", "rb") as f:
        bundle = pickle.load(f)
    model = RadiationLSTM(input_size=bundle["input_size"]).to(DEVICE)
    model.load_state_dict(torch.load(f"models/lstm_{horizon}h.pt", map_location=DEVICE))
    model.eval()
    return model, bundle["scaler_X"], bundle["scaler_y"], bundle["features"]

def predict_lstm(recent_df: pd.DataFrame, horizon: int) -> float:
    model, scaler_X, scaler_y, features = load_lstm(horizon)
    X = scaler_X.transform(recent_df[features].values)
    X_tensor = torch.tensor(X[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_scaled = model(X_tensor).item()
    pred = scaler_y.inverse_transform([[pred_scaled]])[0][0]
    return float(pred)

if __name__ == "__main__":
    train_all(epochs=50)
