import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .config import SEQ_LEN, OUTPUT_DIR, SEED, BUSINESS_DAYS_MONTH
from .features import get_feature_columns


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def make_sequences(X, y, dates, seq_len=SEQ_LEN):
    Xs, ys, ds = [], [], []

    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
        ds.append(dates[i])

    return np.array(Xs), np.array(ys), np.array(ds)


def _train_one_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50,
    lr=1e-3,
    batch_size=64,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_state = copy.deepcopy(model.state_dict())
    best_val = np.inf
    patience, patience_left = 8, 8

    for _ in range(epochs):
        model.train()

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1

            if patience_left <= 0:
                break

    model.load_state_dict(best_state)

    return model


def _predict_numpy(model, X):
    model.eval()

    with torch.no_grad():
        preds = model(torch.tensor(X, dtype=torch.float32)).numpy()

    # La volatilidad no puede tomar valores negativos.
    return np.clip(preds, a_min=0.0, a_max=None)


def _permutation_importance_sequence(
    model,
    X_test,
    y_test,
    feature_cols,
    baseline_rmse,
    model_name,
    split_id,
    n_repeats=5,
    seed=SEED,
):
    """
    Calcula importancia por permutación para modelos secuenciales.

    Para cada variable, se permuta la trayectoria completa de esa variable entre
    observaciones del conjunto de prueba. Esto preserva la estructura temporal
    interna de cada secuencia, pero rompe la asociación entre esa variable y el
    target. La importancia se mide como el aumento promedio del RMSE.
    """
    rng = np.random.default_rng(seed + split_id)
    rows = []

    if len(X_test) < 2:
        return rows

    for j, feature in enumerate(feature_cols):
        rmse_increases = []

        for repeat in range(n_repeats):
            X_perm = X_test.copy()
            perm_idx = rng.permutation(len(X_perm))

            # Permuta la trayectoria completa de la variable j entre observaciones.
            X_perm[:, :, j] = X_perm[perm_idx, :, j]

            preds_perm = _predict_numpy(model, X_perm)
            rmse_perm = np.sqrt(mean_squared_error(y_test, preds_perm))

            rmse_increases.append(rmse_perm - baseline_rmse)

        rows.append(
            {
                "split_id": split_id,
                "model": model_name,
                "feature": feature,
                "importance_rmse_increase": float(np.mean(rmse_increases)),
                "importance_rmse_std": float(np.std(rmse_increases)),
                "baseline_rmse": float(baseline_rmse),
                "n_repeats": n_repeats,
            }
        )

    return rows


@dataclass
class DLForecastResult:
    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    importance: pd.DataFrame


def run_walk_forward_dl(features_df, splits, model_type="lstm"):
    feature_cols = get_feature_columns()
    rows, metric_rows, importance_rows = [], [], []

    for k, split in enumerate(splits, start=1):
        train_df = features_df.loc[
            split["train_mask"],
            feature_cols + ["target_vol_1m"],
        ].dropna()

        val_df = features_df.loc[
            split["val_mask"],
            feature_cols + ["target_vol_1m"],
        ].dropna()

        test_df = features_df.loc[
            split["test_mask"],
            feature_cols + ["target_vol_1m"],
        ].dropna()

        # El target de cada fecha utiliza los siguientes 21 retornos.
        # Se eliminan las últimas 21 observaciones de entrenamiento y validación
        # para impedir que sus targets utilicen información del bloque siguiente.
        horizon = BUSINESS_DAYS_MONTH

        if len(train_df) > horizon:
            train_df = train_df.iloc[:-horizon].copy()

        if len(val_df) > horizon:
            val_df = val_df.iloc[:-horizon].copy()

        scaler = StandardScaler()

        X_train_raw = scaler.fit_transform(train_df[feature_cols])
        X_val_raw = scaler.transform(val_df[feature_cols])
        X_test_raw = scaler.transform(test_df[feature_cols])

        y_train_raw = train_df["target_vol_1m"].values
        y_val_raw = val_df["target_vol_1m"].values
        y_test_raw = test_df["target_vol_1m"].values

        X_train, y_train, _ = make_sequences(
            X_train_raw,
            y_train_raw,
            train_df.index,
        )

        X_val, y_val, _ = make_sequences(
            X_val_raw,
            y_val_raw,
            val_df.index,
        )

        X_test, y_test, d_test = make_sequences(
            X_test_raw,
            y_test_raw,
            test_df.index,
        )

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            continue

        input_size = X_train.shape[2]

        model = (
            LSTMRegressor(input_size)
            if model_type.lower() == "lstm"
            else GRURegressor(input_size)
        )

        model = _train_one_model(model, X_train, y_train, X_val, y_val)

        preds = _predict_numpy(model, X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        model_name = model_type.upper()

        metric_rows.append(
            {
                "split_id": k,
                "model": model_name,
                "train_years": "-".join(map(str, split["train_years"])),
                "val_years": "-".join(map(str, split["val_years"])),
                "test_years": "-".join(map(str, split["test_years"])),
                "mae": mae,
                "rmse": rmse,
            }
        )

        importance_rows.extend(
            _permutation_importance_sequence(
                model=model,
                X_test=X_test,
                y_test=y_test,
                feature_cols=feature_cols,
                baseline_rmse=rmse,
                model_name=model_name,
                split_id=k,
                n_repeats=5,
                seed=SEED,
            )
        )

        for dt, y_true, y_pred in zip(d_test, y_test, preds):
            rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "model": model_name,
                    "y_true_vol_1m": float(y_true),
                    "y_pred_vol_1m": float(y_pred),
                    "y_pred_var_95_1m": 1.645 * float(y_pred),
                    "split_id": k,
                }
            )

    forecasts = pd.DataFrame(rows).sort_values("date")
    metrics = pd.DataFrame(metric_rows)
    importance = pd.DataFrame(importance_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    forecasts.to_csv(
        OUTPUT_DIR / f"{model_type.lower()}_forecasts.csv",
        index=False,
    )

    metrics.to_csv(
        OUTPUT_DIR / f"{model_type.lower()}_metrics.csv",
        index=False,
    )

    importance.to_csv(
        OUTPUT_DIR / f"{model_type.lower()}_permutation_importance.csv",
        index=False,
    )

    return DLForecastResult(
        forecasts=forecasts,
        metrics=metrics,
        importance=importance,
    )
