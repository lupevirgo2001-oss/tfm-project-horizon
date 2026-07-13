import numpy as np
import pandas as pd
from arch import arch_model

from .config import BUSINESS_DAYS_MONTH, OUTPUT_DIR, Z_95


def ewma_vol_1m(
    returns: pd.Series,
    lam: float = 0.94,
    horizon_days: int = BUSINESS_DAYS_MONTH,
) -> pd.Series:
    """
    Calcula volatilidad EWMA a horizonte mensual.
    Primero estima varianza diaria condicional y luego la escala a 1 mes.
    """
    ret = returns.dropna()
    var = np.zeros(len(ret))
    var[0] = ret.var()

    vals = ret.to_numpy()

    for t in range(1, len(ret)):
        var[t] = lam * var[t - 1] + (1 - lam) * vals[t - 1] ** 2

    vol_1m = np.sqrt(var * horizon_days)

    return pd.Series(vol_1m, index=ret.index, name="ewma_vol_1m")


def garch_forecast_monthly(
    returns: pd.Series,
    horizon_days: int = BUSINESS_DAYS_MONTH,
) -> pd.Series:
    """
    Forecast de volatilidad mensual mediante GARCH(1,1).
    La varianza diaria pronosticada se agrega sobre los próximos 21 días hábiles.
    """
    ret = returns.dropna() * 100.0
    preds, dates = [], []
    min_obs = 252

    for i in range(min_obs, len(ret) - horizon_days):
        train = ret.iloc[:i]

        try:
            model = arch_model(train, vol="Garch", p=1, q=1, mean="Zero")
            res = model.fit(disp="off")
            fcast = res.forecast(horizon=horizon_days, reindex=False)

            monthly_var = fcast.variance.values[-1].sum() / (100.0 ** 2)
            preds.append(np.sqrt(monthly_var))
            dates.append(ret.index[i])

        except Exception:
            preds.append(np.nan)
            dates.append(ret.index[i])

    return pd.Series(preds, index=pd.to_datetime(dates), name="garch_vol_1m")


def _build_long_benchmark_forecasts(vol_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las volatilidades benchmark en el mismo formato de forecasts
    usado por LSTM/GRU, para que puedan entrar a la misma regla de hedge ratio.
    """
    rows = []

    mapping = {
        "ewma_vol_1m": "EWMA",
        "garch_vol_1m": "GARCH",
    }

    for col, model_name in mapping.items():
        if col not in vol_df.columns:
            continue

        tmp = vol_df[[col]].dropna().copy()
        tmp = tmp.reset_index()
        tmp.columns = ["date", "y_pred_vol_1m"]

        tmp["model"] = model_name
        tmp["y_true_vol_1m"] = np.nan
        tmp["y_pred_var_95_1m"] = Z_95 * tmp["y_pred_vol_1m"]
        tmp["split_id"] = np.nan

        rows.append(
            tmp[
                [
                    "date",
                    "model",
                    "y_true_vol_1m",
                    "y_pred_vol_1m",
                    "y_pred_var_95_1m",
                    "split_id",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "model",
                "y_true_vol_1m",
                "y_pred_vol_1m",
                "y_pred_var_95_1m",
                "split_id",
            ]
        )

    return pd.concat(rows, ignore_index=True)


def build_benchmarks(features_df: pd.DataFrame) -> pd.DataFrame:
    ret = features_df["ret_eurusd"].dropna()

    ewma_monthly = ewma_vol_1m(ret)
    garch_monthly = garch_forecast_monthly(ret)

    vol_wide = pd.concat([ewma_monthly, garch_monthly], axis=1)
    benchmark_forecasts = _build_long_benchmark_forecasts(vol_wide)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vol_wide.to_csv(OUTPUT_DIR / "benchmark_vol_forecasts_wide.csv")
    benchmark_forecasts.to_csv(OUTPUT_DIR / "benchmark_vol_forecasts.csv", index=False)

    return benchmark_forecasts
