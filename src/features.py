import numpy as np
import pandas as pd

from .config import BUSINESS_DAYS_MONTH, PROCESSED_DIR, Z_95


def _safe_log_return(series: pd.Series) -> pd.Series:
    """
    Calcula retornos logarítmicos solo cuando los valores son positivos, esto
    evita errores cuando alguna serie tiene valores en cero, negativos o faltantes
    """
    series = series.astype(float)
    valid = series.where(series > 0)
    return np.log(valid / valid.shift(1))


def build_features_targets(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Retornos principales
    data["ret_eurusd"] = _safe_log_return(data["EURUSD"])

    # Variables del EUR/USD
    data["vol_21"] = data["ret_eurusd"].rolling(BUSINESS_DAYS_MONTH).std()
    data["ma_21"] = data["EURUSD"].rolling(BUSINESS_DAYS_MONTH).mean()
    data["ma_ratio_21"] = data["EURUSD"] / data["ma_21"] - 1.0
    data["momentum_21"] = data["EURUSD"] / data["EURUSD"].shift(BUSINESS_DAYS_MONTH) - 1.0

    # Variables externas
    data["ret_vix"] = _safe_log_return(data["VIX"])
    data["ret_dxy"] = _safe_log_return(data["DXY"])

    # WTI puede tener valores negativos en ciertos periodos; por eso se usa variación porcentual.
    data["ret_wti"] = data["WTI"].astype(float).pct_change()

    # Diferencial de tasas 10Y
    data["spread_10y"] = data["US10Y"] - data["EU10Y"]

    # Target: volatilidad realizada futura a 1 mes
    # Se calcula como la desviación estándar de los próximos 21 retornos diarios,
    # escalada a horizonte mensual multiplicando por sqrt(21).
    ret = data["ret_eurusd"].to_numpy(dtype=float)
    future_vol_1m = []
    n = len(data)
    h = BUSINESS_DAYS_MONTH

    for i in range(n):
        start = i + 1
        end = i + 1 + h

        if end <= n:
            window = ret[start:end]
            window = window[np.isfinite(window)]

            if len(window) >= 2:
                vol_daily = np.std(window, ddof=1)
                vol_monthly = vol_daily * np.sqrt(h)
                future_vol_1m.append(vol_monthly)
            else:
                future_vol_1m.append(np.nan)
        else:
            future_vol_1m.append(np.nan)

    data["target_vol_1m"] = future_vol_1m
    data["target_var_95_1m"] = Z_95 * data["target_vol_1m"]

    cols = [
        "EURUSD", "VIX", "DXY", "WTI", "US10Y", "EU10Y",
        "ret_eurusd", "vol_21", "ma_ratio_21", "momentum_21",
        "ret_vix", "ret_dxy", "ret_wti", "spread_10y",
        "target_vol_1m", "target_var_95_1m"
    ]

    out = data[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    out.to_csv(PROCESSED_DIR / "features_targets_daily.csv")
    return out


def get_feature_columns():
    return [
        "ret_eurusd", "vol_21", "ma_ratio_21", "momentum_21",
        "ret_vix", "ret_dxy", "ret_wti", "US10Y", "EU10Y", "spread_10y"
    ]
