import io
import requests
import pandas as pd
import yfinance as yf
from pandas_datareader.data import DataReader
from .config import START_DATE, END_DATE, YF_TICKERS, FRED_SERIES, ECB_EURO_10Y_DAILY_KEY, PROCESSED_DIR
from .utils import ensure_datetime_index

def download_yahoo_series() -> pd.DataFrame:
    frames = []
    for name, ticker in YF_TICKERS.items():
        data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
        if data.empty:
            raise ValueError(f"No se pudo descargar {name} ({ticker})")
        col = "Adj Close" if "Adj Close" in data.columns else "Close"
        series = data[col].squeeze("columns")   # <- fuerza a Serie si viene como DataFrame de 1 columna
        frames.append(series.rename(name))
    return ensure_datetime_index(pd.concat(frames, axis=1))


def download_fred_series() -> pd.DataFrame:
    frames = []
    for name, code in FRED_SERIES.items():
        data = DataReader(code, "fred", START_DATE, END_DATE)
        frames.append(data.iloc[:, 0].rename(name))
    return ensure_datetime_index(pd.concat(frames, axis=1))

def download_ecb_series(series_key: str = ECB_EURO_10Y_DAILY_KEY) -> pd.DataFrame:
    url = f"https://data-api.ecb.europa.eu/service/data/YC/{series_key}?startPeriod={START_DATE}&endPeriod={END_DATE}&format=csvdata"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    raw = pd.read_csv(io.StringIO(r.text))
    date_col = "TIME_PERIOD" if "TIME_PERIOD" in raw.columns else raw.columns[0]
    value_col = "OBS_VALUE" if "OBS_VALUE" in raw.columns else raw.columns[-1]
    out = raw[[date_col, value_col]].copy()
    out.columns = ["Date", "EU10Y"]
    out["Date"] = pd.to_datetime(out["Date"])
    out["EU10Y"] = pd.to_numeric(out["EU10Y"], errors="coerce")
    return out.set_index("Date").sort_index()

def align_and_clean_market_data() -> pd.DataFrame:
    yahoo_df = download_yahoo_series()
    fred_df = download_fred_series()
    ecb_df = download_ecb_series()

    df = yahoo_df.join(fred_df, how="left").join(ecb_df, how="left")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.loc[df["EURUSD"].notna()].copy()
    df = df.ffill().dropna().sort_index()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "market_data_daily.csv")
    return df
