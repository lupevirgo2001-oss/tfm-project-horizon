import pandas as pd

def max_drawdown(series: pd.Series) -> float:
    cum = series.cumsum()
    roll_max = cum.cummax()
    dd = cum - roll_max
    return float(dd.min())

def summarize_backtest(bt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, grp in bt.groupby("strategy"):
        pnl = grp["net_pnl"]
        residual = grp["residual_exposure"]
        rows.append({
            "strategy": strategy,
            "mean_net_pnl": pnl.mean(),
            "std_net_pnl": pnl.std(ddof=1),
            "cum_net_pnl": pnl.sum(),
            "residual_exposure_std": residual.std(ddof=1),
            "max_drawdown": max_drawdown(pnl.fillna(0)),
        })
    return pd.DataFrame(rows).sort_values("cum_net_pnl", ascending=False)
