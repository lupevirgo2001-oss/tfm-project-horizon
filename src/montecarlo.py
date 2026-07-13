import numpy as np
import pandas as pd
from .config import BUSINESS_DAYS_MONTH, MC_N_SIMS, MC_MEAN_DAILY_INFLOW, MC_MEAN_DAILY_OUTFLOW, MC_VOL_INFLOW, MC_VOL_OUTFLOW, MC_CORR, SEED, PROCESSED_DIR

def simulate_monthly_exposure_for_period(months: pd.PeriodIndex,
                                         n_sims: int = MC_N_SIMS,
                                         mean_inflow: float = MC_MEAN_DAILY_INFLOW,
                                         mean_outflow: float = MC_MEAN_DAILY_OUTFLOW,
                                         vol_inflow: float = MC_VOL_INFLOW,
                                         vol_outflow: float = MC_VOL_OUTFLOW,
                                         corr: float = MC_CORR,
                                         seed: int = SEED):
    rng = np.random.default_rng(seed)
    cov = np.array([[vol_inflow**2, corr*vol_inflow*vol_outflow],
                    [corr*vol_inflow*vol_outflow, vol_outflow**2]])

    scen_rows, base_rows = [], []
    for month in months:
        seasonality = 1.10 if month.month in [3, 6, 9, 12] else 1.0
        mean_in = mean_inflow * seasonality
        mean_out = mean_outflow * seasonality

        draws = rng.multivariate_normal([0.0, 0.0], cov, size=(n_sims, BUSINESS_DAYS_MONTH))
        inflows = mean_in * np.exp(draws[:, :, 0] / np.sqrt(252))
        outflows = mean_out * np.exp(draws[:, :, 1] / np.sqrt(252))
        monthly_exposure = (inflows - outflows).sum(axis=1)

        q05, q50, q95 = np.percentile(monthly_exposure, [5, 50, 95])
        base_rows.append({
            "month": str(month),
            "mean_exposure": monthly_exposure.mean(),
            "median_exposure": q50,
            "p05_exposure": q05,
            "p95_exposure": q95,
            "base_exposure": q50,
        })

        for sim_id, exp in enumerate(monthly_exposure):
            scen_rows.append({"month": str(month), "sim_id": sim_id, "monthly_exposure": exp})

    scenarios = pd.DataFrame(scen_rows)
    base = pd.DataFrame(base_rows)
    scenarios.to_csv(PROCESSED_DIR / "monthly_exposure_scenarios.csv", index=False)
    base.to_csv(PROCESSED_DIR / "monthly_exposure_base.csv", index=False)
    return scenarios, base
