import pandas as pd

from src.config import TRAIN_YEARS, VAL_YEARS, TEST_YEARS, SEED
from src.utils import set_seed
from src.data_loader import align_and_clean_market_data
from src.features import build_features_targets
from src.montecarlo import simulate_monthly_exposure_for_period
from src.walk_forward import generate_walk_forward_splits
from src.benchmarks import build_benchmarks
from src.models_dl import run_walk_forward_dl
from src.hedging import build_monthly_strategy_signal, backtest_monthly
from src.reporting import compute_monthly_fx_returns, export_reports


def main():
    set_seed(SEED)

    print("1) Descargando y alineando datos...")
    market_df = align_and_clean_market_data()

    print("2) Construyendo features y target...")
    features_df = build_features_targets(market_df)

    print("3) Simulando exposición mensual...")
    months = pd.period_range(
        features_df.index.min().to_period("M"),
        features_df.index.max().to_period("M"),
        freq="M",
    )
    exposure_scenarios, exposure_monthly = simulate_monthly_exposure_for_period(months)

    print("4) Benchmarks clásicos...")
    benchmark_forecasts = build_benchmarks(features_df)

    print("5) Walk-forward...")
    splits = generate_walk_forward_splits(
        features_df.index,
        TRAIN_YEARS,
        VAL_YEARS,
        TEST_YEARS,
    )

    print("6) Entrenando LSTM...")
    lstm_result = run_walk_forward_dl(features_df, splits, model_type="lstm")

    print("7) Entrenando GRU...")
    gru_result = run_walk_forward_dl(features_df, splits, model_type="gru")

    print("8) Señales mensuales...")
    signal_df = pd.concat(
        [
            build_monthly_strategy_signal(lstm_result.forecasts),
            build_monthly_strategy_signal(gru_result.forecasts),
            build_monthly_strategy_signal(benchmark_forecasts),
        ],
        ignore_index=True,
    )

    print("9) Retornos mensuales FX...")
    monthly_fx_returns = compute_monthly_fx_returns(features_df)

    print("10) Backtest...")
    bt = backtest_monthly(
        exposure_monthly,
        signal_df,
        monthly_fx_returns,
        static_hr=0.60,
    )

    print("11) Reportes...")
    export_reports(bt, lstm_result.metrics, gru_result.metrics)

    print("Listo. Revisa data/outputs y reports/.")


if __name__ == "__main__":
    main()
