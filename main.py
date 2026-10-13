import numpy as np
import pandas as pd
import warnings
from datetime import datetime

# Wyciszenie ostrzeżeń systemowych
warnings.filterwarnings("ignore")

import config

from data.data_loader import DataLoader
from research.market_scanner import MarketScanner
from research.basket_scanner import BasketScanner
from research.cointegration_math import CointegrationMath
from execution.signal_generator import SignalGenerator
from execution.zscore_calculator import ZScoreCalculator
from execution.backtest_engine import BacktestEngine


def run_pairs_pipeline(df_prices_train, df_prices_test, df_prices_all, df_caps, is_start, oos_start, oos_end):
    print("\n" + "="*50)
    print("🚀 URUCHAMIANIE STRATEGII: PAIRS TRADING (2 NOGI)")
    print("="*50)
    
    scanner = MarketScanner(**config.SCANNER_PARAMS)
    df_ranking = scanner.scan_sector(df_prices_train, df_caps)

    if df_ranking.empty:
        print("Brak stacjonarnych par w okresie treningowym.")
        return

    best_pair = df_ranking.iloc[0]
    asset_y = best_pair['asset_y']
    asset_x = best_pair['asset_x']
    hl_train = best_pair['half_life']
    beta_train = best_pair['beta']

    print(f"Wyselekcjonowana para: {asset_y} (Y) vs {asset_x} (X) | Beta: {beta_train:.4f} | HL: {hl_train:.1f} dni")
    
    window_padding = int(np.clip(hl_train * 3.0, 15, 90))
    df_prices_padded = df_prices_all.loc[:oos_end].iloc[-(len(df_prices_test) + window_padding):]

    spread_test = np.log(df_prices_padded[asset_y]) - (beta_train * np.log(df_prices_padded[asset_x]))
    df_spread_test = pd.DataFrame({'Date': spread_test.index, 'Spread': spread_test.values})

    z_calc = ZScoreCalculator(**config.ZSCORE_PARAMS)
    zscore_data = z_calc.calculate(df_spread_test, pair_col='Spread', half_life=hl_train, date_col='Date')
    zscore_data_oos = [item for item in zscore_data if str(item[0]) >= oos_start]

    signal_params = config.SIGNAL_PARAMS.copy()
    signal_params['max_holding_period'] = int(hl_train * 3)
    generator = SignalGenerator(**signal_params)
    raw_trades = generator.generate(zscore_data_oos)

    if not raw_trades:
        print("Brak sygnałów transakcyjnych dla par w OOS.")
        return

    engine = BacktestEngine(**config.ENGINE_PARAMS)
    df_trades_prices = engine.match_execution_prices(
        trades=raw_trades,
        df_open_prices=df_prices_test,
        pair=(asset_y, asset_x)
    )

    df_results = engine.calculate_trades_pnl(df_trades_prices, beta=beta_train)
    metrics = engine.calculate_performance_metrics(df_results)

    print("\n--- WYNIKI OOS (PAIRS TRADING) ---")
    print(f"Liczba transakcji:   {metrics['total_trades']}")
    print(f"Zysk Netto (USD):    ${metrics['total_pnl']:,.2f}")
    print(f"Stopa zwrotu:        {metrics['total_return_pct']:.2f}%")
    print(f"Win Rate:            {metrics['win_rate_pct']:.1f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")


def run_basket_pipeline(df_prices_train, df_prices_test, df_prices_all, is_start, oos_start, oos_end):
    print("\n" + "="*50)
    print("🚀 URUCHAMIANIE STRATEGII: JOHANSEN BASKET (N NÓG)")
    print("="*50)
    
    scanner = BasketScanner(basket_size=3, significance_level=0)
    df_ranking = scanner.scan_sector(df_prices_train)

    if df_ranking.empty:
        print("Brak kointegrowanych koszyków w okresie treningowym.")
        return

    best_basket_row = df_ranking.iloc[0]
    basket_tickers = list(best_basket_row['basket'])
    weights = {k: float(np.real(v)) for k, v in best_basket_row['weights'].items()}
    
    log_train = np.log(df_prices_train[basket_tickers])
    spread_train = sum(weights[t] * log_train[t] for t in basket_tickers).to_numpy()
    
    hl_res = CointegrationMath.calculate_half_life(spread_train)
    hl_train = hl_res.half_life if (1.0 <= hl_res.half_life <= 60.0) else 20.0

    formatted_weights = {k: round(v, 4) for k, v in weights.items()}
    print(f"Wybrany koszyk: {basket_tickers} | Wagi: {formatted_weights} | HL: {hl_train:.1f} dni")

    window_padding = int(np.clip(hl_train * 3.0, 15, 90))
    df_prices_padded = df_prices_all.loc[:oos_end].iloc[-(len(df_prices_test) + window_padding):]

    log_padded = np.log(df_prices_padded[basket_tickers])
    spread_test_series = sum(weights[t] * log_padded[t] for t in basket_tickers)
    df_spread_test = pd.DataFrame({'Date': spread_test_series.index, 'Spread': spread_test_series.values})

    z_calc = ZScoreCalculator(**config.ZSCORE_PARAMS)
    zscore_data = z_calc.calculate(df_spread_test, pair_col='Spread', half_life=hl_train, date_col='Date')
    zscore_data_oos = [item for item in zscore_data if str(item[0]) >= oos_start]

    signal_params = config.SIGNAL_PARAMS.copy()
    signal_params['max_holding_period'] = int(hl_train * 3)
    generator = SignalGenerator(**signal_params)
    raw_trades = generator.generate(zscore_data_oos)

    if not raw_trades:
        print("Brak sygnałów transakcyjnych dla koszyka w OOS.")
        return

    engine = BacktestEngine(**config.ENGINE_PARAMS)
    df_trades_prices = engine.match_basket_execution_prices(
        trades=raw_trades,
        df_open_prices=df_prices_test,
        tickers=basket_tickers
    )

    df_results = engine.calculate_basket_trades_pnl(df_trades_prices, weights=weights)
    metrics = engine.calculate_performance_metrics(df_results)

    print("\n--- WYNIKI OOS (JOHANSEN BASKET) ---")
    print(f"Liczba transakcji:   {metrics['total_trades']}")
    print(f"Zysk Netto (USD):    ${metrics['total_pnl']:,.2f}")
    print(f"Stopa zwrotu:        {metrics['total_return_pct']:.2f}%")
    print(f"Win Rate:            {metrics['win_rate_pct']:.1f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")


def main():
    EXECUTION_MODE = 'both'  # 'pairs', 'basket' lub 'both'
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
    is_start, is_end = "2020-01-01", "2021-12-31"
    oos_start, oos_end = "2022-01-01", "2023-12-31"

    print(f"=== INICJALIZACJA SYSTEMU STAT-ARB (Tryb: {EXECUTION_MODE.upper()}) ===")
    
    loader = DataLoader(**config.DATALOADER_PARAMS)
    df_prices_all = loader.get_prices(tickers, start_date=is_start, end_date=oos_end)
    df_prices_train = df_prices_all.loc[is_start:is_end].copy()
    df_prices_test = df_prices_all.loc[oos_start:oos_end].copy()
    
    df_caps = None
    if EXECUTION_MODE in ['pairs', 'both']:
        df_caps = loader.get_market_caps(tickers)

    if EXECUTION_MODE in ['pairs', 'both']:
        run_pairs_pipeline(df_prices_train, df_prices_test, df_prices_all, df_caps, is_start, oos_start, oos_end)

    if EXECUTION_MODE in ['basket', 'both']:
        run_basket_pipeline(df_prices_train, df_prices_test, df_prices_all, is_start, oos_start, oos_end)


if __name__ == "__main__":
    main()