import numpy as np
import pandas as pd
from datetime import datetime

from data.data_loader import DataLoader
from research.market_scanner import MarketScanner
from execution.signal_generator import SignalGenerator
from execution.zscore_calculator import ZScoreCalculator
from execution.backtest_engine import BacktestEngine


def run_pipeline():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
    
    # 1. ROZDZIELENIE OKRESÓW CZASOWYCH
    # Okres In-Sample (Treningowy / Formacja)
    is_start = "2020-01-01"
    is_end = "2021-12-31"

    # Okres Out-of-Sample (Testowy / Egzekucja)
    oos_start = "2022-01-01"
    oos_end = "2023-12-31"

    print(f"Uruchamianie rygorystycznego testu OOS dla {len(tickers)} spółek...")

    # ==========================================
    # 2. POBIERANIE DANYCH
    # ==========================================
    loader = DataLoader(default_interval="1d")
    df_caps = loader.get_market_caps(tickers)
    
    # Pobieramy pełny zakres danych od razu
    df_prices_all = loader.get_prices(tickers, start_date=is_start, end_date=oos_end)

    # Dzielimy macierz cen na zbiór treningowy i testowy
    df_prices_train = df_prices_all.loc[is_start:is_end].copy()
    df_prices_test = df_prices_all.loc[oos_start:oos_end].copy()

    # ==========================================
    # 3. RESEARCH LAYER (Tylko dane In-Sample!)
    # ==========================================
    print("\n--- FAZA 1: TRENING (In-Sample: 2020-2021) ---")
    scanner = MarketScanner(max_pvalue=0.05, min_half_life=1.0, max_half_life=30.0)
    df_ranking = scanner.scan_sector(df_prices_train, df_caps)

    if df_ranking.empty:
        print("Brak stacjonarnych par w okresie treningowym. Zakończenie pracy.")
        return

    best_pair = df_ranking.iloc[0]
    asset_y = best_pair['asset_y']
    asset_x = best_pair['asset_x']
    hl_train = best_pair['half_life']
    beta_train = best_pair['beta']

    print(f"Wyselekcjonowana para: {asset_y} (Y) vs {asset_x} (X)")
    print(f"Parametry z treningu -> Beta: {beta_train:.4f} | Half-Life: {hl_train:.1f} dni")

    # ==========================================
    # 4. EXECUTION LAYER (Tylko dane Out-of-Sample!)
    # ==========================================
    print("\n--- FAZA 2: TEST TRANSAKCYJNY (Out-of-Sample: 2022-2023) ---")
    
    # Do poprawnego wyliczenia kroczącego Z-Score na samym początku okresu testowego
    # dołączamy końcówkę danych treningowych, aby okno kroczące (rolling window) nie miało braków danych
    window_padding = int(np.clip(hl_train * 3.0, 15, 90))
    df_prices_padded = df_prices_all.loc[:oos_end].iloc[-(len(df_prices_test) + window_padding):]

    # Budowa spreadu na nowych cenach przy użyciu ZAMROŻONEJ bety z przeszłości:
    spread_test = np.log(df_prices_padded[asset_y]) - (beta_train * np.log(df_prices_padded[asset_x]))
    df_spread_test = pd.DataFrame({'Date': spread_test.index, 'Spread': spread_test.values})

    # Obliczenie Z-Score
    z_calc = ZScoreCalculator(multiplier=3.0, min_window=15, max_window=90)
    zscore_data = z_calc.calculate(df_spread_test, pair_col='Spread', half_life=hl_train, date_col='Date')

    # Odcinamy sygnały, które przypadają wyłącznie na okres Out-of-Sample
    zscore_data_oos = [item for item in zscore_data if str(item[0]) >= oos_start]

    generator = SignalGenerator(
        entry_threshold=1.5,
        exit_threshold=0.5,
        SL_threshold=3.5,
        max_holding_period=int(hl_train * 3)
    )
    raw_trades = generator.generate(zscore_data_oos)

    if not raw_trades:
        print("Brak sygnałów transakcyjnych w okresie testowym.")
        return

    # ==========================================
    # 5. ACCOUNTING LAYER
    # ==========================================
    engine = BacktestEngine(
        initial_capital=100_000,
        commission_bps=3.0,
        slippage_bps=3.0,
        borrow_rate_annual=0.01,
        allocation_per_trade=0.10
    )

    df_trades_prices = engine.match_execution_prices(
        trades=raw_trades,
        df_open_prices=df_prices_test,
        pair=(asset_y, asset_x)
    )

    # Do wyliczenia PnL przekazujemy zamrożoną betę treningową
    df_results = engine.calculate_trades_pnl(df_trades_prices, beta=beta_train)
    metrics = engine.calculate_performance_metrics(df_results)

    # ==========================================
    # 6. RAPORT KOŃCOWY OUT-OF-SAMPLE
    # ==========================================
    print("\n" + "="*40)
    print("WYNIKI OUT-OF-SAMPLE (REALNY TEST)")
    print("="*40)
    print(f"Liczba transakcji:   {metrics['total_trades']}")
    print(f"Zysk Netto (USD):    ${metrics['total_pnl']:,.2f}")
    print(f"Stopa zwrotu:        {metrics['total_return_pct']:.2f}%")
    print(f"Win Rate:            {metrics['win_rate_pct']:.1f}%")
    print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print("="*40)


if __name__ == "__main__":
    run_pipeline()