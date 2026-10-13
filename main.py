# Główny skrypt spinający moduły w spójny proces
import pandas as pd
from datetime import datetime

#Importy modułow:
import config
from data.data_loader import DataLoader
from research.market_scanner import PairAnalysisResult, MarketScanner
from research.cointegration_math import CointegrationMath
from execution.signal_generator import SignalGenerator
from execution.zscore_calculator import ZScoreCalculator
from execution.backtest_engine import BacktestEngine

def run_pipeline():
    # ==========================================
    # 1. KONFIGURACJA SYSTEMU
    # ==========================================
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'] #koszyk testowy
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    print(f"Uruchamianie systemu dla {len(tickers)} spółek...")

    # ==========================================
    # 2. POBIERANIE DANYCH (Data Layer)
    # ==========================================
    loader = DataLoader(**config.DATALOADER_PARAMS)
    print("Pobieranie kapitalizacji rynkowej...")
    df_caps = loader.get_market_caps(tickers)
    print(df_caps)
    print("Pobieranie historycznych cen akcji...")
    df_prices = loader.get_prices(tickers, start_date, end_date)

    # ==========================================
    # 3. SKANOWANIE RYNKU (Research Layer)
    # ==========================================
    scanner = MarketScanner(**config.SCANNER_PARAMS)
    print("Skanowanie kointegracji i wyliczanie Half-Life...")
    df_ranking = scanner.scan_sector(df_prices, df_caps)
    
    if df_ranking.empty:
        print("Brak stacjonarnych par w koszyku. Zakończenie pracy.")
        return

    # Pobieramy najlepszą parę (index 0 po posortowaniu)
    best_pair_data = df_ranking.iloc[0]
    asset_y = best_pair_data['asset_y']
    asset_x = best_pair_data['asset_x']
    hl = best_pair_data['half_life']
    beta = best_pair_data['beta']
    print(f"Najlepsza para: {asset_y} (Y) vs {asset_x} (X) | Half-Life: {hl:.1f} dni")

    # ==========================================
    # 4. GENEROWANIE SYGNAŁÓW (Execution Layer)
    # ==========================================
    # Liczymy najpierw aktualny spread dla tej pary: Y - (beta * X + alpha)
    # Aby to zrobić bez pełnego OLS (bo mamy już betę), uprośćmy: Spread = Y - beta * X
    spread_series = np.log(df_prices[asset_y]) - (beta * np.log(df_prices[asset_x]))
    df_spread = pd.DataFrame({'Date': df_prices.index, 'Spread': spread_series.values})

    print("Obliczanie dynamicznego Z-Score...")
    z_calc = ZScoreCalculator(**config.ZSCORE_PARAMS)
    zscore_data = z_calc.calculate(df_spread, pair_col='Spread', half_life=hl, date_col='Date')

    print("Generowanie sygnałów transakcyjnych...")
    # Używamy poszerzonych progów z Z-Score
    generator = SignalGenerator(config.SIGNAL_PARAMS)
    raw_trades = generator.generate(zscore_data)

    if not raw_trades:
        print("Brak wygenerowanych transakcji dla wybranej pary.")
        return

    # ==========================================
    # 5. BACKTEST I KOSZTY (Accounting Layer)
    # ==========================================
    print("Księgowanie transakcji i kalkulacja PnL...")
    engine = BacktestEngine(config.ENGINE_PARAMS)

    # Krok A: Mapowanie cen otwarcia (Next Open)
    df_trades_prices = engine.match_execution_prices(
        trades=raw_trades, 
        df_open_prices=df_prices, # Do urealnienia egzekucji powinniśmy tu podać ceny Open, ale do testu log_prices wystarczą
        pair=(asset_y, asset_x)
    )

    # Krok B: Kalkulacja PnL Netto
    df_results = engine.calculate_trades_pnl(df_trades_prices, beta)

    # Krok C: Metryki portfelowe
    metrics = engine.calculate_performance_metrics(df_results)

    # ==========================================
    # 6. RAPORT KOŃCOWY
    # ==========================================
    print("\n" + "="*40)
    print("WYNIKI SYSTEMU STAT-ARB")
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
    import numpy as np # Importowane tutaj dla lokalnego wyliczenia spreadu
    run_pipeline()