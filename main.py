import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional

# Wyciszenie ostrzeżeń systemowych
warnings.filterwarnings("ignore")

import config

from data.data_loader import DataLoader
from research.market_scanner import MarketScanner
from research.market_scanner import BasketScanner
from research.cointegration_math import CointegrationMath
from execution.signal_generator import SignalGenerator
from execution.zscore_calculator import ZScoreCalculator
from execution.backtest_engine import BacktestEngine
from visualization.plotter import StrategyPlotter


def generate_walk_forward_windows(
    oos_start: str,
    oos_end: str,
    train_months: int,
    test_months: int
) -> List[Tuple[str, str, str, str]]:
    """
    Generuje listę krotek (train_start, train_end, test_start, test_end)
    dla kroczącej optymalizacji Walk-Forward.
    """
    windows = []
    current_test_start = pd.to_datetime(oos_start)
    final_oos_end = pd.to_datetime(oos_end)

    while current_test_start <= final_oos_end:
        current_test_end = current_test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)
        if current_test_end > final_oos_end:
            current_test_end = final_oos_end

        current_train_end = current_test_start - pd.Timedelta(days=1)
        current_train_start = current_test_start - pd.DateOffset(months=train_months)

        windows.append((
            current_train_start.strftime("%Y-%m-%d"),
            current_train_end.strftime("%Y-%m-%d"),
            current_test_start.strftime("%Y-%m-%d"),
            current_test_end.strftime("%Y-%m-%d")
        ))

        current_test_start = current_test_start + pd.DateOffset(months=test_months)

    return windows


def run_pairs_walk_forward(
    df_prices_all: pd.DataFrame,
    df_caps: pd.DataFrame,
    windows: List[Tuple[str, str, str, str]],
    top_k_pairs: int = 3  # <--- DODANO: parametr określający maksymalną liczbę par w portfelu
) -> Optional[Dict[str, Any]]:
    print("\n" + "="*65)
    # <--- ZMIANA: aktualizacja nagłówka w konsoli
    print(f"🚀 WALK-FORWARD: PAIRS TRADING (PORTFEL TOP {top_k_pairs} PAR)")
    print("="*65)
    
    scanner = MarketScanner(**config.SCANNER_PARAMS)
    z_calc = ZScoreCalculator(**config.ZSCORE_PARAMS)
    engine = BacktestEngine(**config.ENGINE_PARAMS)

    # <--- DODANO: zapamiętujemy bazową alokację (np. 10%), aby dzielić ją między aktywne pary
    base_allocation = config.ENGINE_PARAMS.get("allocation_per_trade", 0.10)
    
    all_trades_dfs = []
    all_zscore_oos = []

    for idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows, 1):
        df_train = df_prices_all.loc[tr_start:tr_end].copy()
        df_test = df_prices_all.loc[te_start:te_end].copy()

        if df_test.empty:
            continue

        df_ranking = scanner.scan_sector(df_train, df_caps)

        if df_ranking.empty:
            print(f"[Okno {idx:02d} | {te_start} -> {te_end}] Brak stacjonarnej pary w treningu.")
            for dt in df_test.index:
                all_zscore_oos.append((dt, 0.0))
            continue
        
        # <--- ZMIANA: zamiast df_ranking.iloc[0], bierzemy do TOP K najlepszych par
        selected_pairs = df_ranking.head(top_k_pairs)

        # <--- DODANO: dzielimy budżet transakcji równo pomiędzy zakwalifikowane pary w tym oknie
        engine.allocation_per_trade = base_allocation / len(selected_pairs)

        # <--- DODANO: listy pomocnicze do zebrania transakcji i opisów ze wszystkich par w danym oknie
        window_trades_list = []
        window_pairs_Summary = []
        
        # <--- DODANO PĘTLĘ: iterujemy po każdej parze z TOP K w bieżącym oknie
        for pair_idx, (_, best_pair) in enumerate(selected_pairs.iterrows()):
            asset_y = best_pair['asset_y']
            asset_x = best_pair['asset_x']
            hl_train = float(best_pair['half_life'])
            beta_train = float(best_pair['beta'])

            # Padding z końca okresu treningowego pod okno kroczące Z-Score
            window_padding = int(np.clip(hl_train * 3.0, 15, 90))
            df_Subset_upto_test = df_prices_all.loc[:te_end]
            df_prices_padded = df_Subset_upto_test.iloc[-(len(df_test) + window_padding):]

            spread_test = np.log(df_prices_padded[asset_y]) - (beta_train * np.log(df_prices_padded[asset_x]))
            df_spread_test = pd.DataFrame({'Date': spread_test.index, 'Spread': spread_test.values})

            zscore_data = z_calc.calculate(df_spread_test, pair_col='Spread', half_life=hl_train, date_col='Date')
            zscore_window_oos = [item for item in zscore_data if te_start <= str(item[0]) <= te_end]

            # <--- ZMIANA: Z-Score zapisujemy do wykresu tylko dla 1. pary (pair_idx == 0),
            # aby oś czasu na wykresie StrategyPlotter nie miała zduplikowanych dat!
            if pair_idx == 0:
                all_zscore_oos.extend(zscore_window_oos)

            signal_params = config.SIGNAL_PARAMS.copy()
            signal_params['max_holding_period'] = int(hl_train * 3)
            generator = SignalGenerator(**signal_params)
            raw_trades = generator.generate(zscore_window_oos)

            if not raw_trades:
                window_pairs_Summary.append(f"{asset_y}/{asset_x}(0)")
                continue

            df_trades_prices = engine.match_execution_prices(
                trades=raw_trades,
                df_open_prices=df_prices_all,
                pair=(asset_y, asset_x)
            )

            if df_trades_prices.empty:
                window_pairs_Summary.append(f"{asset_y}/{asset_x}(0)")
                continue

            df_res_pair = engine.calculate_trades_pnl(df_trades_prices, beta=beta_train)
            df_res_pair['pair_or_basket'] = f"{asset_y}/{asset_x}"

            # <--- ZMIANA: dodajemy wyniki pojedynczej pary do koszyka bieżącego okna
            window_trades_list.append(df_res_pair)
            window_pairs_Summary.append(f"{asset_y}/{asset_x}({len(df_res_pair)})")

        # <--- DODANO BLOK: podsumowanie całego okna po przeliczeniu wszystkich TOP K par
        pairs_str = ", ".join(window_pairs_Summary)
        if not window_trades_list:
            print(f"[Okno {idx:02d} | {te_start} -> {te_end}] Pary: {pairs_str:<38} | Brak transakcji")
            continue

        df_res_window = pd.concat(window_trades_list, ignore_index=True)
        all_trades_dfs.append(df_res_window)

        win_pnl = df_res_window['net_pnl'].sum()
        print(
            f"[Okno {idx:02d} | {te_start} -> {te_end}] Pary: {pairs_str:<38} "
            f"| Transakcji: {len(df_res_window):2d} | PnL: ${win_pnl:8,.2f}"
        )

    if not all_trades_dfs:
        print("Brak jakichkolwiek transakcji w całym okresie Walk-Forward dla Pairs Trading.")
        return None

    df_all_results = pd.concat(all_trades_dfs, ignore_index=True)
    metrics = engine.calculate_performance_metrics(df_all_results)

    print("\n--- CAŁKOWITE WYNIKI WALK-FORWARD OOS (PAIRS TRADING) ---")
    print(f"Liczba transakcji:   {metrics['total_trades']}")
    print(f"Zysk Netto (USD):    ${metrics['total_pnl']:,.2f}")
    print(f"Stopa zwrotu:        {metrics['total_return_pct']:.2f}%")
    print(f"Win Rate:            {metrics['win_rate_pct']:.1f}%")
    print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")

    return {
        "strategy_name": f"Walk-Forward Pairs Trading (TOP {top_k_pairs})",  # <--- ZMIANA
        "label": f"Portfel TOP {top_k_pairs} Par ({len(windows)} okien OOS)",  # <--- ZMIANA
        "zscore_oos": all_zscore_oos,
        "df_results": df_all_results,
        "metrics": metrics
    }

def run_basket_walk_forward(
    df_prices_all: pd.DataFrame,
    windows: List[Tuple[str, str, str, str]]
) -> Optional[Dict[str, Any]]:
    print("\n" + "="*65)
    print("🚀 WALK-FORWARD: JOHANSEN BASKET (N NÓG)")
    print("="*65)

    scanner = BasketScanner(basket_size=3, significance_level=0)
    z_calc = ZScoreCalculator(**config.ZSCORE_PARAMS)
    engine = BacktestEngine(**config.ENGINE_PARAMS)

    all_trades_dfs = []
    all_zscore_oos = []

    for idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows, 1):
        df_train = df_prices_all.loc[tr_start:tr_end].copy()
        df_test = df_prices_all.loc[te_start:te_end].copy()

        if df_test.empty:
            continue

        df_ranking = scanner.scan_sector(df_train)

        if df_ranking.empty:
            print(f"[Okno {idx:02d} | {te_start} -> {te_end}] Brak kointegrowanego koszyka w treningu.")
            for dt in df_test.index:
                all_zscore_oos.append((dt, 0.0))
            continue

        best_basket_row = df_ranking.iloc[0]
        basket_tickers = list(best_basket_row['basket'])
        weights = {k: float(np.real(v)) for k, v in best_basket_row['weights'].items()}

        log_train = np.log(df_train[basket_tickers])
        spread_train = sum(weights[t] * log_train[t] for t in basket_tickers).to_numpy()

        hl_res = CointegrationMath.calculate_half_life(spread_train)
        hl_train = float(hl_res.half_life) if (1.0 <= hl_res.half_life <= 60.0) else 20.0

        window_padding = int(np.clip(hl_train * 3.0, 15, 90))
        df_subset_upto_test = df_prices_all.loc[:te_end]
        df_prices_padded = df_subset_upto_test.iloc[-(len(df_test) + window_padding):]

        log_padded = np.log(df_prices_padded[basket_tickers])
        spread_test_series = sum(weights[t] * log_padded[t] for t in basket_tickers)
        df_spread_test = pd.DataFrame({'Date': spread_test_series.index, 'Spread': spread_test_series.values})

        zscore_data = z_calc.calculate(df_spread_test, pair_col='Spread', half_life=hl_train, date_col='Date')
        zscore_window_oos = [item for item in zscore_data if te_start <= str(item[0]) <= te_end]
        all_zscore_oos.extend(zscore_window_oos)

        signal_params = config.SIGNAL_PARAMS.copy()
        signal_params['max_holding_period'] = int(hl_train * 3)
        generator = SignalGenerator(**signal_params)
        raw_trades = generator.generate(zscore_window_oos)

        basket_str = "-".join(basket_tickers)
        if not raw_trades:
            print(f"[Okno {idx:02d} | {te_start} -> {te_end}] Koszyk: {basket_str:<14} (HL={hl_train:4.1f}d) | Brak transakcji")
            continue

        df_trades_prices = engine.match_basket_execution_prices(
            trades=raw_trades,
            df_open_prices=df_prices_all,
            tickers=basket_tickers
        )

        if df_trades_prices.empty:
            continue

        df_res_window = engine.calculate_basket_trades_pnl(df_trades_prices, weights=weights)
        df_res_window['pair_or_basket'] = basket_str
        all_trades_dfs.append(df_res_window)

        win_pnl = df_res_window['net_pnl'].sum()
        print(
            f"[Okno {idx:02d} | {te_start} -> {te_end}] Koszyk: {basket_str:<14} "
            f"(HL={hl_train:4.1f}d) | Transakcji: {len(df_res_window):2d} | PnL: ${win_pnl:8,.2f}"
        )

    if not all_trades_dfs:
        print("Brak jakichkolwiek transakcji w całym okresie Walk-Forward dla Johansen Basket.")
        return None

    df_all_results = pd.concat(all_trades_dfs, ignore_index=True)
    metrics = engine.calculate_performance_metrics(df_all_results)

    print("\n--- CAŁKOWITE WYNIKI WALK-FORWARD OOS (JOHANSEN BASKET) ---")
    print(f"Liczba transakcji:   {metrics['total_trades']}")
    print(f"Zysk Netto (USD):    ${metrics['total_pnl']:,.2f}")
    print(f"Stopa zwrotu:        {metrics['total_return_pct']:.2f}%")
    print(f"Win Rate:            {metrics['win_rate_pct']:.1f}%")
    print(f"Profit Factor:       {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")

    return {
        "strategy_name": "Walk-Forward Johansen Basket",
        "label": f"Dynamiczne Koszyki 3-el. ({ len(windows) } okien OOS)",
        "zscore_oos": all_zscore_oos,
        "df_results": df_all_results,
        "metrics": metrics
    }


def main():
    # ==========================================
    # KONFIGURACJA WALK-FORWARD I TRYBU
    # ==========================================
    EXECUTION_MODE = 'both'   # 'pairs', 'basket' lub 'both'
    TRAIN_MONTHS = 12         # Długość kroczącego okna treningowego (np. 12 lub 6)
    TEST_MONTHS = 3           # Długość okna handlowego OOS przed re-estymacją (np. 3 lub 1)
    TOP_K_PAIRS = 3           # <--- DODANO: Maksymalna liczba par handlowanych równolegle w oknie

    oos_start = "2022-01-01"
    oos_end = "2023-12-31"

    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']

    # Wyznaczenie harmonogramu okien Walk-Forward
    windows = generate_walk_forward_windows(oos_start, oos_end, TRAIN_MONTHS, TEST_MONTHS)
    first_train_start = windows[0][0]

    print(f"=== SYSTEM STAT-ARB WALK-FORWARD (Tryb: {EXECUTION_MODE.upper()}) ===")
    print(f"Konfiguracja okien: {TRAIN_MONTHS} msc treningu -> {TEST_MONTHS} msc handlu OOS (Liczba okien: {len(windows)})")
    print(f"Pobieranie danych od {first_train_start} do {oos_end}...")

    loader = DataLoader(**config.DATALOADER_PARAMS)
    df_prices_all = loader.get_prices(tickers, start_date=first_train_start, end_date=oos_end)

    df_caps = None
    if EXECUTION_MODE in ['pairs', 'both']:
        df_caps = loader.get_market_caps(tickers)

    pairs_output = None
    basket_output = None

    if EXECUTION_MODE in ['pairs', 'both']:
        # <--- ZMIANA: przekazujemy parametr top_k_pairs=TOP_K_PAIRS
        pairs_output = run_pairs_walk_forward(df_prices_all, df_caps, windows, top_k_pairs=TOP_K_PAIRS)
        
    if EXECUTION_MODE in ['basket', 'both']:
        basket_output = run_basket_walk_forward(df_prices_all, windows)

    # ==========================================
    # WIZUALIZACJA WYNIKÓW (TEARSHEETS)
    # ==========================================
    initial_cap = config.ENGINE_PARAMS.get("initial_capital", 100_000.0)
    plotter = StrategyPlotter(initial_capital=initial_cap)

    entry_th = config.SIGNAL_PARAMS.get("entry_threshold", 1.5)
    exit_th = config.SIGNAL_PARAMS.get("exit_threshold", 0.5)
    sl_th = config.SIGNAL_PARAMS.get("SL_threshold", 3.5)

    if pairs_output:
        plotter.plot_strategy_tearsheet(
            strategy_name=pairs_output["strategy_name"],
            subtitle=f"{pairs_output['label']} | {TRAIN_MONTHS}m/{TEST_MONTHS}m",
            zscore_data_oos=pairs_output["zscore_oos"],
            df_results=pairs_output["df_results"],
            metrics=pairs_output["metrics"],
            entry_threshold=entry_th,
            exit_threshold=exit_th,
            sl_threshold=sl_th
        )

    if basket_output:
        plotter.plot_strategy_tearsheet(
            strategy_name=basket_output["strategy_name"],
            subtitle=f"{basket_output['label']} | {TRAIN_MONTHS}m/{TEST_MONTHS}m",
            zscore_data_oos=basket_output["zscore_oos"],
            df_results=basket_output["df_results"],
            metrics=basket_output["metrics"],
            entry_threshold=entry_th,
            exit_threshold=exit_th,
            sl_threshold=sl_th
        )

    if pairs_output and basket_output:
        plotter.plot_strategies_comparison(pairs_output, basket_output)

    plt.show()


if __name__ == "__main__":
    main()