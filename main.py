import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional

# Wyciszenie ostrzeżeń systemowych
warnings.filterwarnings("ignore")

import config

from data.data_loader import DataLoader
from research.market_scanner import MarketScanner, BasketScanner
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


# <--- DODANO FUNKCJĘ POMOCNICZĄ: Automatyczne i bezbłędne wycinanie danych z buforem (paddingiem)
# Działa identycznie dla '1d', '1h' oraz '15m' bez ręcznego ustawiania zakresów!
def extract_padded_slice(
    df_all: pd.DataFrame,
    te_start_dt: pd.Timestamp,
    te_end_dt: pd.Timestamp,
    padding_bars: int
) -> pd.DataFrame:
    """
    Zwraca wycinek danych z df_all obejmujący okres testowy [te_start_dt, te_end_dt]
    poprzedzony dokładnie `padding_bars` świecami z końca okresu treningowego.
    """
    idx = df_all.index
    start_pos = idx.searchsorted(te_start_dt, side='left')
    end_pos = idx.searchsorted(te_end_dt, side='right')
    pad_start_pos = max(0, start_pos - padding_bars)
    return df_all.iloc[pad_start_pos:end_pos].copy()


def select_diversified_pairs(df_ranking: pd.DataFrame, top_k: int) -> pd.DataFrame:
    chosen_indices = []
    used_tickers = set()

    for idx, row in df_ranking.iterrows():
        y, x = row['asset_y'], row['asset_x']
        if y not in used_tickers and x not in used_tickers:
            chosen_indices.append(idx)
            used_tickers.update([y, x])
            if len(chosen_indices) == top_k:
                break

    if len(chosen_indices) < top_k:
        for idx in df_ranking.index:
            if idx not in chosen_indices:
                chosen_indices.append(idx)
                if len(chosen_indices) == top_k:
                    break

    return df_ranking.loc[chosen_indices]


def select_diversified_baskets(df_ranking: pd.DataFrame, top_k: int = 2) -> pd.DataFrame:
    chosen_indices = []
    used_tickers = set()

    for idx, row in df_ranking.iterrows():
        basket_tickers = set(row['basket'])
        if not basket_tickers.intersection(used_tickers):
            chosen_indices.append(idx)
            used_tickers.update(basket_tickers)
            if len(chosen_indices) == top_k:
                break

    if len(chosen_indices) < top_k:
        for idx in df_ranking.index:
            if idx not in chosen_indices:
                chosen_indices.append(idx)
                if len(chosen_indices) == top_k:
                    break

    return df_ranking.loc[chosen_indices]


def run_pairs_walk_forward(
    df_prices_all: pd.DataFrame,
    df_open_all: pd.DataFrame,
    df_caps: pd.DataFrame,
    windows: List[Tuple[str, str, str, str]],
    sectors_dict: Dict[str, List[str]],
    top_k_pairs: int = 3
) -> Optional[Dict[str, Any]]:
    print("\n" + "="*75)
    print(f"🚀 WALK-FORWARD: PAIRS TRADING (PORTFEL TOP {top_k_pairs} PAR | INTERWAŁ: {config.ACTIVE_INTERVAL})")
    print("="*75)

    scanner = MarketScanner(**config.SCANNER_PARAMS)
    z_calc = ZScoreCalculator(**config.ZSCORE_PARAMS)
    engine = BacktestEngine(**config.ENGINE_PARAMS)

    base_allocation = config.ENGINE_PARAMS.get("allocation_per_trade", 0.10)

    all_trades_dfs = []
    all_zscore_oos = []

    for idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows, 1):
        # <--- ZMIANA: Pełne znaczniki doby (23:59:59), aby na 1h nie ucinać ostatniego dnia okna!
        tr_start_dt = pd.to_datetime(tr_start)
        tr_end_dt = pd.to_datetime(tr_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        te_start_dt = pd.to_datetime(te_start)
        te_end_dt = pd.to_datetime(te_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)

        df_train = df_prices_all.loc[(df_prices_all.index >= tr_start_dt) & (df_prices_all.index <= tr_end_dt)].copy()
        df_test = df_prices_all.loc[(df_prices_all.index >= te_start_dt) & (df_prices_all.index <= te_end_dt)].copy()

        if df_test.empty or df_train.empty:
            continue

        sector_rankings = []
        for sector_name, sector_tickers in sectors_dict.items():
            available_tickers = [t for t in sector_tickers if t in df_train.columns]
            if len(available_tickers) < 2:
                continue
            df_sec_rank = scanner.scan_sector(df_train[available_tickers], df_caps)
            if not df_sec_rank.empty:
                df_sec_rank['sector'] = sector_name
                sector_rankings.append(df_sec_rank)

        if not sector_rankings:
            print(f"[Okno {idx:02d} | {te_start} -> {te_end}] Brak stacjonarnych par w żadnym sektorze.")
            for dt in df_test.index:
                all_zscore_oos.append((dt, 0.0))
            continue

        df_ranking = pd.concat(sector_rankings, ignore_index=True)
        df_ranking = df_ranking.sort_values(by="half_life", ascending=True).reset_index(drop=True)

        selected_pairs = select_diversified_pairs(df_ranking, top_k_pairs)
        engine.allocation_per_trade = base_allocation / len(selected_pairs)

        window_trades_list = []
        window_pairs_summary = []

        for pair_idx, (_, best_pair) in enumerate(selected_pairs.iterrows()):
            asset_y = best_pair['asset_y']
            asset_x = best_pair['asset_x']
            hl_train = float(best_pair['half_life'])
            beta_train = float(best_pair['beta'])

            # <--- ZMIANA: Automatyczny padding wyliczany bezpośrednio z ZScoreCalculator!
            window_padding = z_calc.get_window_length(hl_train) + 5
            df_prices_padded = extract_padded_slice(df_prices_all, te_start_dt, te_end_dt, window_padding)

            spread_test = np.log(df_prices_padded[asset_y]) - (beta_train * np.log(df_prices_padded[asset_x]))
            df_spread_test = pd.DataFrame({'Date': spread_test.index, 'Spread': spread_test.values})

            zscore_data = z_calc.calculate(df_spread_test, pair_col='Spread', half_life=hl_train, date_col='Date')
            zscore_window_oos = [item for item in zscore_data if te_start_dt <= pd.to_datetime(item[0]) <= te_end_dt]

            if pair_idx == 0:
                all_zscore_oos.extend(zscore_window_oos)

            signal_params = config.PAIRS_SIGNAL_PARAMS.copy()
            signal_params['max_holding_period'] = int(hl_train * 3)
            generator = SignalGenerator(**signal_params)
            raw_trades = generator.generate(zscore_window_oos)

            if not raw_trades:
                window_pairs_summary.append(f"{asset_y}/{asset_x}(0)")
                continue

            df_trades_prices = engine.match_execution_prices(
                trades=raw_trades,
                df_open_prices=df_open_all,
                pair=(asset_y, asset_x)
            )

            if df_trades_prices.empty:
                window_pairs_summary.append(f"{asset_y}/{asset_x}(0)")
                continue

            df_res_pair = engine.calculate_trades_pnl(df_trades_prices, beta=beta_train)
            df_res_pair['pair_or_basket'] = f"{asset_y}/{asset_x}"

            window_trades_list.append(df_res_pair)
            window_pairs_summary.append(f"{asset_y}/{asset_x}({len(df_res_pair)})")

        pairs_str = ", ".join(window_pairs_summary)
        if not window_trades_list:
            print(f"[Okno {idx:02d} | {te_start} -> {te_end}] Pary: {pairs_str:<42} | Brak transakcji")
            continue

        df_res_window = pd.concat(window_trades_list, ignore_index=True)
        all_trades_dfs.append(df_res_window)

        win_pnl = df_res_window['net_pnl'].sum()
        print(
            f"[Okno {idx:02d} | {te_start} -> {te_end}] Pary: {pairs_str:<42} "
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
        "strategy_name": f"Walk-Forward Pairs Trading (TOP {top_k_pairs})",
        "label": f"Portfel TOP {top_k_pairs} Par ({len(windows)} okien OOS)",
        "zscore_oos": all_zscore_oos,
        "df_results": df_all_results,
        "metrics": metrics
    }


def run_basket_walk_forward(
    df_prices_all: pd.DataFrame,
    df_open_all: pd.DataFrame,
    windows: List[Tuple[str, str, str, str]],
    sectors_dict: Dict[str, List[str]],
    top_k_baskets: int = 2
) -> Optional[Dict[str, Any]]:
    print("\n" + "="*75)
    print(f"🚀 WALK-FORWARD: JOHANSEN BASKET (PORTFEL TOP {top_k_baskets} KOSZYKÓW | INTERWAŁ: {config.ACTIVE_INTERVAL})")
    print("="*75)

    # <--- ZMIANA: Pobieramy parametry skanera koszyków (w tym max_half_life dopasowany do 1d/1h) z config.py!
    scanner = BasketScanner(**config.BASKET_SCANNER_PARAMS)
    z_calc = ZScoreCalculator(**config.ZSCORE_PARAMS)
    engine = BacktestEngine(**config.ENGINE_PARAMS)

    base_allocation = config.ENGINE_PARAMS.get("allocation_per_trade", 0.10)

    all_trades_dfs = []
    all_zscore_oos = []

    for idx, (tr_start, tr_end, te_start, te_end) in enumerate(windows, 1):
        # <--- ZMIANA: Pełne znaczniki doby (23:59:59) dla poprawnej obsługi świec 1h
        tr_start_dt = pd.to_datetime(tr_start)
        tr_end_dt = pd.to_datetime(tr_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        te_start_dt = pd.to_datetime(te_start)
        te_end_dt = pd.to_datetime(te_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)

        df_train = df_prices_all.loc[(df_prices_all.index >= tr_start_dt) & (df_prices_all.index <= tr_end_dt)].copy()
        df_test = df_prices_all.loc[(df_prices_all.index >= te_start_dt) & (df_prices_all.index <= te_end_dt)].copy()

        if df_test.empty or df_train.empty:
            continue

        sector_baskets = []
        for sector_name, sector_tickers in sectors_dict.items():
            available_tickers = [t for t in sector_tickers if t in df_train.columns]
            if len(available_tickers) < 3:
                continue
            df_sec_baskets = scanner.scan_sector(df_train[available_tickers])
            if not df_sec_baskets.empty:
                df_sec_baskets['sector'] = sector_name
                sector_baskets.append(df_sec_baskets)

        if not sector_baskets:
            print(f"[Okno {idx:02d} | {te_start} -> {te_end}] Brak kointegrowanego koszyka w żadnym sektorze.")
            for dt in df_test.index:
                all_zscore_oos.append((dt, 0.0))
            continue

        df_ranking = pd.concat(sector_baskets, ignore_index=True)
        df_ranking = df_ranking.sort_values(by="max_eigenvalue", ascending=False).reset_index(drop=True)

        selected_baskets = select_diversified_baskets(df_ranking, top_k_baskets)
        engine.allocation_per_trade = base_allocation / len(selected_baskets)

        window_trades_list = []
        window_baskets_summary = []

        max_hl_allowed = config.BASKET_SCANNER_PARAMS.get('max_half_life', 84.0)

        for basket_idx, (_, best_basket_row) in enumerate(selected_baskets.iterrows()):
            basket_tickers = list(best_basket_row['basket'])
            weights = {k: float(np.real(v)) for k, v in best_basket_row['weights'].items()}
            sector = best_basket_row['sector']

            log_train = np.log(df_train[basket_tickers])
            spread_train = sum(weights[t] * log_train[t] for t in basket_tickers).to_numpy()

            hl_res = CointegrationMath.calculate_half_life(spread_train)
            hl_train = float(hl_res.half_life) if (1.0 <= hl_res.half_life <= max_hl_allowed) else 20.0

            # <--- ZMIANA: Automatyczny padding wyliczany bezpośrednio z ZScoreCalculator!
            window_padding = z_calc.get_window_length(hl_train) + 5
            df_prices_padded = extract_padded_slice(df_prices_all, te_start_dt, te_end_dt, window_padding)

            log_padded = np.log(df_prices_padded[basket_tickers])
            spread_test_series = sum(weights[t] * log_padded[t] for t in basket_tickers)
            df_spread_test = pd.DataFrame({'Date': spread_test_series.index, 'Spread': spread_test_series.values})

            zscore_data = z_calc.calculate(df_spread_test, pair_col='Spread', half_life=hl_train, date_col='Date')
            zscore_window_oos = [item for item in zscore_data if te_start_dt <= pd.to_datetime(item[0]) <= te_end_dt]

            if basket_idx == 0:
                all_zscore_oos.extend(zscore_window_oos)

            signal_params = config.BASKET_SIGNAL_PARAMS.copy()
            signal_params['max_holding_period'] = int(hl_train * 3)
            generator = SignalGenerator(**signal_params)
            raw_trades = generator.generate(zscore_window_oos)

            basket_str = "-".join(basket_tickers)
            if not raw_trades:
                window_baskets_summary.append(f"{basket_str}(0)")
                continue

            df_trades_prices = engine.match_basket_execution_prices(
                trades=raw_trades,
                df_open_prices=df_open_all,
                tickers=basket_tickers
            )

            if df_trades_prices.empty:
                window_baskets_summary.append(f"{basket_str}(0)")
                continue

            df_res_basket = engine.calculate_basket_trades_pnl(df_trades_prices, weights=weights)
            df_res_basket['pair_or_basket'] = f"{basket_str} [{sector}]"

            window_trades_list.append(df_res_basket)
            window_baskets_summary.append(f"{basket_str}({len(df_res_basket)})")

        baskets_summary_str = ", ".join(window_baskets_summary)
        if not window_trades_list:
            print(f"[Okno {idx:02d} | {te_start} -> {te_end}] Koszyki: {baskets_summary_str:<45} | Brak transakcji")
            continue

        df_res_window = pd.concat(window_trades_list, ignore_index=True)
        all_trades_dfs.append(df_res_window)

        win_pnl = df_res_window['net_pnl'].sum()
        print(
            f"[Okno {idx:02d} | {te_start} -> {te_end}] Koszyki: {baskets_summary_str:<45} "
            f"| Transakcji: {len(df_res_window):2d} | PnL: ${win_pnl:8,.2f}"
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
        "strategy_name": f"Walk-Forward Johansen Basket (TOP {top_k_baskets})",
        "label": f"Portfel TOP {top_k_baskets} Koszyków ({len(windows)} okien OOS)",
        "zscore_oos": all_zscore_oos,
        "df_results": df_all_results,
        "metrics": metrics
    }


def main():
    # <--- ZMIANA: Wszystkie flagi pobieramy automatycznie z config.py!
    EXECUTION_MODE = config.EXECUTION_MODE
    INTERVAL = config.ACTIVE_INTERVAL
    TRAIN_MONTHS = config.TRAIN_MONTHS
    TEST_MONTHS = config.TEST_MONTHS
    TOP_K_PAIRS = config.TOP_K_PAIRS
    TOP_K_BASKETS = config.TOP_K_BASKETS
    oos_start = config.OOS_START
    oos_end = config.OOS_END

    active_sectors_dict = {sec: config.SECTORS[sec] for sec in config.ACTIVE_SECTORS if sec in config.SECTORS}
    all_tickers = sorted(list({t for sec_list in active_sectors_dict.values() for t in sec_list}))

    windows = generate_walk_forward_windows(oos_start, oos_end, TRAIN_MONTHS, TEST_MONTHS)
    first_train_start = windows[0][0]

    print(f"=== SYSTEM STAT-ARB WALK-FORWARD (Tryb: {EXECUTION_MODE.upper()} | Interwał: {INTERVAL}) ===")
    print(f"Aktywne sektory ({len(active_sectors_dict)}): {', '.join(active_sectors_dict.keys())} (Łącznie {len(all_tickers)} spółek)")
    print(f"Konfiguracja okien: {TRAIN_MONTHS} msc treningu -> {TEST_MONTHS} msc handlu OOS (Liczba okien: {len(windows)})")
    print(f"Pobieranie danych ({INTERVAL}) od {first_train_start} do {oos_end}...")

    loader = DataLoader(default_interval=INTERVAL)
    df_prices_all, df_open_all = loader.get_prices(
        all_tickers, 
        start_date=first_train_start, 
        end_date=oos_end, 
        interval=INTERVAL,
        return_open=True
    )

    df_caps = None
    if EXECUTION_MODE in ['pairs', 'both']:
        df_caps = loader.get_market_caps(all_tickers)

    pairs_output = None
    basket_output = None

    if EXECUTION_MODE in ['pairs', 'both']:
        pairs_output = run_pairs_walk_forward(
            df_prices_all=df_prices_all,
            df_open_all=df_open_all,
            df_caps=df_caps,
            windows=windows,
            sectors_dict=active_sectors_dict,
            top_k_pairs=TOP_K_PAIRS
        )

    if EXECUTION_MODE in ['basket', 'both']:
        basket_output = run_basket_walk_forward(
            df_prices_all=df_prices_all,
            df_open_all=df_open_all,
            windows=windows,
            sectors_dict=active_sectors_dict,
            top_k_baskets=TOP_K_BASKETS
        )

    # ==========================================
    # WIZUALIZACJA WYNIKÓW (TEARSHEETS)
    # ==========================================
    initial_cap = config.ENGINE_PARAMS.get("initial_capital", 100_000.0)
    plotter = StrategyPlotter(initial_capital=initial_cap)

    entry_th_pairs = config.PAIRS_SIGNAL_PARAMS.get("entry_threshold", 2.0)
    exit_th_pairs = config.PAIRS_SIGNAL_PARAMS.get("exit_threshold", 0.5)
    sl_th_pairs = config.PAIRS_SIGNAL_PARAMS.get("SL_threshold", 3.5)

    entry_th_basket = config.BASKET_SIGNAL_PARAMS.get("entry_threshold", 1.5)
    exit_th_basket = config.BASKET_SIGNAL_PARAMS.get("exit_threshold", 0.5)
    sl_th_basket = config.BASKET_SIGNAL_PARAMS.get("SL_threshold", 3.5)

    if pairs_output:
        plotter.plot_strategy_tearsheet(
            strategy_name=pairs_output["strategy_name"],
            subtitle=f"{pairs_output['label']} | Okno: {TRAIN_MONTHS}m/{TEST_MONTHS}m | Interwał: {INTERVAL}",
            zscore_data_oos=pairs_output["zscore_oos"],
            df_results=pairs_output["df_results"],
            metrics=pairs_output["metrics"],
            entry_threshold=entry_th_pairs,
            exit_threshold=exit_th_pairs,
            sl_threshold=sl_th_pairs
        )

    if basket_output:
        plotter.plot_strategy_tearsheet(
            strategy_name=basket_output["strategy_name"],
            subtitle=f"{basket_output['label']} | Okno: {TRAIN_MONTHS}m/{TEST_MONTHS}m | Interwał: {INTERVAL}",
            zscore_data_oos=basket_output["zscore_oos"],
            df_results=basket_output["df_results"],
            metrics=basket_output["metrics"],
            entry_threshold=entry_th_basket,
            exit_threshold=exit_th_basket,
            sl_threshold=sl_th_basket
        )

    if pairs_output and basket_output:
        plotter.plot_strategies_comparison(pairs_output, basket_output, interval_label=INTERVAL)
    
    plt.show()


if __name__ == "__main__":
    main()
