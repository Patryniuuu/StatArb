import os
import copy
import pandas as pd
import numpy as np
import matplotlib
# Używamy backendu bezokienkowego 'Agg', aby wykresy zapisywały się na dysk błyskawicznie w tle
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from data.data_loader import DataLoader
from visualization.plotter import StrategyPlotter
from main import (
    generate_walk_forward_windows,
    run_pairs_walk_forward,
    run_basket_walk_forward
)


# =====================================================================
# DEFINICJA 8 SCENARIUSZY BADAWCZYCH DO RAPORTU LATEX
# =====================================================================
SCENARIOS = [
    {
        "id": "S1_Baseline_1d",
        "short_name": "S1: Baseline (1d, No Tech)",
        "interval": "1d",
        "mode": "both",
        "top_k_pairs": 1,
        "top_k_baskets": 1,
        "sectors": ["BANKS_FINANCE", "ENERGY_OIL", "CONSUMER_STAPLES"],
        "oos_start": "2020-01-01",
        "oos_end": "2024-12-31",
        "train_months": 12,
        "test_months": 3,
        "overrides": {}
    },
    {
        "id": "S2_Diversified_1d",
        "short_name": "S2: Diversified (3P / 2B)",
        "interval": "1d",
        "mode": "both",
        "top_k_pairs": 3,
        "top_k_baskets": 2,
        "sectors": ["TECH_SEMI", "BANKS_FINANCE", "ENERGY_OIL", "CONSUMER_STAPLES"],
        "oos_start": "2020-01-01",
        "oos_end": "2024-12-31",
        "train_months": 12,
        "test_months": 3,
        "overrides": {}
    },
    {
        "id": "S3_BearMarket2022_1d",
        "short_name": "S3: Bear Market (2022)",
        "interval": "1d",
        "mode": "both",
        "top_k_pairs": 1,
        "top_k_baskets": 1,
        "sectors": ["TECH_SEMI", "BANKS_FINANCE", "ENERGY_OIL", "CONSUMER_STAPLES"],
        "oos_start": "2022-01-01",
        "oos_end": "2022-12-31",
        "train_months": 12,
        "test_months": 3,
        "overrides": {}
    },
    {
        "id": "S4_TechOnly_1d",
        "short_name": "S4: Tech Sector Only (1d)",
        "interval": "1d",
        "mode": "both",
        "top_k_pairs": 1,
        "top_k_baskets": 1,
        "sectors": ["TECH_SEMI"],
        "oos_start": "2020-01-01",
        "oos_end": "2024-12-31",
        "train_months": 12,
        "test_months": 3,
        "overrides": {}
    },
    {
        "id": "S5_HighThreshold_1d",
        "short_name": "S5: High Entry (2.5σ Pairs)",
        "interval": "1d",
        "mode": "pairs",
        "top_k_pairs": 1,
        "top_k_baskets": 1,
        "sectors": ["TECH_SEMI", "BANKS_FINANCE", "ENERGY_OIL", "CONSUMER_STAPLES"],
        "oos_start": "2020-01-01",
        "oos_end": "2024-12-31",
        "train_months": 12,
        "test_months": 3,
        "overrides": {
            "pairs_signal_params": {"entry_threshold": 2.5}
        }
    },
    {
        "id": "S6_Intraday_1h",
        "short_name": "S6: Intraday Baseline (1h)",
        "interval": "1h",
        "mode": "both",
        "top_k_pairs": 1,
        "top_k_baskets": 1,
        "sectors": ["TECH_SEMI", "BANKS_FINANCE", "ENERGY_OIL", "CONSUMER_STAPLES"],
        "oos_start": "2025-05-01",
        "oos_end": "2026-09-30",
        "train_months": 3,
        "test_months": 1,
        "overrides": {}
    },
    {
        "id": "S7_FastHalfLife_1h",
        "short_name": "S7: Fast Half-Life (1h, HL<=20)",
        "interval": "1h",
        "mode": "both",
        "top_k_pairs": 1,
        "top_k_baskets": 1,
        "sectors": ["TECH_SEMI", "BANKS_FINANCE", "ENERGY_OIL", "CONSUMER_STAPLES"],
        "oos_start": "2025-05-01",
        "oos_end": "2026-09-30",
        "train_months": 3,
        "test_months": 1,
        "overrides": {
            "scanner_params": {"max_half_life": 20.0},
            "basket_scanner_params": {"max_half_life": 20.0}
        }
    },
    {
        "id": "S8_JohansenSolo_1d",
        "short_name": "S8: Johansen Basket Solo (1d)",
        "interval": "1d",
        "mode": "basket",
        "top_k_pairs": 1,
        "top_k_baskets": 1,
        "sectors": ["TECH_SEMI", "BANKS_FINANCE", "ENERGY_OIL", "CONSUMER_STAPLES"],
        "oos_start": "2020-01-01",
        "oos_end": "2024-12-31",
        "train_months": 12,
        "test_months": 3,
        "overrides": {}
    }
]


def apply_scenario_config(scen: dict) -> None:
    """Dynamicznie ustawia parametry w module config pod dany scenariusz."""
    interval = scen["interval"]
    base_profile = copy.deepcopy(config.INTERVAL_PROFILES[interval])

    # Aplikujemy ewentualne nadpisania (overrides)
    for dict_key, updates in scen.get("overrides", {}).items():
        if dict_key in base_profile:
            base_profile[dict_key].update(updates)

    config.ACTIVE_INTERVAL = interval
    config.EXECUTION_MODE = scen["mode"]
    config.TOP_K_PAIRS = scen["top_k_pairs"]
    config.TOP_K_BASKETS = scen["top_k_baskets"]
    config.ACTIVE_SECTORS = scen["sectors"]
    config.TRAIN_MONTHS = scen["train_months"]
    config.TEST_MONTHS = scen["test_months"]
    config.OOS_START = scen["oos_start"]
    config.OOS_END = scen["oos_end"]

    config.ZSCORE_PARAMS = base_profile["zscore_params"]
    config.SCANNER_PARAMS = base_profile["scanner_params"]
    config.BASKET_SCANNER_PARAMS = base_profile["basket_scanner_params"]
    config.PAIRS_SIGNAL_PARAMS = base_profile["pairs_signal_params"]
    config.BASKET_SIGNAL_PARAMS = base_profile["basket_signal_params"]


def extract_cost_breakdown(df_results: pd.DataFrame) -> dict:
    """Wylicza sumę Gross PnL, łącznych kosztów (Execution + Borrow) oraz Net PnL."""
    if df_results is None or df_results.empty:
        return {"gross_pnl": 0.0, "exec_cost": 0.0, "borrow_cost": 0.0, "total_costs": 0.0, "net_pnl": 0.0}
    gross = float(df_results["gross_pnl"].sum())
    exec_c = float(df_results["execution_cost"].sum())
    borrow_c = float(df_results["borrow_cost"].sum())
    net = float(df_results["net_pnl"].sum())
    return {
        "gross_pnl": round(gross, 2),
        "exec_cost": round(exec_c, 2),
        "borrow_cost": round(borrow_c, 2),
        "total_costs": round(exec_c + borrow_c, 2),
        "net_pnl": round(net, 2)
    }


def generate_latex_master_table(summary_rows: list, output_path: str) -> None:
    """
    Generuje kompletną, akademicką tabelę LaTeX (booktabs) gotową do wklejenia w Overleafie.
    """
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Summary of Out-of-Sample Performance Metrics Across All Experimental Scenarios}",
        r"\label{tab:master_results}",
        r"\small",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Strategy} & \textbf{Trades} & \textbf{Gross PnL} & \textbf{Costs} & \textbf{Net PnL} & \textbf{Win Rate} & \textbf{Profit Factor} & \textbf{Sharpe} \\",
        r"\midrule"
    ]

    for row in summary_rows:
        scen_label = row["Scenario"].replace("_", r"\_")
        strat = row["Strategy"]
        trades = row["Trades"]
        gross = f"\\${row['Gross_PnL']:,.2f}"
        costs = f"\\${row['Total_Costs']:,.2f}"
        net = f"\\${row['Net_PnL']:,.2f}"
        wr = f"{row['Win_Rate']:.1f}\\%"
        pf = f"{row['Profit_Factor']:.2f}"
        sharpe = f"{row['Sharpe']:.2f}"

        lines.append(
            f"{scen_label} & {strat} & {trades} & {gross} & {costs} & \\textbf{{{net}}} & {wr} & {pf} & \\textbf{{{sharpe}}} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    all_tickers = sorted(list({t for sec_list in config.SECTORS.values() for t in sec_list}))

    print("="*80)
    print("📦 KROK 1/3: POBIERANIE DANYCH RYNKOWYCH DO PAMIĘCI PODRĘCZNEJ (CACHE)")
    print("="*80)

    loader_1d = DataLoader(default_interval="1d")
    print("-> Pobieranie kapitalizacji rynkowych (Market Caps) dla 36 spółek...")
    df_caps = loader_1d.get_market_caps(all_tickers)

    print("-> Pobieranie danych dziennych (1d: 2019-01-01 do 2024-12-31)...")
    df_close_1d, df_open_1d = loader_1d.get_prices(
        all_tickers, start_date="2019-01-01", end_date="2024-12-31", interval="1d", return_open=True
    )

    loader_1h = DataLoader(default_interval="1h")
    print("-> Pobieranie danych godzinowych (1h: 2025-02-01 do 2026-09-30)...")
    df_close_1h, df_open_1h = loader_1h.get_prices(
        all_tickers, start_date="2025-02-01", end_date="2026-09-30", interval="1h", return_open=True
    )

    plotter = StrategyPlotter(initial_capital=config.ENGINE_PARAMS["initial_capital"])

    summary_rows = []
    cost_drag_records = []
    daily_basket_curves = {}
    daily_pairs_curves = {}

    print("\n" + "="*80)
    print("🚀 KROK 2/3: URUCHAMIANIE 8 SCENARIUSZY BADAWCZYCH")
    print("="*80)

    for scen in SCENARIOS:
        scen_id = scen["id"]
        print(f"\n>>> Wykonywanie: {scen['short_name']} [{scen_id}] <<<")
        apply_scenario_config(scen)

        windows = generate_walk_forward_windows(
            config.OOS_START, config.OOS_END, config.TRAIN_MONTHS, config.TEST_MONTHS
        )
        active_sectors_dict = {sec: config.SECTORS[sec] for sec in config.ACTIVE_SECTORS}

        if config.ACTIVE_INTERVAL == "1d":
            df_c, df_o = df_close_1d, df_open_1d
        else:
            df_c, df_o = df_close_1h, df_open_1h

        pairs_out = None
        basket_out = None

        if config.EXECUTION_MODE in ["pairs", "both"]:
            pairs_out = run_pairs_walk_forward(
                df_prices_all=df_c,
                df_open_all=df_o,
                df_caps=df_caps,
                windows=windows,
                sectors_dict=active_sectors_dict,
                top_k_pairs=config.TOP_K_PAIRS
            )

        if config.EXECUTION_MODE in ["basket", "both"]:
            basket_out = run_basket_walk_forward(
                df_prices_all=df_c,
                df_open_all=df_o,
                windows=windows,
                sectors_dict=active_sectors_dict,
                top_k_baskets=config.TOP_K_BASKETS
            )

        # Zapis wykresów i metryk dla Pairs Trading
        if pairs_out:
            m_p = pairs_out["metrics"]
            c_p = extract_cost_breakdown(pairs_out["df_results"])
            summary_rows.append({
                "Scenario": scen["short_name"],
                "Strategy": f"Pairs (Top {config.TOP_K_PAIRS})",
                "Interval": config.ACTIVE_INTERVAL,
                "Trades": m_p["total_trades"],
                "Gross_PnL": c_p["gross_pnl"],
                "Total_Costs": c_p["total_costs"],
                "Net_PnL": m_p["total_pnl"],
                "Return_Pct": m_p["total_return_pct"],
                "Win_Rate": m_p["win_rate_pct"],
                "Profit_Factor": m_p["profit_factor"],
                "Max_DD_Pct": m_p["max_drawdown_pct"],
                "Sharpe": m_p["sharpe_ratio"]
            })
            cost_drag_records.append({
                "label": f"{scen_id}\n(Pairs)",
                "gross_pnl": c_p["gross_pnl"],
                "total_costs": c_p["total_costs"],
                "net_pnl": c_p["net_pnl"]
            })
            plotter.plot_strategy_tearsheet(
                strategy_name=f"{scen['short_name']} — Pairs Trading",
                subtitle=f"Window: {config.TRAIN_MONTHS}m/{config.TEST_MONTHS}m | Interval: {config.ACTIVE_INTERVAL}",
                zscore_data_oos=pairs_out["zscore_oos"],
                df_results=pairs_out["df_results"],
                metrics=m_p,
                entry_threshold=config.PAIRS_SIGNAL_PARAMS["entry_threshold"],
                exit_threshold=config.PAIRS_SIGNAL_PARAMS["exit_threshold"],
                sl_threshold=config.PAIRS_SIGNAL_PARAMS["SL_threshold"],
                save_path=f"results/plots/{scen_id}_pairs_tearsheet.png"
            )
            if config.ACTIVE_INTERVAL == "1d" and scen_id in ["S1_Baseline_1d", "S2_Diversified_1d", "S4_TechOnly_1d", "S5_HighThreshold_1d"]:
                daily_pairs_curves[scen["short_name"]] = (pairs_out["df_results"], pairs_out["zscore_oos"])

        # Zapis wykresów i metryk dla Johansen Basket
        if basket_out:
            m_b = basket_out["metrics"]
            c_b = extract_cost_breakdown(basket_out["df_results"])
            summary_rows.append({
                "Scenario": scen["short_name"],
                "Strategy": f"Basket (Top {config.TOP_K_BASKETS})",
                "Interval": config.ACTIVE_INTERVAL,
                "Trades": m_b["total_trades"],
                "Gross_PnL": c_b["gross_pnl"],
                "Total_Costs": c_b["total_costs"],
                "Net_PnL": m_b["total_pnl"],
                "Return_Pct": m_b["total_return_pct"],
                "Win_Rate": m_b["win_rate_pct"],
                "Profit_Factor": m_b["profit_factor"],
                "Max_DD_Pct": m_b["max_drawdown_pct"],
                "Sharpe": m_b["sharpe_ratio"]
            })
            cost_drag_records.append({
                "label": f"{scen_id}\n(Basket)",
                "gross_pnl": c_b["gross_pnl"],
                "total_costs": c_b["total_costs"],
                "net_pnl": c_b["net_pnl"]
            })
            plotter.plot_strategy_tearsheet(
                strategy_name=f"{scen['short_name']} — Johansen Basket",
                subtitle=f"Window: {config.TRAIN_MONTHS}m/{config.TEST_MONTHS}m | Interval: {config.ACTIVE_INTERVAL}",
                zscore_data_oos=basket_out["zscore_oos"],
                df_results=basket_out["df_results"],
                metrics=m_b,
                entry_threshold=config.BASKET_SIGNAL_PARAMS["entry_threshold"],
                exit_threshold=config.BASKET_SIGNAL_PARAMS["exit_threshold"],
                sl_threshold=config.BASKET_SIGNAL_PARAMS["SL_threshold"],
                save_path=f"results/plots/{scen_id}_basket_tearsheet.png"
            )
            if config.ACTIVE_INTERVAL == "1d" and scen_id in ["S1_Baseline_1d", "S2_Diversified_1d", "S4_TechOnly_1d", "S8_JohansenSolo_1d"]:
                daily_basket_curves[scen["short_name"]] = (basket_out["df_results"], basket_out["zscore_oos"])

        # Zapis wykresu porównawczego (Pairs vs Basket) dla danego scenariusza
        if pairs_out and basket_out:
            plotter.plot_strategies_comparison(
                pairs_data=pairs_out,
                basket_data=basket_out,
                interval_label=config.ACTIVE_INTERVAL,
                scenario_title=scen["short_name"],
                save_path=f"results/plots/{scen_id}_comparison.png"
            )

    print("\n" + "="*80)
    print("📊 KROK 3/3: GENEROWANIE TABEL LATEX ORAZ WYKRESÓW SYNTETYCZNYCH")
    print("="*80)

    # 1. Zbiorcza tabela CSV i LaTeX
    df_master = pd.DataFrame(summary_rows)
    df_master.to_csv("results/tables/master_results_summary.csv", index=False)
    generate_latex_master_table(summary_rows, "results/tables/master_results_table.tex")

    # 2. Wykres porównujący krzywe kapitału Koszyków Johansena między scenariuszami
    if daily_basket_curves:
        plotter.plot_multi_scenario_equity(
            curves_dict=daily_basket_curves,
            title="Johansen Basket Equity Curves Across Key Daily Scenarios (2020–2024)",
            save_path="results/plots/summary_basket_scenarios_equity.png"
        )

    # 3. Wykres porównujący krzywe kapitału Par Engle-Grangera między scenariuszami
    if daily_pairs_curves:
        plotter.plot_multi_scenario_equity(
            curves_dict=daily_pairs_curves,
            title="Engle-Granger Pairs Equity Curves Across Key Daily Scenarios (2020–2024)",
            save_path="results/plots/summary_pairs_scenarios_equity.png"
        )

    # 4. Wykres wpływu kosztów transakcyjnych (Gross PnL vs Costs vs Net PnL)
    plotter.plot_cost_drag_analysis(
        cost_records=cost_drag_records,
        save_path="results/plots/summary_cost_drag_analysis.png"
    )

    print("\n✅ WSZYSTKIE EKSPERYMENTY ZAKOŃCZONE SUKCESEM!")
    print("-> Wykresy PNG (300 DPI) zapisano w: results/plots/")
    print("-> Tabela CSV zapisana w:            results/tables/master_results_summary.csv")
    print("-> Gotowa tabela LaTeX zapisana w:   results/tables/master_results_table.tex\n")
    print(df_master.to_string(index=False))


if __name__ == "__main__":
    main()