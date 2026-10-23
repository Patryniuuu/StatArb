import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Tuple, Dict, Any, Optional


class StrategyPlotter:
    """
    Moduł wizualizacji wyników strategii Statistical Arbitrage.
    Generuje zsynchronizowane wykresy: Equity Curve, Underwater Drawdown oraz Z-Score z sygnałami.
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # def _build_daily_equity(
    #     self, 
    #     df_results: pd.DataFrame, 
    #     daily_dates: pd.DatetimeIndex
    # ) -> Tuple[pd.Series, pd.Series]:
    #     """
    #     Przekształca tabelę zamkniętych transakcji na dzienny szereg czasowy kapitału (Equity)
    #     oraz dzienny szereg obsunięcia kapitału (Drawdown %).
    #     """
    #     daily_pnl = pd.Series(0.0, index=daily_dates)

    #     if not df_results.empty and "net_pnl" in df_results.columns:
    #         df_temp = df_results.copy()
    #         df_temp["close_dt"] = pd.to_datetime(df_temp["close_date"])
    #         # Sumujemy PnL z transakcji zamkniętych danego dnia
    #         grouped_pnl = df_temp.groupby("close_dt")["net_pnl"].sum()
            
    #         # Dopasowujemy do osi dni sesyjnych OOS
    #         valid_dates = grouped_pnl.index.intersection(daily_dates)
    #         daily_pnl.loc[valid_dates] = grouped_pnl.loc[valid_dates]

    #     equity_series = self.initial_capital + daily_pnl.cumsum()
    #     running_max = np.maximum.accumulate(equity_series)
    #     drawdown_pct = ((equity_series - running_max) / running_max) * 100.0

    #     return equity_series, drawdown_pct

    def _build_daily_equity(
        self, 
        df_results: pd.DataFrame, 
        daily_dates: pd.DatetimeIndex
    ) -> Tuple[pd.Series, pd.Series]:
        daily_pnl = pd.Series(0.0, index=daily_dates)

        if not df_results.empty and "net_pnl" in df_results.columns:
            df_temp = df_results.copy()
            df_temp["close_dt"] = pd.to_datetime(df_temp["close_date"])
            grouped_pnl = df_temp.groupby("close_dt")["net_pnl"].sum()
            
            valid_dates = grouped_pnl.index.intersection(daily_dates)
            daily_pnl.loc[valid_dates] = grouped_pnl.loc[valid_dates]

        equity_series = self.initial_capital + daily_pnl.cumsum()
        running_max = np.maximum.accumulate(equity_series)
        drawdown_pct = ((equity_series - running_max) / running_max) * 100.0

        return equity_series, drawdown_pct

    def plot_strategy_tearsheet(
        self,
        strategy_name: str,
        subtitle: str,
        zscore_data_oos: List[Tuple[Any, float]],
        df_results: pd.DataFrame,
        metrics: Dict[str, float],
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.5,
        sl_threshold: float = 3.5,
        save_path: Optional[str] = None
    ) -> None:
        """
        Rysuje 3-panelowy raport dla pojedynczej strategii:
        1. Krzywa kapitału (Equity Curve) + High-Water Mark
        2. Obsunięcie kapitału (Drawdown %)
        3. Szereg czasowy Z-Score z zaznaczonymi wejściami (Long/Short) i wyjściami
        """
        if not zscore_data_oos:
            print(f"Brak danych Z-Score do narysowania wykresu dla: {strategy_name}")
            return

        # 1. Przygotowanie szeregu Z-Score na osi dat
        df_z = pd.DataFrame(zscore_data_oos, columns=["Date", "ZScore"])
        df_z["Date"] = pd.to_datetime(df_z["Date"])
        df_z["ZScore"] = df_z["ZScore"].astype(float)
        df_z.set_index("Date", inplace=True)
        df_z.sort_index(inplace=True)

        # 2. Budowa dziennej krzywej kapitału i drawdownu na tej samej osi czasu
        equity_series, drawdown_pct = self._build_daily_equity(df_results, df_z.index)
        running_max = np.maximum.accumulate(equity_series)

        # 3. Inicjalizacja figury (3 zsynchronizowane wiersze)
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1,
            figsize=(14, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.2, 2.5]}
        )
        fig.suptitle(
            f"{strategy_name} — {subtitle}\n"
            f"Net PnL: ${metrics['total_pnl']:,.2f} ({metrics['total_return_pct']:.2f}%) | "
            f"Trades: {metrics['total_trades']} | Win Rate: {metrics['win_rate_pct']:.1f}% | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | Max DD: {metrics['max_drawdown_pct']:.2f}%",
            fontsize=13,
            fontweight="bold",
            y=0.98
        )

        # ==========================================
        # PANEL 1: KRZYWA KAPITAŁU (EQUITY CURVE)
        # ==========================================
        ax1.plot(equity_series.index, equity_series.values, color="#1f77b4", lw=2.0, label="Kapitał portfela (Equity)")
        ax1.plot(running_max.index, running_max.values, color="#2ca02c", lw=1.0, ls="--", alpha=0.8, label="Szczyt kapitału (High-Water Mark)")
        ax1.axhline(self.initial_capital, color="gray", lw=1.0, ls=":", alpha=0.7)
        ax1.set_ylabel("Kapitał (USD)", fontsize=10, fontweight="bold")
        ax1.legend(loc="upper left", frameon=True)
        ax1.grid(True, alpha=0.3)

        # ==========================================
        # PANEL 2: DRAWDOWN (UNDERWATER PLOT)
        # ==========================================
        ax2.fill_between(drawdown_pct.index, drawdown_pct.values, 0, color="#d62728", alpha=0.35, label="Drawdown (%)")
        ax2.plot(drawdown_pct.index, drawdown_pct.values, color="#d62728", lw=1.2)
        ax2.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
        ax2.set_ylim(min(drawdown_pct.min() * 1.2, -1.0), 0.2)
        ax2.legend(loc="lower left", frameon=True)
        ax2.grid(True, alpha=0.3)

        # ==========================================
        # PANEL 3: Z-SCORE ORAZ SYGNAŁY WEJŚCIA/WYJŚCIA
        # ==========================================
        ax3.plot(df_z.index, df_z["ZScore"], color="#333333", lw=1.2, alpha=0.85, label="Z-Score")

        # Poziomy progowe
        ax3.axhline(0.0, color="black", lw=1.0, ls="-", alpha=0.5)
        ax3.axhline(entry_threshold, color="#d62728", lw=1.2, ls="--", alpha=0.7, label=f"Próg wejścia (±{entry_threshold})")
        ax3.axhline(-entry_threshold, color="#2ca02c", lw=1.2, ls="--", alpha=0.7)
        ax3.axhline(exit_threshold, color="#1f77b4", lw=1.0, ls=":", alpha=0.7, label=f"Próg wyjścia (±{exit_threshold})")
        ax3.axhline(-exit_threshold, color="#1f77b4", lw=1.0, ls=":", alpha=0.7)
        ax3.axhline(sl_threshold, color="#8c564b", lw=1.0, ls="-.", alpha=0.6, label=f"Stop-Loss (±{sl_threshold})")
        ax3.axhline(-sl_threshold, color="#8c564b", lw=1.0, ls="-.", alpha=0.6)

        # Nanoszenie markerów wejścia i wyjścia z tabeli transakcji
        if not df_results.empty:
            entry_dates = pd.to_datetime(df_results["entry_date"])
            close_dates = pd.to_datetime(df_results["close_date"])
            positions = df_results["position"]

            long_entries_dt = entry_dates[positions == "long"]
            short_entries_dt = entry_dates[positions == "short"]

            # Dopasowanie wartości Z-Score dla dat transakcji
            valid_long = long_entries_dt[long_entries_dt.isin(df_z.index)]
            valid_short = short_entries_dt[short_entries_dt.isin(df_z.index)]
            valid_exits = close_dates[close_dates.isin(df_z.index)]

            ax3.scatter(
                valid_long, df_z.loc[valid_long, "ZScore"],
                marker="^", color="#2ca02c", edgecolors="black", s=90, zorder=5, label="Wejście LONG"
            )
            ax3.scatter(
                valid_short, df_z.loc[valid_short, "ZScore"],
                marker="v", color="#d62728", edgecolors="black", s=90, zorder=5, label="Wejście SHORT"
            )
            ax3.scatter(
                valid_exits, df_z.loc[valid_exits, "ZScore"],
                marker="o", color="#ffbf00", edgecolors="black", s=65, zorder=5, label="Wyjście (Zamknięcie)"
            )

        ax3.set_ylabel("Z-Score (σ)", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Data (Out-of-Sample)", fontsize=10, fontweight="bold")
        ax3.legend(loc="upper right", ncol=3, fontsize=8.5, frameon=True)
        ax3.grid(True, alpha=0.3)

        # Formatowanie osi dat
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())        
        plt.xticks(rotation=30)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        if save_path:
            plt.savefig(save_path, dpi=150)

    def plot_strategies_comparison(
        self,
        pairs_data: Dict[str, Any],
        basket_data: Dict[str, Any],
        interval_label: str = "1d"  # <--- DODANO: informacja o interwale w tytule
    ) -> None:
        """
        Zestawia na jednym wykresie Krzywą Kapitału i Drawdown obu strategii (Pairs vs. Johansen Basket)
        oraz symuluje połączony portfel (50% Pairs + 50% Basket).
        """
        # Oś czasu z danych OOS
        dates_p = pd.to_datetime([x[0] for x in pairs_data["zscore_oos"]])
        dates_b = pd.to_datetime([x[0] for x in basket_data["zscore_oos"]])
        common_dates = dates_p.union(dates_b).sort_values()
        
        # <--- DODANO: Dynamiczne odczytanie rzeczywistego zakresu dat z danych OOS
        start_str = common_dates.min().strftime("%Y-%m-%d") if len(common_dates) > 0 else "N/A"
        end_str = common_dates.max().strftime("%Y-%m-%d") if len(common_dates) > 0 else "N/A"

        eq_pairs, dd_pairs = self._build_daily_equity(pairs_data["df_results"], common_dates)
        eq_basket, dd_basket = self._build_daily_equity(basket_data["df_results"], common_dates)

        # Portfel łączony (suma zysków z obu strategii na wspólnym kapitale bazowym)
        combined_pnl = (eq_pairs - self.initial_capital) + (eq_basket - self.initial_capital)
        eq_combined = self.initial_capital + combined_pnl
        running_max_comb = np.maximum.accumulate(eq_combined)
        dd_combined = ((eq_combined - running_max_comb) / running_max_comb) * 100.0

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )
        # <--- ZMIANA: Dynamiczny tytuł zamiast sztywnego "(2022-2023)"!
        fig.suptitle(
            f"PORÓWNANIE STRATEGII OUT-OF-SAMPLE ({start_str} do {end_str} | Interwał: {interval_label})\n"
            f"PAIRS TRADING (PnL: ${pairs_data['metrics']['total_pnl']:,.2f}, Sharpe: {pairs_data['metrics']['sharpe_ratio']:.2f}) vs. "
            f"JOHANSEN BASKET (PnL: ${basket_data['metrics']['total_pnl']:,.2f}, Sharpe: {basket_data['metrics']['sharpe_ratio']:.2f})",
            fontsize=12,
            fontweight="bold",
            y=0.98
        )

        # Panel 1: Krzywe kapitału
        ax1.plot(eq_pairs.index, eq_pairs.values, label=f"Pairs Trading ({pairs_data['label']})", color="#1f77b4", lw=1.8)
        ax1.plot(eq_basket.index, eq_basket.values, label=f"Johansen Basket ({basket_data['label']})", color="#ff7f0e", lw=1.8)
        ax1.plot(eq_combined.index, eq_combined.values, label="Portfel Połączony (Pairs + Basket)", color="#2ca02c", lw=2.4, ls="-")
        ax1.axhline(self.initial_capital, color="gray", ls=":", lw=1.0)
        ax1.set_ylabel("Kapitał Portfela (USD)", fontsize=10, fontweight="bold")
        ax1.legend(loc="upper left", frameon=True, fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Porównanie Drawdownów
        ax2.plot(dd_pairs.index, dd_pairs.values, label="DD Pairs (%)", color="#1f77b4", lw=1.2, alpha=0.8)
        ax2.plot(dd_basket.index, dd_basket.values, label="DD Basket (%)", color="#ff7f0e", lw=1.2, alpha=0.8)
        ax2.fill_between(dd_combined.index, dd_combined.values, 0, color="#2ca02c", alpha=0.2, label="DD Portfel Połączony (%)")
        ax2.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Data (Out-of-Sample)", fontsize=10, fontweight="bold")
        ax2.legend(loc="lower left", frameon=True, fontsize=9)
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=30)

        plt.tight_layout(rect=[0, 0, 1, 0.94])