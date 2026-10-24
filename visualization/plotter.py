import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Tuple, Dict, Any, Optional


class StrategyPlotter:
    """
     Academic-grade visualization module for Statistical Arbitrage strategies (English labels for LaTeX report).
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.initial_capital = initial_capital
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        # Ustawienia czcionek pod publikację LaTeX
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.edgecolor"] = "#cccccc"
        plt.rcParams["axes.linewidth"] = 0.8

    def _build_daily_equity(
        self, 
        df_results: pd.DataFrame, 
        daily_dates: pd.DatetimeIndex
    ) -> Tuple[pd.Series, pd.Series]:
        # Normalizujemy oś czasu do dni dla spójnego wykresu kapitału (działa dla 1d oraz 1h)
        norm_index = pd.to_datetime(daily_dates).normalize().unique().sort_values()
        daily_pnl = pd.Series(0.0, index=norm_index)

        if not df_results.empty and "net_pnl" in df_results.columns:
            df_temp = df_results.copy()
            df_temp["close_dt"] = pd.to_datetime(df_temp["close_date"]).dt.normalize()
            grouped_pnl = df_temp.groupby("close_dt")["net_pnl"].sum()
            
            valid_dates = grouped_pnl.index.intersection(daily_pnl.index)
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
        if not zscore_data_oos:
            print(f"No Z-Score data to plot for: {strategy_name}")
            return

        df_z = pd.DataFrame(zscore_data_oos, columns=["Date", "ZScore"])
        df_z["Date"] = pd.to_datetime(df_z["Date"])
        df_z["ZScore"] = df_z["ZScore"].astype(float)
        df_z.set_index("Date", inplace=True)
        df_z.sort_index(inplace=True)

        equity_series, drawdown_pct = self._build_daily_equity(df_results, df_z.index)
        running_max = np.maximum.accumulate(equity_series)

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1,
            figsize=(14, 10),
            sharex=False,
            gridspec_kw={"height_ratios": [2.2, 1.2, 2.5]}
        )
        fig.suptitle(
            f"{strategy_name} — {subtitle}\n"
            f"Net PnL: ${metrics['total_pnl']:,.2f} ({metrics['total_return_pct']:.2f}%) | "
            f"Trades: {metrics['total_trades']} | Win Rate: {metrics['win_rate_pct']:.1f}% | "
            f"Sharpe: {metrics['sharpe_ratio']:.2f} | Max DD: {metrics['max_drawdown_pct']:.2f}%",
            fontsize=12.5,
            fontweight="bold",
            y=0.98
        )

        # PANEL 1: EQUITY CURVE
        ax1.plot(equity_series.index, equity_series.values, color="#1f77b4", lw=2.0, label="Portfolio Equity (USD)")
        ax1.plot(running_max.index, running_max.values, color="#2ca02c", lw=1.1, ls="--", alpha=0.8, label="High-Water Mark")
        ax1.axhline(self.initial_capital, color="gray", lw=1.0, ls=":", alpha=0.7, label="Initial Capital")
        ax1.set_ylabel("Equity (USD)", fontsize=10, fontweight="bold")
        ax1.legend(loc="upper left", frameon=True, fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

        # PANEL 2: UNDERWATER DRAWDOWN
        ax2.fill_between(drawdown_pct.index, drawdown_pct.values, 0, color="#d62728", alpha=0.30, label="Underwater Drawdown (%)")
        ax2.plot(drawdown_pct.index, drawdown_pct.values, color="#d62728", lw=1.2)
        ax2.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
        ax2.set_ylim(min(drawdown_pct.min() * 1.25, -0.5), 0.15)
        ax2.legend(loc="lower left", frameon=True, fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())

        # PANEL 3: Z-SCORE & EXECUTIONS
        ax3.plot(df_z.index, df_z["ZScore"], color="#333333", lw=1.1, alpha=0.85, label="Out-of-Sample Z-Score")
        ax3.axhline(0.0, color="black", lw=1.0, ls="-", alpha=0.5)
        ax3.axhline(entry_threshold, color="#d62728", lw=1.2, ls="--", alpha=0.75, label=f"Entry Threshold (±{entry_threshold}σ)")
        ax3.axhline(-entry_threshold, color="#2ca02c", lw=1.2, ls="--", alpha=0.75)
        ax3.axhline(exit_threshold, color="#1f77b4", lw=1.0, ls=":", alpha=0.75, label=f"Exit Threshold (±{exit_threshold}σ)")
        ax3.axhline(-exit_threshold, color="#1f77b4", lw=1.0, ls=":", alpha=0.75)
        ax3.axhline(sl_threshold, color="#8c564b", lw=1.0, ls="-.", alpha=0.65, label=f"Stop-Loss (±{sl_threshold}σ)")
        ax3.axhline(-sl_threshold, color="#8c564b", lw=1.0, ls="-.", alpha=0.65)

        if not df_results.empty:
            entry_dates = pd.to_datetime(df_results["entry_date"])
            close_dates = pd.to_datetime(df_results["close_date"])
            positions = df_results["position"]

            long_entries_dt = entry_dates[positions == "long"]
            short_entries_dt = entry_dates[positions == "short"]

            valid_long = long_entries_dt[long_entries_dt.isin(df_z.index)]
            valid_short = short_entries_dt[short_entries_dt.isin(df_z.index)]
            valid_exits = close_dates[close_dates.isin(df_z.index)]

            ax3.scatter(
                valid_long, df_z.loc[valid_long, "ZScore"],
                marker="^", color="#2ca02c", edgecolors="black", s=75, zorder=5, label="LONG Entry"
            )
            ax3.scatter(
                valid_short, df_z.loc[valid_short, "ZScore"],
                marker="v", color="#d62728", edgecolors="black", s=75, zorder=5, label="SHORT Entry"
            )
            ax3.scatter(
                valid_exits, df_z.loc[valid_exits, "ZScore"],
                marker="o", color="#ffbf00", edgecolors="black", s=55, zorder=5, label="Position Exit"
            )

        ax3.set_ylabel("Z-Score (σ)", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Date (Out-of-Sample)", fontsize=10, fontweight="bold")
        ax3.legend(loc="upper right", ncol=3, fontsize=8.5, frameon=True)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())

        plt.tight_layout(rect=[0, 0, 1, 0.94])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    def plot_strategies_comparison(
        self,
        pairs_data: Dict[str, Any],
        basket_data: Dict[str, Any],
        interval_label: str = "1d",
        scenario_title: str = "",
        save_path: Optional[str] = None
    ) -> None:
        dates_p = pd.to_datetime([x[0] for x in pairs_data["zscore_oos"]])
        dates_b = pd.to_datetime([x[0] for x in basket_data["zscore_oos"]])
        common_dates = dates_p.union(dates_b).sort_values()
        
        start_str = common_dates.min().strftime("%Y-%m-%d") if len(common_dates) > 0 else "N/A"
        end_str = common_dates.max().strftime("%Y-%m-%d") if len(common_dates) > 0 else "N/A"

        eq_pairs, dd_pairs = self._build_daily_equity(pairs_data["df_results"], common_dates)
        eq_basket, dd_basket = self._build_daily_equity(basket_data["df_results"], common_dates)

        # Ujednolicenie indeksu dziennego
        full_idx = eq_pairs.index.union(eq_basket.index).sort_values()
        eq_pairs = eq_pairs.reindex(full_idx).ffill().fillna(self.initial_capital)
        eq_basket = eq_basket.reindex(full_idx).ffill().fillna(self.initial_capital)

        combined_pnl = (eq_pairs - self.initial_capital) + (eq_basket - self.initial_capital)
        eq_combined = self.initial_capital + combined_pnl
        running_max_comb = np.maximum.accumulate(eq_combined)
        dd_combined = ((eq_combined - running_max_comb) / running_max_comb) * 100.0

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(14, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [2.1, 1]}
        )
        prefix = f"{scenario_title} — " if scenario_title else ""
        fig.suptitle(
            f"{prefix}Out-of-Sample Strategy Comparison ({start_str} to {end_str} | Interval: {interval_label})\n"
            f"Engle-Granger Pairs (Net PnL: ${pairs_data['metrics']['total_pnl']:,.2f}, Sharpe: {pairs_data['metrics']['sharpe_ratio']:.2f})  vs.  "
            f"Johansen Basket (Net PnL: ${basket_data['metrics']['total_pnl']:,.2f}, Sharpe: {basket_data['metrics']['sharpe_ratio']:.2f})",
            fontsize=12,
            fontweight="bold",
            y=0.98
        )

        # Panel 1: Equity Curves
        ax1.plot(eq_pairs.index, eq_pairs.values, label=f"Engle-Granger Pairs ({pairs_data['label']})", color="#1f77b4", lw=1.8)
        ax1.plot(eq_basket.index, eq_basket.values, label=f"Johansen Basket ({basket_data['label']})", color="#ff7f0e", lw=1.8)
        ax1.plot(eq_combined.index, eq_combined.values, label="Combined Portfolio (Pairs + Basket)", color="#2ca02c", lw=2.3, ls="-")
        ax1.axhline(self.initial_capital, color="gray", ls=":", lw=1.0)
        ax1.set_ylabel("Portfolio Equity (USD)", fontsize=10, fontweight="bold")
        ax1.legend(loc="upper left", frameon=True, fontsize=9.5)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Drawdowns
        ax2.plot(dd_pairs.index, dd_pairs.values, label="Pairs Drawdown (%)", color="#1f77b4", lw=1.2, alpha=0.85)
        ax2.plot(dd_basket.index, dd_basket.values, label="Basket Drawdown (%)", color="#ff7f0e", lw=1.2, alpha=0.85)
        ax2.fill_between(dd_combined.index, dd_combined.values, 0, color="#2ca02c", alpha=0.20, label="Combined Portfolio DD (%)")
        ax2.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Date (Out-of-Sample)", fontsize=10, fontweight="bold")
        ax2.legend(loc="lower left", frameon=True, fontsize=9)
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=25)
        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

    # <--- NOWY WYKRES 1 DO RAPORTU: Zbiorcze porównanie krzywych kapitału z wielu scenariuszy
    def plot_multi_scenario_equity(
        self,
        curves_dict: Dict[str, Tuple[pd.DataFrame, List[Tuple[Any, float]]]],
        title: str,
        save_path: str
    ) -> None:
        """
        Rysuje na jednym wykresie krzywe kapitału z kilku wybranych scenariuszy
        (np. S1 Baseline vs S2 Diversified vs S4 Tech-Only vs S8 Johansen Solo).
        """
        fig, ax = plt.subplots(figsize=(13, 6.5))
        colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#8c564b"]

        for idx, (label, (df_res, z_oos)) in enumerate(curves_dict.items()):
            if not z_oos:
                continue
            dates = pd.to_datetime([x[0] for x in z_oos])
            eq_series, _ = self._build_daily_equity(df_res, dates)
            color = colors[idx % len(colors)]
            ax.plot(eq_series.index, eq_series.values, label=label, lw=2.0, color=color)

        ax.axhline(self.initial_capital, color="black", ls=":", lw=1.0, alpha=0.7, label="Initial Capital ($100,000)")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Portfolio Equity (USD)", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Date (Out-of-Sample)", fontsize=10.5, fontweight="bold")
        ax.legend(loc="upper left", frameon=True, fontsize=9.5)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # <--- NOWY WYKRES 2 DO RAPORTU: Gross PnL vs. Transaction Costs vs. Net PnL (Friction Drag)
    def plot_cost_drag_analysis(
        self,
        cost_records: List[Dict[str, Any]],
        save_path: str
    ) -> None:
        """
        Pokazuje analizę wpływu kosztów transakcyjnych (Gross PnL vs Friction/Borrow Costs vs Net PnL)
        dla wszystkich scenariuszy. Idealny wykres do omówienia "Friction Bleed" na interwale 1h vs 1d.
        """
        df_c = pd.DataFrame(cost_records)
        if df_c.empty:
            return

        fig, ax = plt.subplots(figsize=(14, 6.5))
        x = np.arange(len(df_c))
        width = 0.27

        ax.bar(x - width, df_c["gross_pnl"], width, label="Gross PnL ($)", color="#2ca02c", alpha=0.85, edgecolor="black")
        ax.bar(x, df_c["total_costs"], width, label="Transaction & Borrow Costs ($)", color="#d62728", alpha=0.85, edgecolor="black")
        ax.bar(x + width, df_c["net_pnl"], width, label="Net Realized PnL ($)", color="#1f77b4", alpha=0.90, edgecolor="black")

        ax.axhline(0, color="black", lw=1.0)
        ax.set_title("Impact of Market Friction: Gross PnL vs. Transaction Costs & Net PnL Across Scenarios", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Amount (USD)", fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(df_c["label"], rotation=25, ha="right", fontsize=9)
        ax.legend(loc="upper right", frameon=True, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)