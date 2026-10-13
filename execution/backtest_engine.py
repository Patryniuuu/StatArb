import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


class BacktestEngine:
    """
    Symulator egzekucji zleceń, księgowości portfela, kosztów tarcia rynkowego
    oraz kalkulacji metryk ryzyka dla strategii Statistical Arbitrage.
    """
    def __init__(
        self, 
        initial_capital: float = 100_000.0, 
        commission_bps: float = 3.0,          # Prowizja brokerska (3 bps = 0.03%)  
        slippage_bps: float = 3.0,            # Poślizg cenowy + spread Bid/Ask (3 bps = 0.03%)
        borrow_rate_annual: float = 0.01,     # Koszt pożyczki akcji do shorta (1.0% rocznie)
        allocation_per_trade: float = 0.10,   # Maksymalny kapitał na jedną parę (np. 10%)
        execution_mode: str = 'classic'       # 'classic' (Long+Short) lub 'follower'
    ):
        self.initial_capital = initial_capital
        self.allocation_per_trade = allocation_per_trade
        self.execution_mode = execution_mode
        
        # Jednostronny koszt wejścia/wyjścia na jednej nodze (prowizja + slippage)
        self.friction_per_leg = (commission_bps + slippage_bps) / 10_000.0
        
        # Dzienny koszt pożyczki dla pozycji krótkiej (365 dni sesyjne w roku)
        self.daily_borrow_rate = borrow_rate_annual / 365.0

    def match_execution_prices(
        self, 
        trades: List[Dict[str, Any]], 
        df_open_prices: pd.DataFrame, 
        pair: Tuple[str, str]
    ) -> pd.DataFrame:
        """
        Krok 1: Mapuje sygnały z SignalGenerator na rzeczywiste ceny egzekucji Next Open.
        """
        # 1. Zabezpieczenie przed pustą listą transakcji
        if not trades:
            return pd.DataFrame()

        # 2. Konwersja do DataFrame
        df_trades = pd.DataFrame(trades)
        asset_y, asset_x = pair

        # 3. Przygotowanie cen Next Open (przesunięcie w tył o 1 okres: cena otwarcia kolejnego bara)
        next_open = df_open_prices.shift(-1)

        # 4. Mapowanie cen egzekucji na podstawie dat sygnałów
        df_trades['entry_y'] = df_trades['entry_date'].map(next_open[asset_y]) #next open[asset_y] to pd.Series ktory zawiera ceny i na indeksie sa daty, a map porownuje df_trades[daty] i przypisuje tam ceny z asset_y
        df_trades['exit_y'] = df_trades['close_date'].map(next_open[asset_y])
        df_trades['entry_x'] = df_trades['entry_date'].map(next_open[asset_x])
        df_trades['exit_x'] = df_trades['close_date'].map(next_open[asset_x])

        # 5. Odrzucenie transakcji, dla których nie ma ceny Next Open (np. sygnał na ostatnim barze w danych)
        df_trades.dropna(subset=['entry_y', 'exit_y', 'entry_x', 'exit_x'], inplace=True)
        df_trades.reset_index(drop=True, inplace=True)

        return df_trades
    
    def calculate_trades_pnl(
        self, 
        df_trades_with_prices: pd.DataFrame, 
        beta: float
    ) -> pd.DataFrame:
        """
        Krok 2: Wylicza stopę zwrotu brutto, koszty transakcyjne (friction + short borrow)
        oraz ostateczny PnL netto w dolarach.
        
        :param df_trades_with_prices: Tabela z cenami egzekucji z metody match_execution_prices.
        :param beta: Współczynnik hedge ratio z OLS.
        :return: DataFrame z dodanymi kolumnami:
                 ['gross_pnl', 'execution_cost', 'borrow_cost', 'net_pnl', 'net_return_pct'].
        """
        #Obliczenie budżetu na transakcje i kwot na akcje Y i X
        budget_per_trade = self.initial_capital * self.allocation_per_trade
        capital_y = budget_per_trade/(1+abs(beta))
        capital_x = abs(beta) * capital_y
        
        #Obliczenie zwrotów dla Y i X
        df_trades_with_prices.loc[df_trades_with_prices['position'] == 'long', 'return_y'] = (df_trades_with_prices['exit_y'] - df_trades_with_prices['entry_y'])/df_trades_with_prices['entry_y']
        df_trades_with_prices.loc[df_trades_with_prices['position'] == 'long', 'return_x'] = (df_trades_with_prices['entry_x'] - df_trades_with_prices['exit_x'])/df_trades_with_prices['entry_x']
        df_trades_with_prices.loc[df_trades_with_prices['position'] == 'short', 'return_x'] = (df_trades_with_prices['exit_x'] - df_trades_with_prices['entry_x'])/df_trades_with_prices['entry_x']
        df_trades_with_prices.loc[df_trades_with_prices['position'] == 'short', 'return_y'] = (df_trades_with_prices['entry_y'] - df_trades_with_prices['exit_y'])/df_trades_with_prices['entry_y']

        #Obliczenie zysku brutto i kosztów transakcji (ważne jest wzięcie pod uwagę znaku bety)
        df_trades_with_prices['gross_pnl'] = (capital_y * df_trades_with_prices['return_y']) + (np.sign(beta)*capital_x * df_trades_with_prices['return_x'])        
        df_trades_with_prices['execution_cost'] = (capital_y + capital_x) * self.friction_per_leg + (capital_y * (df_trades_with_prices['exit_y'] / df_trades_with_prices['entry_y']) + capital_x * (df_trades_with_prices['exit_x'] / df_trades_with_prices['entry_x'])) * self.friction_per_leg
        df_trades_with_prices['days_in_trade'] = (pd.to_datetime(df_trades_with_prices['close_date']) - pd.to_datetime(df_trades_with_prices['entry_date'])).dt.days

        #Obliczenie kosztu pożyczania akcji przy shortowaniu
        df_trades_with_prices['borrow_cost'] = 0.0
        df_trades_with_prices.loc[df_trades_with_prices['position'] == 'short', 'borrow_cost'] += capital_y * df_trades_with_prices['days_in_trade'] * self.daily_borrow_rate
        if beta > 0:
            # Dodatnia beta: X jest na krótko, gdy spread jest na długo (long)
            df_trades_with_prices.loc[df_trades_with_prices['position'] == 'long', 'borrow_cost'] += capital_x * df_trades_with_prices['days_in_trade'] * self.daily_borrow_rate
        else:
            # Ujemna beta: X jest na krótko, gdy spread jest na krótko (short) - w tym wypadku mamy shorta na Y i shorta na X!
            df_trades_with_prices.loc[df_trades_with_prices['position'] == 'short', 'borrow_cost'] += capital_x * df_trades_with_prices['days_in_trade'] * self.daily_borrow_rate

        #Obliczenie zwrotów netto
        df_trades_with_prices['net_pnl'] = df_trades_with_prices['gross_pnl'] - df_trades_with_prices['execution_cost'] - df_trades_with_prices['borrow_cost']
        df_trades_with_prices['net_return_pct'] = (df_trades_with_prices['net_pnl'] / (capital_y + capital_x)) * 100

        return df_trades_with_prices
        

    # ==========================================
    # --- NOWE METODY DLA KOSZYKÓW (JOHANSEN) ---
    # ==========================================

    def match_basket_execution_prices(
        self,
        trades: List[Dict[str, Any]],
        df_open_prices: pd.DataFrame,
        tickers: List[str]
    ) -> pd.DataFrame:
        """Mapuje ceny Next Open dla wszystkich N spółek w koszyku."""
        if not trades:
            return pd.DataFrame()

        df_trades = pd.DataFrame(trades)
        next_open = df_open_prices.shift(-1)

        cols_to_check = []
        for ticker in tickers:
            df_trades[f'entry_{ticker}'] = df_trades['entry_date'].map(next_open[ticker])
            df_trades[f'exit_{ticker}'] = df_trades['close_date'].map(next_open[ticker])
            cols_to_check.extend([f'entry_{ticker}', f'exit_{ticker}'])

        df_trades.dropna(subset=cols_to_check, inplace=True)
        df_trades.reset_index(drop=True, inplace=True)
        return df_trades

    def calculate_basket_trades_pnl(
        self,
        df_trades_with_prices: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Kalkulacja PnL dla dowolnego koszyka N-elementowego na bazie wag z Johansena.
        """
        if df_trades_with_prices.empty:
            return df_trades_with_prices

        budget_per_trade = self.initial_capital * self.allocation_per_trade
        total_abs_weight = sum(abs(w) for w in weights.values())
        
        # Alokacja kapitału proporcjonalnie do modułu wag |w_i|
        cap_alloc = {t: budget_per_trade * (abs(w) / total_abs_weight) for t, w in weights.items()}

        df_trades_with_prices['days_in_trade'] = (
            pd.to_datetime(df_trades_with_prices['close_date']) - 
            pd.to_datetime(df_trades_with_prices['entry_date'])
        ).dt.days

        gross_pnl = np.zeros(len(df_trades_with_prices))
        exec_cost = np.zeros(len(df_trades_with_prices))
        borrow_cost = np.zeros(len(df_trades_with_prices))

        for ticker, weight in weights.items():
            cap = cap_alloc[ticker]
            entry_p = df_trades_with_prices[f'entry_{ticker}']
            exit_p = df_trades_with_prices[f'exit_{ticker}']

            # Zwrot z ruchu ceny
            price_ret = (exit_p - entry_p) / entry_p

            # Kierunek pozycji na danym aktywie
            is_spread_long = (df_trades_with_prices['position'] == 'long')
            direction = np.where(is_spread_long, np.sign(weight), -np.sign(weight))

            leg_pnl = cap * direction * price_ret
            gross_pnl += leg_pnl

            # Koszty wejścia i wyjścia (friction)
            leg_exec = (cap + cap * (exit_p / entry_p)) * self.friction_per_leg
            exec_cost += leg_exec

            # Koszt pożyczki akcji (naliczany tylko, gdy na danym aktywie jesteśmy na krótko)
            is_short_leg = (direction < 0)
            leg_borrow = np.where(is_short_leg, cap * df_trades_with_prices['days_in_trade'] * self.daily_borrow_rate, 0.0)
            borrow_cost += leg_borrow

        df_trades_with_prices['gross_pnl'] = gross_pnl
        df_trades_with_prices['execution_cost'] = exec_cost
        df_trades_with_prices['borrow_cost'] = borrow_cost
        df_trades_with_prices['net_pnl'] = gross_pnl - exec_cost - borrow_cost
        df_trades_with_prices['net_return_pct'] = (df_trades_with_prices['net_pnl'] / budget_per_trade) * 100.0

        return df_trades_with_prices
    
    
    def calculate_performance_metrics(
        self, 
        df_trades_results: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Krok 3: Wylicza syntetyczne metryki portfelowe (Sharpe, Drawdown, Win Rate, Profit Factor).
        
        :param df_trades_results: Wyniki transakcji z metody calculate_trades_pnl.
        :return: Słownik ze statystykami ryzyka i stopy zwrotu.
        """
        # 1. Zabezpieczenie przed pustym zbiorem transakcji (brak sygnałów)
        if df_trades_results.empty or 'net_pnl' not in df_trades_results.columns:
            return {
                'total_trades': 0,
                'total_pnl': 0.0,
                'total_return_pct': 0.0,
                'win_rate_pct': 0.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0
            }

        total_trades = len(df_trades_results)
        net_pnls = df_trades_results['net_pnl'].to_numpy()

        # 2. Total PnL i całkowity zwrot z kapitału
        total_pnl = float(np.sum(net_pnls))
        total_return_pct = (total_pnl / self.initial_capital) * 100.0

        # 3. Win Rate
        winning_trades = np.sum(net_pnls > 0)
        win_rate_pct = (winning_trades / total_trades) * 100.0

        # 4. Profit Factor
        gross_profit = float(np.sum(net_pnls[net_pnls > 0]))
        gross_loss = float(np.abs(np.sum(net_pnls[net_pnls < 0])))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = np.inf if gross_profit > 0 else 0.0

        # 5. Krzywa kapitału i Maximum Drawdown (MDD)
        equity_curve = self.initial_capital + np.cumsum(net_pnls)
        # Dodajemy kapitał początkowy jako punkt t=0
        full_equity = np.insert(equity_curve, 0, self.initial_capital)
        
        # High-Water Mark (kumulatywne maksimum)
        running_max = np.maximum.accumulate(full_equity)
        drawdowns = (full_equity - running_max) / running_max
        max_drawdown_pct = float(np.abs(np.min(drawdowns))) * 100.0

        # 6. Sharpe Ratio (annualizowany)
        # Korzystamy ze stóp zwrotu z transakcji
        if 'net_return_pct' in df_trades_results.columns:
            returns = df_trades_results['net_return_pct'].to_numpy() / 100.0
        else:
            returns = net_pnls / self.initial_capital

        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0.0

        # Wyznaczenie liczby transakcji rocznie na podstawie dat
        if total_trades > 1 and 'close_date' in df_trades_results.columns and 'entry_date' in df_trades_results.columns:
            start_date = pd.to_datetime(df_trades_results['entry_date'].min())
            end_date = pd.to_datetime(df_trades_results['close_date'].max())
            total_days = max((end_date - start_date).days, 1)
            years = total_days / 365.25
            trades_per_year = total_trades / years if years > 0 else total_trades
        else:
            trades_per_year = 252.0  # Przyjęcie standardowego benchmarku sesji

        if std_return > 0:
            sharpe_ratio = float((mean_return / std_return) * np.sqrt(trades_per_year))
        else:
            sharpe_ratio = 0.0

        return {
            'total_trades': total_trades,
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            'win_rate_pct': round(win_rate_pct, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != np.inf else np.inf,
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'sharpe_ratio': round(sharpe_ratio, 2)
        }