import pandas as pd
import yfinance as yf
from typing import List, Optional
from datetime import datetime

class DataLoader:
    """
    Odpowiada za pobieranie, czyszczenie i standaryzację danych rynkowych.
    """
    def __init__(self, default_interval: str = "1d"):
        self.default_interval = default_interval

    def get_market_caps(self, tickers: List[str]) -> pd.DataFrame:
        records = []
        for t in tickers:
            try:
                tick = yf.Ticker(t)
                cap = tick.fast_info.get('market_cap', None)
            except Exception:
                cap = None
            records.append({'ticker': t, 'marketCap': cap})
        return pd.DataFrame(records)

    def get_prices(
        self, 
        tickers: List[str], 
        start_date: datetime, 
        end_date: datetime, 
        interval: Optional[str] = None,
        return_open: bool = False
    ):
        selected_interval = interval if interval is not None else self.default_interval
        
        data = yf.download(
            tickers=tickers, 
            start=start_date, 
            end=end_date, 
            interval=selected_interval,
            progress=False
        )
        
        # <--- NAPRAWA BŁĘDU NaN NA 1h: Usuwamy strefę czasową na poziomie całego DataFrame 'data'
        # Dzięki temu zarówno 'Close', jak i 'Open' mają identyczny indeks tz-naive!
        if getattr(data.index, 'tz', None) is not None:
            data.index = data.index.tz_localize(None)
        
        close_prices = data['Close'].ffill(limit=3).dropna(how='all').copy()
            
        if return_open:
            open_prices = data['Open'].ffill(limit=3).reindex(close_prices.index).copy()
            return close_prices, open_prices
            
        return close_prices