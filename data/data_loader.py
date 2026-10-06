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
        """
        Pobiera aktualną kapitalizację rynkową dla listy tickerów.
        Zwraca: DataFrame z kolumnami ['ticker', 'marketCap']
        """
        records = []
        for t in tickers:
            tick = yf.Ticker(t)
            # Zabezpieczenie przed brakiem danych w yfinance (częsty problem API)
            cap = tick.fast_info.get('market_cap', None)
            records.append({'ticker': t, 'marketCap': cap})
        
        # Tworzymy czystą tabelę zgodną z kontraktem dla MarketScanner
        df = pd.DataFrame(records)
        return df

    def get_prices(
        self, 
        tickers: List[str], 
        start_date: datetime, 
        end_date: datetime, 
        interval: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Pobiera ceny zamknięcia (Close) dla zadanego uniwersum i przedziału czasu.
        Zwraca: Wyczyszczony DataFrame, gdzie indeksem jest Date/Datetime, a kolumnami tickery.
        """
        # Używamy interwału przekazanego w argumencie, a jeśli brak - bierzemy domyślny
        selected_interval = interval if interval is not None else self.default_interval
        
        data = yf.download(
            tickers=tickers, 
            start=start_date, 
            end=end_date, 
            interval=selected_interval,
            progress=False
        )
        
        # Wyciągamy Close i czyścimy wiersze z brakami
        close_prices = data['Close'].dropna().copy()
        
        return close_prices