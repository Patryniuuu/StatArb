import pandas as pd
import numpy as np


#Obliczanie Z-score na podstawie spreadu



class ZScoreCalculator:
    def __init__(self, multiplier: float = 3.0, min_window: int = 15, max_window: int = 90):
        """
        Inicjalizacja kalkulatora Z-score.
        """
        self.multiplier = multiplier
        self.min_window = min_window
        self.max_window = max_window

    
    def calculate(
        self, 
        df_spread: pd.DataFrame, 
        pair_col: str, 
        half_life: float, 
        date_col: str = 'Date'
    ) -> list:
        """
                Oblicza Z-score na podstawie kroczącej średniej i odchylenia standardowego.
                
                :param df_spread: DataFrame z kolumną 'Date' oraz kolumną o nazwie pair_name.
                :param pair_name: Nazwa kolumny ze spreadem, np. ('INTC', 'ORCL').
                :param half_life: Parametr half-life dla danej pary.
                :return: Lista krotek (Data, Z-score) kompatybilna z SignalGenerator.
        """
        window_length = int(np.clip(half_life * self.multiplier, self.min_window, self.max_window))
        
        df = df_spread[[date_col, pair_col]].copy()
        
        # Przesunięcie o 1 bar zapobiega look-ahead bias
        df['res_lagged'] = df[pair_col].shift(1)
        
        df['mean'] = df['res_lagged'].rolling(window=window_length).mean()
        df['std'] = df['res_lagged'].rolling(window=window_length).std()
        
        df['Z_score'] = (df[pair_col] - df['mean']) / df['std']
        df.dropna(inplace=True)
        
        return list(zip(df[date_col], df['Z_score']))