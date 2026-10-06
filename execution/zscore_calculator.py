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

    def calculate(self, df_spread: pd.DataFrame, pair_name: tuple, half_life: float) -> list:
        """
        Oblicza Z-score na podstawie kroczącej średniej i odchylenia standardowego.
        
        :param df_spread: DataFrame z kolumną 'Date' oraz kolumną o nazwie pair_name.
        :param pair_name: Nazwa kolumny ze spreadem, np. ('INTC', 'ORCL').
        :param half_life: Parametr half-life dla danej pary.
        :return: Lista krotek (Data, Z-score) kompatybilna z SignalGenerator.
        """
        # Obliczenie długości okna z ograniczeniami min/max
        window_length = int(np.clip(half_life * self.multiplier, self.min_window, self.max_window))
        
        # Praca na kopii wycinka danych
        df = df_spread[['Date', pair_name]].copy()
        
        # Przesunięcie o 1 dzień (lag) zapobiega "zaglądaniu w przyszłość" (look-ahead bias)
        # Średnia i odchylenie do dzisiejszego Z-score są liczone ze wczorajszego stanu
        df['res_lagged'] = df[pair_name].shift(1)
        
        # Wyliczenie parametrów rozkładu w oknie kroczącym
        df['mean'] = df['res_lagged'].rolling(window=window_length).mean()
        df['std'] = df['res_lagged'].rolling(window=window_length).std()
        
        # Finalny wynik Z-score na bazie dzisiejszej wartości spreadu
        df['Z_score'] = (df[pair_name] - df['mean']) / df['std']
        
        # Usunięcie braków danych z okresu, gdy okno kroczące się wypełniało
        df.dropna(inplace=True)
        
        # Zwracamy dane od razu w formacie listy krotek, gotowe dla generatora sygnałów
        return list(zip(df['Date'], df['Z_score']))