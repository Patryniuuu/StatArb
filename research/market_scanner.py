import pandas as pd
import numpy as np
from itertools import combinations
from cointegration_math import CointegrationMath
# Importujemy nasze struktury i silnik z poprzedniego pliku (zakładając, że są w cointegration_math.py)
# from cointegration_math import CointegrationMath

# ==========================================
# 1. STRUKTURA DANYCH
# ==========================================

class PairAnalysisResult:
    """Raport z analizy pojedynczej pary."""
    def __init__(self, asset_y: str, asset_x: str, pval_adf: float, half_life: float, beta: float):
        self.asset_y = asset_y
        self.asset_x = asset_x
        self.pval_adf = pval_adf
        self.half_life = half_life
        self.beta = beta


# ==========================================
# 2. KLASA SKANERA (Z użyciem klasycznego OOP)
# ==========================================

class MarketScanner:
    """Odpowiada za iterację po sektorze i selekcję par do arbitrażu."""
    
    def __init__(self, max_pvalue: float = 0.05, min_half_life: float = 1.0, max_half_life: float = 30.0):
        # Skaner "zapamiętuje" kryteria akceptacji par
        self.max_pvalue = max_pvalue
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life

    def scan_sector(self, df_prices: pd.DataFrame, df_market_cap: pd.DataFrame) -> pd.DataFrame:
        """
        Główna pętla skanująca.
        df_prices: DataFrame z logarytmami cen spółek.
        df_market_cap: DataFrame z kapitalizacją (z kolumnami 'ticker' i 'marketCap').
        Zwraca: DataFrame z posortowanym rankingiem.
        """
        dopuszczone_pary = []
        tickers = df_prices.columns.to_list()
        
        # Słownik dla szybszego wyszukiwania kapitalizacji
        cap_dict = dict(zip(df_market_cap['ticker'], df_market_cap['marketCap']))

        for pair in combinations(tickers, 2):
            capX = cap_dict[pair[0]]
            capY = cap_dict[pair[1]]
            
            # 1. FAZA PRZYGOTOWANIA DANYCH (Tylko to różni się w zależności od kapitalizacji)
            if capX > capY:
                objasniajaca = pair[0]
                objasniana = pair[1]
                # Pamiętaj o .to_numpy(), bo nasz silnik matematyczny tego oczekuje!
                X = np.log(df_prices[pair[0]]).to_numpy()
                Y = np.log(df_prices[pair[1]]).to_numpy()
            else:
                objasniajaca = pair[1]
                objasniana = pair[0]
                X = np.log(df_prices[pair[1]]).to_numpy()
                Y = np.log(df_prices[pair[0]]).to_numpy()

            ols_res = CointegrationMath.calculate_ols(Y, X)
            
            adf_res = CointegrationMath.check_adf(
                residuals=ols_res.residuals,
                max_pvalue=self.max_pvalue
            )
            
            # Szybki filtr ADF
            if not adf_res.is_stationary:
                continue
                
            half_life_res = CointegrationMath.calculate_half_life(ols_res.residuals)
            
            # Szybki filtr Half-Life
            if not (self.min_half_life <= half_life_res.half_life <= self.max_half_life):
                continue
                
            # 3. SUKCES - Dodajemy do listy
            dopuszczone_pary.append(
                PairAnalysisResult(
                    asset_y=objasniana,
                    asset_x=objasniajaca,
                    pval_adf=adf_res.pval,
                    half_life=half_life_res.half_life,
                    beta=ols_res.beta
                )
            )

        # 4. ZAKOŃCZENIE PĘTLI I BUDOWA RANKINGU
        if not dopuszczone_pary:
            # Zabezpieczenie, jeśli żadna para nie przetrwa filtrów
            return pd.DataFrame()
        
        #zamiana na dataframea    
        df_ranking = pd.DataFrame([vars(p) for p in dopuszczone_pary])
        
        # Sortujemy od najlepszego (najkrótszego) Half-Life
        df_ranking = df_ranking.sort_values(by="half_life", ascending=True).reset_index(drop=True)
        
        return df_ranking