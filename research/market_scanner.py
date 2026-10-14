import pandas as pd
import numpy as np
from itertools import combinations
from research.cointegration_math import CointegrationMath
# Importujemy nasze struktury i silnik z poprzedniego pliku (zakładając, że są w cointegration_math.py)
# from cointegration_math import CointegrationMath

# ==========================================
# 1. STRUKTURA DANYCH
# ==========================================

class PairAnalysisResult:
    """Raport z analizy pojedynczej pary."""
    def __init__(self, asset_y: str, asset_x: str, pval_adf: float, half_life: float, beta: float, ecm_alpha_y: float):
        self.asset_y = asset_y
        self.asset_x = asset_x
        self.pval_adf = pval_adf
        self.half_life = half_life
        self.beta = beta
        self.ecm_alpha_y = ecm_alpha_y


# ==========================================
# 2. KLASA SKANERA (Z użyciem klasycznego OOP)
# ==========================================


class MarketScanner:
    def __init__(
        self,
        max_pvalue: float = 0.05,
        min_half_life: float = 1.0,
        max_half_life: float = 30.0,
        min_beta: float = 0.35,        # Minimalna beta (ochrona przed brakiem hedge'a)
        max_beta: float = 2.50,        # Maksymalna beta (ochrona przed przeważeniem nogi X)
        min_r_squared: float = 0.60,   # Minimalne dopasowanie OLS
        min_return_corr: float = 0.45  # Minimalna korelacja dziennych zwrotów
    ):
        self.max_pvalue = max_pvalue
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.min_r_squared = min_r_squared
        self.min_return_corr = min_return_corr

    def _validate_pair(self, s_y: pd.Series, s_x: pd.Series, beta: float, r_squared: float) -> bool:
        # 1. Sprawdzenie granic współczynnika Beta
        if not (self.min_beta <= beta <= self.max_beta):
            return False

        # 2. Sprawdzenie R^2
        if r_squared < self.min_r_squared:
            return False

        # 3. Sprawdzenie korelacji dziennych zwrotów
        ret_y = s_y.pct_change().dropna()
        ret_x = s_x.pct_change().dropna()
        corr = ret_y.corr(ret_x)
        if corr < self.min_return_corr:
            return False

        return True

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
            # Używamy .get(), aby bezpiecznie pobrać kapitalizację
            capX = cap_dict.get(pair[0])
            capY = cap_dict.get(pair[1])
            
            # Zabezpieczenie przed brakami danych (None lub NaN) z API
            if capX is None or capY is None or pd.isna(capX) or pd.isna(capY):
                continue
            
            # 1. FAZA PRZYGOTOWANIA DANYCH
            if capX > capY:
                objasniajaca = pair[0]
                objasniana = pair[1]
                X = np.log(df_prices[pair[0]]).to_numpy()
                Y = np.log(df_prices[pair[1]]).to_numpy()
            else:
                objasniajaca = pair[1]
                objasniana = pair[0]
                X = np.log(df_prices[pair[1]]).to_numpy()
                Y = np.log(df_prices[pair[0]]).to_numpy()

            # 2. PIERWSZY FILTR: OLS + Walidacja jakościowa (Beta, R^2, Korelacja zwrotów)
            ols_res = CointegrationMath.calculate_ols(Y, X)

            # <--- DODANO WYWOŁANIE FILTRU WALIDACYJNEGO (4 linijki):
            if not self._validate_pair(
                s_y=df_prices[objasniana],
                s_x=df_prices[objasniajaca],
                beta=ols_res.beta,
                r_squared=ols_res.r_squared
            ):
                continue
            
            # 3. DRUGI FILTR: ADF
            adf_res = CointegrationMath.check_adf(
                residuals=ols_res.residuals,
                max_pvalue=self.max_pvalue
            )
            
            if not adf_res.is_stationary:
                continue
                
            # 4. TRZECI FILTR: Half-Life (Przywrócony!)
            half_life_res = CointegrationMath.calculate_half_life(ols_res.residuals)
            
            if not (self.min_half_life <= half_life_res.half_life <= self.max_half_life):
                continue
                
            # 5. CZWARTY FILTR: Error Correction Model (ECM)
            ecm_res = CointegrationMath.fit_ecm(
                logX=X, 
                logY=Y, 
                residuals=ols_res.residuals, 
                significance_lvl=0.05
            )
            
            # Jeśli ECM odrzuci parę (brak sprzężenia zwrotnego na nodze Y), przerywamy
            if not ecm_res.is_valid_ecm:
                continue

            # 6. SUKCES - Dodajemy do listy
            dopuszczone_pary.append(
                PairAnalysisResult(
                    asset_y=objasniana,
                    asset_x=objasniajaca,
                    pval_adf=adf_res.pval,
                    half_life=half_life_res.half_life,
                    beta=ols_res.beta,
                    ecm_alpha_y=ecm_res.alpha_y
                )
            )

        # 7. ZAKOŃCZENIE PĘTLI I BUDOWA RANKINGU
        if not dopuszczone_pary:
            # Zabezpieczenie, jeśli żadna para nie przetrwa filtrów
            return pd.DataFrame()
        
        #zamiana na dataframea    
        df_ranking = pd.DataFrame([vars(p) for p in dopuszczone_pary])
        
        # Sortujemy od najlepszego (najkrótszego) Half-Life
        df_ranking = df_ranking.sort_values(by="half_life", ascending=True).reset_index(drop=True)
        
        return df_ranking
    

class BasketAnalysisResult:
    """Raport z analizy kointegracji wielowymiarowej (Johansen)."""
    def __init__(
        self, 
        basket: tuple[str, ...], 
        rank: int, 
        max_eigenvalue: float, 
        weights: dict[str, float]
    ):
        self.basket = basket
        self.rank = rank
        # Największa wartość własna świadczy o sile kointegracji danego wektora
        self.max_eigenvalue = max_eigenvalue
        # Słownik z wagami, np. {'AAPL': 1.0, 'MSFT': -0.4, 'GOOGL': -0.7}
        self.weights = weights

class BasketScanner:
    """
    Skaner rynkowy szukający kointegracji w koszykach N-elementowych 
    przy użyciu testu Johansena.
    """
    def __init__(self, basket_size: int = 3, significance_level: int = 1):
        """
        :param basket_size: Liczba spółek w koszyku (np. 3 dla trypletów).
        :param significance_level: Poziom ufności: 0 (90%), 1 (95%), 2 (99%).
        """
        self.basket_size = basket_size
        self.significance_level = significance_level

    def scan_sector(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Główna pętla skanująca koszyki wielowymiarowe.
        :param df_prices: DataFrame z historycznymi cenami (poziomy, nie logarytmy).
        :return: Posortowany DataFrame z rankingiem koszyków.
        """
        dopuszczone_koszyki = []
        tickers = df_prices.columns.to_list()

        # Tworzymy wszystkie możliwe kombinacje N-elementowe
        for basket in combinations(tickers, self.basket_size):
            # Wycięcie cen tylko dla badanego koszyka i logarytmowanie
            df_basket_prices = df_prices[list(basket)].copy()
            df_log_prices = np.log(df_basket_prices)

            # Uruchomienie testu Johansena
            johansen_res = CointegrationMath.calculate_johansen(
                log_prices=df_log_prices,
                det_order=0,
                k_ar_diff=1,
                significance_level=self.significance_level
            )

            # Odrzucamy koszyki bez kointegracji (rank == 0)
            if johansen_res is None or not johansen_res.is_cointegrated:
                continue

            # Wyciągamy wagi dla najsilniejszego wektora (Spread_1)
            # # Metoda calculate_johansen zwraca DataFrame, z którego robimy słownik
            # best_vector_weights = johansen_res.weights["Spread_1"].to_dict()
            
            # # Wartości własne są posortowane malejąco w eigenvalues[0]
            # max_eig = johansen_res.eigenvalues[0]

            raw_weights = johansen_res.weights["Spread_1"].to_dict()
            clean_weights = {ticker: float(np.real(w)) for ticker, w in raw_weights.items()}
            max_eig = float(np.real(johansen_res.eigenvalues[0]))
            
            dopuszczone_koszyki.append(
                BasketAnalysisResult(
                    basket=basket,
                    rank=johansen_res.rank,
                    max_eigenvalue=max_eig,
                    weights=clean_weights
                )
            )

        # Zabezpieczenie przed brakiem wyników
        if not dopuszczone_koszyki:
            return pd.DataFrame()

        # Zamiana na DataFrame i sortowanie rankingu
        df_ranking = pd.DataFrame([vars(k) for k in dopuszczone_koszyki])
        
        # Sortujemy od największej wartości własnej (najsilniejsza kointegracja)
        df_ranking = df_ranking.sort_values(by="max_eigenvalue", ascending=False).reset_index(drop=True)

        return df_ranking