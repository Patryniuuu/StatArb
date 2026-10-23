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
        dopuszczone_pary = []
        
        # <--- DODANO: zostawiamy tylko spółki posiadające komplet cen w badanym oknie treningowym
        valid_cols = df_prices.columns[df_prices.notna().all()].to_list()
        if len(valid_cols) < 2:
            return pd.DataFrame()

        df_clean = df_prices[valid_cols]
        
        # <--- OPTYMALIZACJA: liczymy logarytmy i macierz korelacji zwrotów RAZ przed pętlą
        df_log = np.log(df_clean)
        corr_matrix = df_clean.pct_change().dropna().corr()

        cap_dict = dict(zip(df_market_cap['ticker'], df_market_cap['marketCap'])) if df_market_cap is not None else {}

        for pair in combinations(valid_cols, 2):
            t0, t1 = pair[0], pair[1]

            # <--- SZYBKI FILTR 0: korelacja zwrotów z gotowej macierzy (O(1) zamiast liczenia Series.corr w pętli)
            ret_corr = corr_matrix.loc[t0, t1]
            if pd.isna(ret_corr) or ret_corr < self.min_return_corr:
                continue

            capX = cap_dict.get(t0)
            capY = cap_dict.get(t1)

            # Jeśli brak kapitalizacji z API, większą spółkę wyznaczamy po niższej zmienności zwrotów
            if capX is None or capY is None or pd.isna(capX) or pd.isna(capY):
                 compare_flag = df_clean[t0].pct_change().std() <= df_clean[t1].pct_change().std()
            else:
                compare_flag = (capX > capY)

            if compare_flag:
                objasniajaca, objasniana = t0, t1
            else:
                objasniajaca, objasniana = t1, t0

            X = df_log[objasniajaca].to_numpy()
            Y = df_log[objasniana].to_numpy()

            # 1. PIERWSZY FILTR: OLS + Walidacja Beta i R^2
            ols_res = CointegrationMath.calculate_ols(Y, X)
            if not (self.min_beta <= ols_res.beta <= self.max_beta):
                continue
            if ols_res.r_squared < self.min_r_squared:
                continue

            # 2. DRUGI FILTR: Half-Life (szybszy niż ADF i ECM - robimy go przed nimi!)
            half_life_res = CointegrationMath.calculate_half_life(ols_res.residuals)
            if not (self.min_half_life <= half_life_res.half_life <= self.max_half_life):
                continue

            # 3. TRZECI FILTR: ADF
            adf_res = CointegrationMath.check_adf(
                residuals=ols_res.residuals,
                max_pvalue=self.max_pvalue
            )
            if not adf_res.is_stationary:
                continue

            # 4. CZWARTY FILTR: Error Correction Model (ECM)
            ecm_res = CointegrationMath.fit_ecm(
                logX=X, 
                logY=Y, 
                residuals=ols_res.residuals, 
                significance_lvl=0.05
            )
            if not ecm_res.is_valid_ecm:
                continue

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

        if not dopuszczone_pary:
            return pd.DataFrame()
        
        df_ranking = pd.DataFrame([vars(p) for p in dopuszczone_pary])
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

# W pliku research/market_scanner.py — ZASTĄP klasę BasketScanner:

class BasketScanner:
    def __init__(
        self, 
        basket_size: int = 3, 
        significance_level: int = 1,
        min_half_life: float = 1.0,
        max_half_life: float = 30.0,
        min_pair_corr: float = 0.40  # <--- DODANO: minimalna korelacja wewnątrz koszyka (chroni przed eksplozją kombinacji)
    ):
        self.basket_size = basket_size
        self.significance_level = significance_level
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.min_pair_corr = min_pair_corr

    def scan_sector(self, df_prices: pd.DataFrame) -> pd.DataFrame:
        dopuszczone_koszyki = []
        valid_cols = df_prices.columns[df_prices.notna().all()].to_list()
        if len(valid_cols) < self.basket_size:
            return pd.DataFrame()

        df_log_all = np.log(df_prices[valid_cols])
        corr_matrix = df_prices[valid_cols].pct_change().dropna().corr()

        for basket in combinations(valid_cols, self.basket_size):
            # <--- SZYBKI PRE-FILTR: odrzucamy trójki nieskorelowanych spółek przed uruchomieniem Johansena
            if self.basket_size == 3:
                c1 = corr_matrix.loc[basket[0], basket[1]]
                c2 = corr_matrix.loc[basket[0], basket[2]]
                c3 = corr_matrix.loc[basket[1], basket[2]]
                if min(c1, c2, c3) < self.min_pair_corr:
                    continue

            df_log_prices = df_log_all[list(basket)]

            johansen_res = CointegrationMath.calculate_johansen(
                log_prices=df_log_prices,
                det_order=0,
                k_ar_diff=1,
                significance_level=self.significance_level
            )

            if johansen_res is None or not johansen_res.is_cointegrated:
                continue

            raw_weights = johansen_res.weights["Spread_1"].to_dict()
            clean_weights = {ticker: float(np.real(w)) for ticker, w in raw_weights.items()}
            
            # <--- DODANO: Weryfikacja Half-Life dla koszyka wewnątrz skanera!
            spread_train = sum(clean_weights[t] * df_log_prices[t] for t in basket).to_numpy()
            hl_res = CointegrationMath.calculate_half_life(spread_train)
            if not (self.min_half_life <= hl_res.half_life <= self.max_half_life):
                continue

            max_eig = float(np.real(johansen_res.eigenvalues[0]))
            dopuszczone_koszyki.append(
                BasketAnalysisResult(
                    basket=basket,
                    rank=johansen_res.rank,
                    max_eigenvalue=max_eig,
                    weights=clean_weights
                )
            )

        if not dopuszczone_koszyki:
            return pd.DataFrame()

        df_ranking = pd.DataFrame([vars(k) for k in dopuszczone_koszyki])
        df_ranking = df_ranking.sort_values(by="max_eigenvalue", ascending=False).reset_index(drop=True)
        return df_ranking
