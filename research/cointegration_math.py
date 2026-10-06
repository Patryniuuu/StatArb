# Czysta statystyka i testy badawcze
import numpy as np
import pandas as pd
from typing import Optional
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from scipy import stats



#tworzymy klasy trzymające wyniki

class OLSResult:
    """Kontener na wyniki regresji liniowej."""
    def __init__(self, alpha: float, beta: float, residuals: np.ndarray):
        self.alpha = alpha
        self.beta = beta
        self.residuals = residuals
        
class HalfLifeResult:
    """Kontener na estymację czasu powrotu do średniej."""
    def __init__(self, half_life: float, theta_param: float):
        self.half_life = half_life
        self.theta_param = theta_param

class ECMResult:
    """Kontener na wyniki Modelu Korekty Błędu (ECM)."""
    def __init__(self, alpha_y: float, pval_y: float, alpha_x: float, pval_x: float, is_valid_ecm: bool):
        self.alpha_y = alpha_y
        self.alpha_x = alpha_x
        self.pval_y = pval_y
        self.pval_x = pval_x
        self.is_valid_ecm = is_valid_ecm

class ADFResult:
    """Kontener na wyniki testu ADF"""
    def __init__(self, t_stat: float, pval: float, used_lags: int, is_stationary: bool):
        self.t_stat = t_stat
        self.pval = pval
        self.used_lags = used_lags
        self.is_stationary = is_stationary

class JohansenResult:
    """Konter na wyniki testu Johansena"""
    def __init__(self, is_cointegrated: bool, rank: int, weights: np.ndarray, eigenvalues: np.ndarray):
        self.is_cointegrated = is_cointegrated
        self.rank = rank
        self.weights = weights
        self.eigenvalues = eigenvalues
        

#Tworzymy klase z wszystkimi matematycznymi testami/obliczeniami

class CointegrationMath:
    
    @staticmethod #static method, bo to jest tylko kalkulator, wzor matematyczny, nie potrzebuje posiadac stanu, obiektu self
    def calculate_ols(log_y: np.ndarray, log_x: np.ndarray) -> OLSResult:
        if len(log_x) != len(log_y):
            raise ValueError("Wektory log_x i log_y muszą mieć tę samą długość!")
        
        # 1. Zbuduj wektor jedynek o długości takiej, jak wektor X
        ones = np.ones(len(log_x))
        # 2. Sklej wektor jedynek z wektorem log_x w jedną macierz (tzw. macierz planu)
        X = np.column_stack((ones,log_x))
        # 3. Zastosuj wzór: B = (X^T * X)^-1 * X^T * Y
        B = np.linalg.inv(X.T @ X) @ X.T @ log_y
        # 4. Wyciągnij alpha (B[0]) i beta (B[1])
        alpha = B[0]
        beta = B[1]
        # 5. Oblicz residuals (log_y - (beta * log_x + alpha))
        residuals = log_y - (beta * log_x + alpha)
        # 6. Zwróć nowy obiekt OLSResult(...)
        return OLSResult(alpha=alpha, beta=beta, residuals=residuals)
    
    @staticmethod
    def calculate_half_life(spread: np.ndarray) -> HalfLifeResult:
        """ Half-Life wyliczamy z równania AR(1) na spreadzie (resztach z OLS).
            Zmienna objaśniająca to spread opóźniony o 1 okres: $X = S_{t-1}$.
            Zmienna objaśniana to różnica spreadu z dzisiaj i wczoraj: $Y = \Delta S$."""
        # 1. Zbuduj wektor opóźniony (X) - u Ciebie w kodzie: S[:-1]
        X = spread[:-1]
        # 3. Zbuduj wektor różnic pierwszego rzędu (Y) - u Ciebie: np.diff(S)
        Y = np.diff(spread)
        # 4. Policz B ze wzoru OLS i wyciągnij zmienną kierunkową (theta_param) - w Twoim notatniku to parametr 'beta' = B[1]
        ols_res = CointegrationMath.calculate_ols(log_y=Y, log_x=X)
        theta_param = ols_res.beta
        # 5. Zabezpiecz się przed brakiem powrotu do średniej (jeśli theta_param >= 0, half_life = np.inf)
        if theta_param >= 0: 
            half_life = np.inf
        # 6. Jeśli theta_param < 0, policz half_life ze wzoru: -np.log(2) / theta_param
        else: 
            half_life = -np.log(2) / theta_param
        # 7. Zwróć obiekt HalfLifeResult
        return HalfLifeResult(half_life=half_life, theta_param=theta_param)

    @staticmethod
    def check_adf(residuals: np.ndarray, max_pvalue: float = 0.05) -> ADFResult:
        """ Wykonuje test ADF w celu sprawdzenia stacjonarnosci spreadu"""
        adf_res = adfuller(
        residuals, 
        regression='n', 
        autolag='AIC', 
        store=True, 
        regresults=True, 
        result_object=True
        )
        
        t_stat = adf_res[0]
        pval = adf_res[1]
        used_lags = adf_res[2]
        is_stationary = bool(pval <= max_pvalue)
        
        return ADFResult(
            t_stat=t_stat, 
            pval=pval, 
            used_lags=used_lags, 
            is_stationary=is_stationary
        )
        
    @staticmethod
    def fit_ecm(logX: np.ndarray, logY: np.ndarray, residuals: np.ndarray, significance_lvl: float) -> ECMResult:
        """Estymuje równania ECM dla obu nóg spreadu oraz oblicza
        błędy standardowe, statystyki t i wartości p-value."""
        
        #tu powinna byc logika sprawdzajaca czy len(residuals) = len(logY)
        if not (len(logX) == len(logY) == len(residuals)):
            raise ValueError("logX, logY oraz residuals muszą mieć identyczną długość.")
        
        #Obliczamy pierwsze różnice cen (długość T-1)
        delta_x = np.diff(logX)
        delta_y = np.diff(logY)
        
        #chcemy policzyc: deltaY = W*b_y + u_y,t
        #b_y = [miu_y (wyraz wolny), alpha_y]
        eps_prev = residuals[:-1]
    
        N_obs = len(eps_prev)  # T - 1
        df = N_obs - 2         # T - 3 stopnie swobody
        
        #Liczymy macierze
        ones = np.ones(N_obs)
        W = np.column_stack([ones, eps_prev]) 
        #rozwiazanie na b to $$\mathbf{b} = (\mathbf{W}^T \mathbf{W})^{-1} \mathbf{W}^T \mathbf{Z}$$, 
        #gdzie Z to odpowiednia delta_x, delta_y
        inv_WTW = np.linalg.inv(W.T @ W)
        inv_WTW_WT = inv_WTW @ W.T
        
        #Wyznacz wektory parametrów dla Y oraz dla X
        b_Y = inv_WTW_WT @ delta_y
        b_X = inv_WTW_WT @ delta_x

        #Wyciagamy alphy
        alpha_y = b_Y[1]
        alpha_x = b_X[1]

        #liczymy e
        e_y = delta_y - (W @ b_Y)
        e_x = delta_x - (W @ b_X)
        
        #liczymy s^2
        s2_y = np.dot(e_y, e_y) / df
        s2_x = np.dot(e_x, e_x) / df
        
        # Błędy standardowe z elementu [1, 1] macierzy s^2 * (W^T W)^(-1)
        se_alpha_y = np.sqrt(s2_y * inv_WTW[1, 1])
        se_alpha_x = np.sqrt(s2_x * inv_WTW[1, 1])
        
        #statystyka t
        t_stat_y = alpha_y/se_alpha_y
        t_stat_x = alpha_x / se_alpha_x

        # Wartości p-value
        # Y: jednostronny lewostronny (H1: alpha_y < 0)
        pval_y = stats.t.cdf(t_stat_y, df=df)

        # X: obustronny (H1: alpha_x != 0)
        pval_x = 2.0*(1.0 - stats.t.cdf(np.abs(t_stat_x), df = df))

        # Flaga dopuszczenia pary pod kątem ECM
        # alpha_y musi być ujemne i istotne statystycznie
        # alpha_x nie może rozrywać spreadu (jeśli istotne, musi być dodatnie)
        is_valid_ecm = (pval_y < significance_lvl) and (alpha_y < 0) and not (pval_x < significance_lvl and alpha_x < 0)
        return ECMResult(alpha_y = alpha_y, 
                         pval_y=pval_y, 
                         alpha_x = alpha_x, 
                         pval_x = pval_x, 
                         is_valid_ecm=is_valid_ecm) 
        
    @staticmethod
    def calculate_johansen(log_prices: pd.DataFrame, date_col=None, det_order: int = 0, k_ar_diff: int = 1, significance_level: int = 1) -> Optional[JohansenResult]:
        """
        Szybki Johansen.
        data_df: DataFrame z logarytmami cen.
        det_order: 1 oznacza dodanie stałej (driftu) do modelu.
        k_ar_diff: liczba opóźnionych różnic (nasze p-1).
        significance_level: 0 = 90%, 1 = 95%, 2 = 99%
        Zwraca JohansenResult jeśli występuje kointegracja, w przeciwnym razie zwraca None.
        """
        # 1. Odizolowanie wyłącznie kolumn z cenami
        df_clean = log_prices.copy()
        if date_col and date_col in df_clean.columns:
            df_clean = df_clean.drop(columns=[date_col])
        elif "Date" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Date"])
        
        # det_order=0: stała w relacji kointegrującej (brak trendu)
        # k_ar_diff=1: 1 opóźniona różnica (odpowiednik VAR(2) w poziomach)
        res = coint_johansen(df_clean, det_order, k_ar_diff)

        num_vars = df_clean.shape[1]
        exact_rank = 0

        for i in range(num_vars):
            trace_stat = res.lr1[i]
            crit_val = res.cvt[i, significance_level]
            
            #Test H0: r <= i przeciwko H1: r > i
            if trace_stat > crit_val:
                exact_rank = i + 1
            else:
                #Pierwszy brak odrzucenia H0 wyznacza dokładny rząd r
                break

        #Jeśli brak kointegracji
        if exact_rank == 0:
            return None

        #Wyciągnięcie i normalizacja tylu wektorów, ile wynosi exact_rank
        #Wektory własne są w kolumnach res.evec
        raw_betas = res.evec[:, :exact_rank]
        
        #Normalizacja każdego wektora przez jego pierwszy element
        #Dzielimy każdą kolumnę przez wartość z jej pierwszego wiersza
        normalized_betas = raw_betas / raw_betas[0, :]
        
        #Przekształcenie w czytelny DataFrame z wagami dla każdej spółki
        weights_df = pd.DataFrame(
            normalized_betas,
            index=df_clean.columns,
            columns=[f"Spread_{i+1}" for i in range(exact_rank)]
        )
        
        return JohansenResult(
            is_cointegrated = True,
            rank = exact_rank,
            weights = weights_df,
            eigenvalues = res.eig[:exact_rank]
        )
            