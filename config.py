
# PARAMETRY SILNIKA DO BACKTESTÓW
ENGINE_PARAMS = {
    'initial_capital': 100_000.0,   # Kapitał początkowy w walucie bazowej
    'commission_bps': 3.0,          # Prowizja brokerska w punktach bazowych (3 bps = 0.03%)
    'slippage_bps': 3.0,            # Zakładany poślizg cenowy i spread bid/ask w punktach bazowych (3 bps = 0.03%)
    'borrow_rate_annual': 0.01,     # Roczny koszt pożyczenia akcji do krótkiej sprzedaży
    'allocation_per_trade': 0.10,   # Kapitał alokowany na pojedynczą transakcję 
    'execution_mode': 'classic'     # Tryb egzekucji strategii
}

# PARAMETRY GENERATORA SYGNAŁÓW
SIGNAL_PARAMS = {
    'entry_threshold': 1.5,       # Z-Score, przy którym otwieramy pozycję
    'exit_threshold': 0.5,        # Z-Score, przy którym zamykamy z zyskiem
    'SL_threshold': 3.5,          # Z-Score, przy którym robimy Stop Loss
    'max_holding_period': None    # Maksymalny czas trzymania pozycji w świecach
}

# PARAMETRY KALKULATORA Z-SCORE
ZSCORE_PARAMS = {
    'multiplier': 3.0,     # Mnożnik dla half-life (określa bazową długość okna kroczącego)
    'min_window': 15,      # Minimalna dopuszczalna długość okna (w dniach/okresach)
    'max_window': 90       # Maksymalna dopuszczalna długość okna (w dniach/okresach)
}

# PARAMETRY DO POBIERANIA DANYCH
DATALOADER_PARAMS = {
    'default_interval': '1d'  # Domyślny interwał czasowy dla pobieranych danych (np. '1d', '1h', '15m')
}

# PARAMETY SKANERA RYNKU
SCANNER_PARAMS = {
    'max_pvalue': 0.05,       # Maksymalny poziom p-value dla testu ADF (standardowo 5%)
    'min_half_life': 1.0,     # Minimalny akceptowalny czas powrotu do średniej (w dniach/okresach)
    'max_half_life': 30.0     # Maksymalny czas powrotu do średniej (odrzucamy pary, które wracają zbyt wolno)
}