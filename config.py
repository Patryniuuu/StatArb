# =====================================================================
# 1. GŁÓWNE FLAGI STERUJĄCE EKSPERYMENTEM (ZMIENIAJ TYLKO TUTAJ!)
# =====================================================================
ACTIVE_INTERVAL = '1d'      # Wybierz: '1d' (dzienny wieloletni) lub '1h' (godzinowy intraday)
EXECUTION_MODE = 'both'     # Wybierz: 'pairs', 'basket' lub 'both'
TOP_K_PAIRS = 1             # Liczba par w portfelu Pairs Trading
TOP_K_BASKETS = 1           # Liczba koszyków w portfelu Johansen Basket

# Wybór aktywnych sektorów do symulacji
ACTIVE_SECTORS = ["TECH_SEMI", "BANKS_FINANCE", "ENERGY_OIL", "CONSUMER_STAPLES"]

# =====================================================================
# 2. AUTOMATYCZNE PROFILE INTERWAŁÓW (1d vs 1h)
# =====================================================================
INTERVAL_PROFILES = {
    '1d': {
        'train_months': 12,
        'test_months': 3,
        'oos_start': '2020-01-01',
        'oos_end': '2025-12-31',
        'zscore_params': {
            'multiplier': 3.0,
            'min_window': 15,     # 15 dni sesyjnych
            'max_window': 90      # 90 dni sesyjnych
        },
        'scanner_params': {
            'max_pvalue': 0.05,
            'min_half_life': 1.0,
            'max_half_life': 30.0,  # Do 30 dni sesyjnych
            'min_beta': 0.35,
            'max_beta': 2.50,
            'min_r_squared': 0.60,
            'min_return_corr': 0.45
        },
        'basket_scanner_params': {
            'basket_size': 3,
            'significance_level': 1,
            'min_half_life': 1.0,
            'max_half_life': 30.0,
            'min_pair_corr': 0.40
        },
        'pairs_signal_params': {
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'SL_threshold': 3.5,
            'max_holding_period': None
        },
        'basket_signal_params': {
            'entry_threshold': 1.5,
            'exit_threshold': 0.5,
            'SL_threshold': 3.5,
            'max_holding_period': None
        }
    },
    '1h': {
        'train_months': 3,        # 3 msc na 1h to ~440 świec godzinowych
        'test_months': 1,         # 1 msc handlu OOS (~147 świec godzinowych)
        'oos_start': '2025-05-01',
        'oos_end': '2026-09-30',
        'zscore_params': {
            'multiplier': 2.5,
            'min_window': 21,     # Min. 3 dni sesyjne (3 * 7h = 21 świec)
            'max_window': 105     # Max. 15 dni sesyjnych (15 * 7h = 105 świec)
        },
        'scanner_params': {
            'max_pvalue': 0.05,
            'min_half_life': 3.0,   # Min. ~pół dnia sesyjnego (3 świece 1h)
            'max_half_life': 84.0,  # Max. 12 dni sesyjnych (12 * 7h = 84 świece 1h)
            'min_beta': 0.35,
            'max_beta': 2.50,
            'min_r_squared': 0.60,
            'min_return_corr': 0.40
        },
        'basket_scanner_params': {
            'basket_size': 3,
            'significance_level': 1,
            'min_half_life': 3.0,
            'max_half_life': 84.0,  # Na 1h koszyki potrzebują limitu w godzinach, nie dniach!
            'min_pair_corr': 0.35
        },
        'pairs_signal_params': {
            'entry_threshold': 1.75,
            'exit_threshold': 0.5,
            'SL_threshold': 3.5,
            'max_holding_period': None
        },
        'basket_signal_params': {
            'entry_threshold': 1.5,
            'exit_threshold': 0.5,
            'SL_threshold': 3.5,
            'max_holding_period': None
        }
    }
}

# Automatyczne wypakowanie aktywnego profilu do zmiennych używanych przez resztę systemu
CURRENT_PROFILE = INTERVAL_PROFILES[ACTIVE_INTERVAL]

TRAIN_MONTHS = CURRENT_PROFILE['train_months']
TEST_MONTHS = CURRENT_PROFILE['test_months']
OOS_START = CURRENT_PROFILE['oos_start']
OOS_END = CURRENT_PROFILE['oos_end']

ZSCORE_PARAMS = CURRENT_PROFILE['zscore_params']
SCANNER_PARAMS = CURRENT_PROFILE['scanner_params']
BASKET_SCANNER_PARAMS = CURRENT_PROFILE['basket_scanner_params']
PAIRS_SIGNAL_PARAMS = CURRENT_PROFILE['pairs_signal_params']
BASKET_SIGNAL_PARAMS = CURRENT_PROFILE['basket_signal_params']

DATALOADER_PARAMS = {
    'default_interval': ACTIVE_INTERVAL
}

# PARAMETRY SILNIKA DO BACKTESTÓW
ENGINE_PARAMS = {
    'initial_capital': 100_000.0,
    'commission_bps': 3.0,
    'slippage_bps': 3.0,
    'borrow_rate_annual': 0.01,
    'allocation_per_trade': 0.10,
    'execution_mode': 'classic'
}

# UNIWERSUM SPÓŁEK PODZIELONE NA JEDNORODNE SEKTORY (36 płynnych spółek z USA)
SECTORS = {
    "TECH_SEMI": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "QCOM", "TXN"
    ],
    "BANKS_FINANCE": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "STT"
    ],
    "ENERGY_OIL": [
        "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY"
    ],
    "CONSUMER_STAPLES": [
        "KO", "PEP", "PG", "CL", "KMB", "WMT", "COST", "TGT", "MDLZ"
    ]
}