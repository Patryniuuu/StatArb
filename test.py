from data.data_loader import DataLoader
from research.market_scanner import BasketScanner

# 1. Pobranie danych
loader = DataLoader(default_interval="1d")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
df_prices = loader.get_prices(tickers, "2020-01-01", "2021-12-31")

# 2. Inicjalizacja skanera (szukamy trójek - portfeli 3-spółkowych)

scanner = BasketScanner(basket_size=3, significance_level=0)
df_ranking = scanner.scan_sector(df_prices)
print(df_ranking)