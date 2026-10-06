from typing import List, Tuple, Optional, Dict, Any


class SignalGenerator:
    """
    Deterministyczna maszyna stanów generująca sygnały transakcyjne
    dla strategii StatArb na podstawie szeregu Z-Score.
    """
    def __init__(
        self, 
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.5, 
        SL_threshold: float = 3.5,
        max_holding_period: Optional[int] = None  # Np. int(3 * half_life)
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.SL_threshold = SL_threshold
        self.max_holding_period = max_holding_period
        self._reset_state()

    def _reset_state(self) -> None:
        """Resetuje wewnętrzny stan maszyny do wartości domyślnych."""
        self.position = 'none'  # 'none', 'short', 'long'
        self.entry_date = None
        self.days_in_trade = 0
        self.entry_z = 0.0

    def _close_trade(self, close_date: Any, exit_z: float, reason: str) -> Dict[str, Any]:
        """Tworzy rekord zamkniętej transakcji i czyści stan pozycji."""
        trade = {
            'entry_date': self.entry_date,
            'close_date': close_date,
            'position': self.position,
            'days_in_trade': self.days_in_trade,
            'entry_z': self.entry_z,
            'exit_z': exit_z,
            'reason': reason
        }
        self._reset_state()
        return trade

    def generate(self, zscore_data: List[Tuple[Any, float]]) -> List[Dict[str, Any]]:
        """
        Przetwarza szereg (Data, Z-score) i generuje listę wykonanych transakcji.
        """
        trades: List[Dict[str, Any]] = []
        self._reset_state()
        previous_z: Optional[float] = None

        for current_date, current_z in zscore_data:
            if previous_z is None:
                previous_z = current_z
                continue

            # Flaga zabezpieczająca przed otwarciem nowej pozycji na tej samej świecy,
            # na której właśnie zrealizowano wyjście (ochrona przed same-bar re-entry)
            just_closed = False

            
            if self.position == 'short':
                self.days_in_trade += 1
                
                # Take Profit: spadek poniżej progu wyjścia
                if current_z < self.exit_threshold:
                    trades.append(self._close_trade(current_date, current_z, 'TP'))
                    just_closed = True
                # Stop Loss: wybicie w górę powyżej progu obronnego
                elif current_z > self.SL_threshold:
                    trades.append(self._close_trade(current_date, current_z, 'SL'))
                    just_closed = True
                # Time Stop: pozycja trwa zbyt długo względem dynamiki powrotu do średniej
                elif self.max_holding_period is not None and self.days_in_trade >= self.max_holding_period:
                    trades.append(self._close_trade(current_date, current_z, 'TIME'))
                    just_closed = True

            elif self.position == 'long':
                self.days_in_trade += 1
                
                # Take Profit: powrót powyżej ujemnego progu wyjścia
                if current_z > -self.exit_threshold:
                    trades.append(self._close_trade(current_date, current_z, 'TP'))
                    just_closed = True
                # Stop Loss: dalszy spadek poniżej ujemnego progu obronnego
                elif current_z < -self.SL_threshold:
                    trades.append(self._close_trade(current_date, current_z, 'SL'))
                    just_closed = True
                # Time Stop
                elif self.max_holding_period is not None and self.days_in_trade >= self.max_holding_period:
                    trades.append(self._close_trade(current_date, current_z, 'TIME'))
                    just_closed = True

            # 2. WEJŚCIE W POZYCJĘ (Tylko jeśli portfel jest pusty)
            if self.position == 'none' and not just_closed:
                # Otwórz SHORT: przebicie górnego progu z dołu do góry
                if previous_z < self.entry_threshold and current_z >= self.entry_threshold:
                    self.position = 'short'
                    self.entry_date = current_date
                    self.entry_z = current_z
                    self.days_in_trade = 0

                # Otwórz LONG: przebicie dolnego progu z góry do dołu
                elif previous_z > -self.entry_threshold and current_z <= -self.entry_threshold:
                    self.position = 'long'
                    self.entry_date = current_date
                    self.entry_z = current_z
                    self.days_in_trade = 0

            # Przesunięcie okna o jeden krok w przód
            previous_z = current_z

        # 3. ZAMKNIĘCIE AWARYJNE (Koniec danych testowych)
        if self.position != 'none' and zscore_data:
            last_date, last_z = zscore_data[-1]
            trades.append(self._close_trade(last_date, last_z, 'EOF'))

        return trades