# Logika wejścia/wyjścia na bazie Z-Score

class SignalGenerator:
    def __init__(self, entry_treshold: float, exit_treshold: float, SL_treshold: float):
        self.entry_treshold = entry_treshold
        self.exit_treshold = exit_treshold
        self.SL_treshold = SL_treshold
        self._reset_state()

    def _reset_state(self):
        """Resets the internal state of the strategy to default values."""
        self.position = 'none'  # 'none', 'short', 'long'
        self.entry_date = None
        self.days_in_trade = 0
        self.entry_z = 0.0
    def _close_trade(self, close_date, exit_z: float, reason: str) -> dict:
        """Closes the current position, generates a trade summary, and resets the state."""
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
    def generate(self, zscore_data: list) -> list:
        """
        Main method generating signals based on a list of tuples (date, z_score).
        """
        trades = []
        self._reset_state()
        previous_z = None

        for current_date, current_z in zscore_data:
            if previous_z is None:
                previous_z = current_z
                continue

            # 1. Check exit conditions (Take Profit and Stop Loss)
            if self.position == 'short':
                if current_z < self.exit_threshold:      # Take Profit
                    trades.append(self._close_trade(current_date, current_z, 'TP'))
                elif current_z > self.stop_loss_threshold: # Stop Loss
                    trades.append(self._close_trade(current_date, current_z, 'SL'))

            elif self.position == 'long':
                if current_z > -self.exit_threshold:     # Take Profit
                    trades.append(self._close_trade(current_date, current_z, 'TP'))
                elif current_z < -self.stop_loss_threshold:# Stop Loss
                    trades.append(self._close_trade(current_date, current_z, 'SL'))

            # 2. Check entry conditions (only if no open position)
            if self.position == 'none':
                # Open SHORT
                if previous_z < self.entry_threshold and current_z > self.entry_threshold:
                    self.position = 'short'
                    self.entry_date = current_date
                    self.entry_z = current_z
                # Open LONG
                elif previous_z > -self.entry_threshold and current_z < -self.entry_threshold:
                    self.position = 'long'
                    self.entry_date = current_date
                    self.entry_z = current_z

            # 3. Update days in trade for open positions
            if self.position in ('short', 'long'):
                self.days_in_trade += 1

            # 4. Shift window
            previous_z = current_z

        # Forced close at the end of data (End Of File)
        if self.position != 'none':
            trades.append(self._close_trade(current_date, current_z, 'EOF'))

        return trades
