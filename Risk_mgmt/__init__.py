try:
    from research.Risk_mgmt.bet_size import (avg_active_signals, get_signal, dynamic_bet, co_bets_size)
    from research.Risk_mgmt.ef3m import mts_fit
except:
    from Risk_mgmt.bet_size import (avg_active_signals, get_signal, dynamic_bet, co_bets_size)
    from Risk_mgmt.ef3m import mts_fit