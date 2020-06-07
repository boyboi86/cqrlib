import warnings
import pandas as pd
import numpy as np
from scipy.stats import norm

from research.Util.multiprocess import mp_pandas_obj

def _avg_active_signals(signals, molecule):
    """
    Part of SNIPPET 10.2
    A function to be passed to the 'mp_pandas_obj' function to allow the bet sizes to be averaged using multiprocessing.
    At time loc, average signal among those still active.
    Signal is active if (a) it is issued before or at loc, and (b) loc is before the signal's end time,
    or end time is still unknown (NaT).
    :param signals: (pandas.DataFrame) Contains at least the following columns: 'signal' (the bet size) and 't1' (the closing time of the bet).
    :param molecule: (list) Indivisible tasks to be passed to 'mp_pandas_obj', in this case a list of datetimes.
    :return: (pandas.Series) The averaged bet size sub-series.
    """
    out = pd.Series()
    for loc in molecule:
        signals_times = (signals.index.values <= loc)&((loc < signals['t1'])|pd.isnull(signals['t1']))
        active = signals[signals_times].index
        if len(active) > 0:
            # Average active signals if they exist.
            out[loc] = signals.loc[active, 'signal'].mean()
        else:
            # Return zero if no signals are active at this time step.
            out[loc] = 0

    return out


def avg_active_signals(signals: pd.DataFrame, num_threads: int = 1):
    """
    SNIPPET 10.2 - BETS ARE AVERAGED AS LONG AS THEY ARE STILL ACTIVE
    Function averages the bet sizes of all concurrently active bets. This function makes use of multiprocessing.
    :param signals: (pandas.DataFrame) Contains at least the following columns:
     'signal' - the bet size
     't1' - the closing time of the bet
     And the index must be datetime format.
    :param num_threads: (int) Number of threads to use in multiprocessing, default value is 1.
    :return: (pandas.Series) The averaged bet sizes.
    """
    # 1) Time points where signals change (either one start or one ends).
    t_pnts = set(signals['t1'].dropna().values)
    t_pnts = t_pnts.union(signals.index.values)
    t_pnts = list(t_pnts)
    t_pnts.sort()
    out = mp_pandas_obj(_avg_active_signals, ('molecule', t_pnts), num_threads, signals=signals)
    return out

def discrete_signal(signal0, step_size):
    """
    SNIPPET 10.3 - SIZE DISCRETIZATION TO PREVENT OVERTRADING
    Discretizes the bet size signal based on the step size given.
    :param signal0: (pandas.Series) The signal to discretize.
    :param step_size: (float) Step size.
    :return: (pandas.Series) The discretized signal.
    """
    signal1 = (signal0 / step_size).round() * step_size
    signal1[signal1 > 1] = 1  # Cap
    signal1[signal1 < -1] = -1  # Floor
    return signal1

def get_signal(events: pd.DataFrame,
               step_size: int = 1,
               prob: pd.Series = None,
               pred: pd.Series=None,
               n_classes: int = 1.,
               discretization: bool = False,
               num_threads: int = 1, **kargs):
    """
    SNIPPET 10.1 - FROM PROBABILITIES TO BET SIZE
    Calculates the given size of the bet given the side and the probability (i.e. confidence) of the prediction. In this
    representation, the probability will always be between 1/num_classes and 1.0.
    :param prob: (pd.Series) The probability of the predicted bet side.
    :param num_classes: (int) The number of predicted bet sides.
    :param pred: (pd.Series) The predicted bet side. Default value is None which will return a relative bet size
     (i.e. without multiplying by the side).
    :return: (pd.Series) The bet size.
    """
    # Get signals from predictions.
    if prob.shape[0] == 0:
        return pd.Series()

    # 1) Generate signals from multinomial classification (one-vs-rest), looking into sklearn OvR
    bet_sizes = (prob - 1/n_classes) / (prob * (1 - prob))**0.5

    # Allow for bet size to be returned with or without side.
    if pred is not None:
        # signal = side * size
        bet_sizes = pred * (2 * norm.cdf(bet_sizes) - 1)
    else:
        # signal = size only
        bet_sizes = bet_sizes.apply(lambda _sig: 2 * norm.cdf(_sig) - 1)
    # for quantamental models chapter 3
    if 'side' in events:
        bet_sizes *= events.loc[bet_sizes.index, 'side']
    signals_df = bet_sizes.to_frame('signal').join(events['t1'], how = 'left')
    signals_df = avg_active_signals(signals = signals_df, num_threads = num_threads)
    if discretization:
        signals_df = discrete_signal(signal0 = signals_df, step_size = step_size)
    return signals_df
# ==============================================================================
# SNIPPET 10.4 - DYNAMIC POSITION SIZE AND LIMIT PRICE
# The below functions are part of or derived from the functions
# in snippet 10.4.
# ==============================================================================
# Bet size calculations based on a sigmoid function.
def _bet_size(width_coef, price_dvg):
    """
    Part of SNIPPET 10.4
    Calculates the bet size from the price divergence and a regulating coefficient.
    Based on a sigmoid function for a bet size algorithm.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param price_div: (float) Price divergence, forecast price - market price.
    :return: (float) The bet size.
    """
    return price_dvg * ((width_coef + price_dvg ** 2) ** (-0.5))


def _target_position(width_coef, forecast_price, market_price, max_pos_size):
    """
    Part of SNIPPET 10.4
    Calculates the target position given the forecast price, market price, maximum position size, and a regulating
    coefficient. Based on a sigmoid function for a bet size algorithm.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param forecast_price: (float) Forecast price.
    :param market_price: (float) Market price.
    :param max_pos: (int) Maximum absolute position size.
    :return: (int) Target position.
    """
    return int(_bet_size(width_coef, forecast_price - market_price) * max_pos_size)


def _inv_price(forecast_price, width_coef, m_bet_size):
    """
    Part of SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the market price.
    Based on a sigmoid function for a bet size algorithm.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to market price.
    """
    return forecast_price - m_bet_size * (width_coef / (1 - m_bet_size**2))**0.5


def _limit_price(target_pos, pos, forecast_price, width_coef, max_pos_size):
    """
    Part of SNIPPET 10.4
    Calculates the limit price.
    Based on a sigmoid function for a bet size algorithm.
    :param target_pos: (int) Target position.
    :param pos: (int) Current position.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (int) Maximum absolute position size.
    :return: (float) Limit price.
    """
    if target_pos == pos:
        # Return NaN if the current and target positions are the same to avoid divide-by-zero error.
        return np.nan

    sgn = np.sign(target_pos-pos)
    limit_p = 0
    for j in np.arange(abs(pos + sgn), abs(target_pos + 1)):
        limit_p += _inv_price(forecast_price = forecast_price,
                              width_coef = width_coef,
                              m_bet_size = j/float(max_pos_size))

    limit_p = limit_p / abs(target_pos - pos)
    return limit_p


def _width_coef(price_dvg, m_bet_size):
    """
    Part of SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.
    Based on a sigmoid function for a bet size algorithm.
    :param price_div: (float) Price divergence, forecast price - market price.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to the
        regulating coefficient.
    """
    return (price_dvg**2) * ((m_bet_size**(-2)) - 1)


def dynamic_bet(pos: int = 0,
                max_pos_size: float = 100.,
                market_price: float = 100.,
                forecast_price: float = 115.,
                _w_params: dict = {'divergence': 10, 'm_bet_size': .95}):

    _wc = _width_coef(price_dvg = _w_params['divergence'], 
                     m_bet_size = _w_params['m_bet_size'])
    
    _trgt_pos = _target_position(width_coef = _wc,
                                 forecast_price = forecast_price,
                                 market_price = market_price,
                                 max_pos_size = max_pos_size)
    
    _lp = _limit_price(target_pos = _trgt_pos,
                       pos = pos,
                       forecast_price = forecast_price,
                       width_coef = _wc,
                       max_pos_size = max_pos_size)
    
    return _wc, _trgt_pos, _lp

# ==============================================================================
# Bet size calculations based on a power function.
def bet_size_power(width_coef, price_dvg):
    """
    Derived from SNIPPET 10.4
    Calculates the bet size from the price divergence and a regulating coefficient.
    Based on a power function for a bet size algorithm.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param price_div: (float) Price divergence, f - market_price, must be between -1 and 1, inclusive.
    :return: (float) The bet size.
    """
    if not (-1 <= price_dvg <= 1):
        raise ValueError(f"Price divergence must be between -1 and 1, inclusive. Found price divergence value:"
                         f" {price_dvg}")
    if price_dvg == 0.0:
        return 0.0

    return np.sign(price_dvg) * abs(price_dvg)**width_coef


def get_target_pos_power(w_param, forecast_price, market_price, max_pos):
    """
    Derived from SNIPPET 10.4
    Calculates the target position given the forecast price, market price, maximum position size, and a regulating
    coefficient. Based on a power function for a bet size algorithm.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param forecast_price: (float) Forecast price.
    :param market_price: (float) Market price.
    :param max_pos: (float) Maximum absolute position size.
    :return: (float) Target position.
    """
    return int(bet_size_power(w_param, forecast_price-market_price) * max_pos)


def inv_price_power(forecast_price, w_param, m_bet_size):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the market price.
    Based on a power function for a bet size algorithm.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to market price.
    """
    if m_bet_size == 0.0:
        return forecast_price
    return forecast_price - np.sign(m_bet_size) * abs(m_bet_size)**(1/w_param)


def limit_price_power(target_pos, pos, forecast_price, w_param, max_pos):
    """
    Derived from SNIPPET 10.4
    Calculates the limit price. Based on a power function for a bet size algorithm.
    :param target_pos: (float) Target position.
    :param pos: (float) Current position.
    :param forecast_price: (float) Forecast price.
    :param w_param: (float) Coefficient regulating the width of the bet size function.
    :param max_pos: (float) Maximum absolute position size.
    :return: (float) Limit price.
    """
    sgn = np.sign(target_pos-pos)
    l_p = 0
    for j in range(abs(pos+sgn), abs(target_pos+1)):
        l_p += inv_price_power(forecast_price, w_param, j/float(max_pos))

    l_p = l_p / abs(target_pos-pos)
    return l_p


def get_w_power(price_div, m_bet_size):
    """
    Derived from SNIPPET 10.4
    Calculates the inverse of the bet size with respect to the regulating coefficient 'w'.
    The 'w' coefficient must be greater than or equal to zero.
    Based on a power function for a bet size algorithm.
    :param price_div: (float) Price divergence, forecast price - market price.
    :param m_bet_size: (float) Bet size.
    :return: (float) Inverse of bet size with respect to the regulating coefficient.
    """
    if not -1 <= price_div <= 1:
        raise ValueError("Price divergence argument 'x' must be between -1 and 1,"
                         " inclusive when using function 'power'.")

    w_calc = np.log(m_bet_size/np.sign(price_div)) / np.log(abs(price_div))
    if w_calc < 0:
        warnings.warn("'w' parameter evaluates to less than zero. Zero is returned.", UserWarning)

    return max(0, w_calc)