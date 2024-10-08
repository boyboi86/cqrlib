import numpy as np
import pandas as pd


def get_roll_measure(close_prices: pd.Series, window: int = 20):
    """
    Advances in Financial Machine Learning, page 282.

    Get Roll Measure

    Roll Measure gives the estimate of effective bid-ask spread
    without using quote-data.

    :param close_prices: (pd.Series) Close prices
    :param window: (int) Estimation window
    :return: (pd.Series) Roll measure
    """
    price_diff = close_prices.diff()
    price_diff_lag = price_diff.shift(1)
    return 2 * np.sqrt(abs(price_diff.rolling(window=window).cov(price_diff_lag)))



def get_roll_impact(close_prices: pd.Series, dollar_volume: pd.Series, window: int = 20):
    """
    Get Roll Impact.

    Derivate from Roll Measure which takes into account dollar volume traded.

    :param close_prices: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volume series
    :param window: (int) Estimation window
    :return: (pd.Series) Roll impact
    """
    roll_measure = get_roll_measure(close_prices, window)
    return roll_measure / dollar_volume



# Corwin-Schultz algorithm
def _get_beta(high: pd.Series, low: pd.Series, window: int):
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get beta estimate from Corwin-Schultz algorithm

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Beta estimates
    """
    ret = np.log(high / low)
    high_low_ret = ret ** 2
    beta = high_low_ret.rolling(window=2).sum()
    beta = beta.rolling(window=window).mean()
    return beta


def _get_gamma(high: pd.Series, low: pd.Series):
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get gamma estimate from Corwin-Schultz algorithm.

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :return: (pd.Series) Gamma estimates
    """
    high_max = high.rolling(window=2).max()
    low_min = low.rolling(window=2).min()
    gamma = np.log(high_max / low_min) ** 2
    return gamma


def _get_alpha(beta: pd.Series, gamma: pd.Series):
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get alpha from Corwin-Schultz algorithm.

    :param beta: (pd.Series) Beta estimates
    :param gamma: (pd.Series) Gamma estimates
    :return: (pd.Series) Alphas
    """
    den = 3 - 2 * 2 ** .5
    alpha = (2 ** .5 - 1) * (beta ** .5) / den
    alpha -= (gamma / den) ** .5
    alpha[alpha < 0] = 0  # Set negative alphas to 0 (see p.727 of paper)
    return alpha


def corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int = 20):
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get Corwin-Schultz spread estimator using high-low prices

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Corwin-Schultz spread estimators
    """
    # Note: S<0 iif alpha<0
    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)
    alpha = _get_alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    start_time = pd.Series(high.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, start_time], axis=1)
    spread.columns = ['Spread', 'Start_Time']  # 1st loc used to compute beta
    return spread.Spread



def bekker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20):
    """
    Advances in Financial Machine Learning, Snippet 19.2, page 286.

    Get Bekker-Parkinson volatility from gamma and beta in Corwin-Schultz algorithm.

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Bekker-Parkinson volatility estimates
    """
    # pylint: disable=invalid-name
    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)

    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2 ** .5
    sigma = (2 ** -0.5 - 1) * beta ** 0.5 / (k2 * den)
    sigma += (gamma / (k2 ** 2 * den)) ** 0.5
    sigma[sigma < 0] = 0
    return sigma

def get_bar_based_kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p. 286-288.

    Get Kyle lambda from bars data

    :param close: (pd.Series) Close prices
    :param volume: (pd.Series) Bar volume
    :param window: (int) Rolling window used for estimation
    :return: (pd.Series) Kyle lambdas
    """
    close_diff = close.diff()
    close_diff_sign = close_diff.apply(np.sign)
    close_diff_sign.replace(0, method='pad', inplace=True)  # Replace 0 values with previous
    volume_mult_trade_signs = volume * close_diff_sign  # bt * Vt
    return (close_diff / volume_mult_trade_signs).rolling(window=window).mean()



def get_bar_based_amihud_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p.288-289.

    Get Amihud lambda from bars data

    :param close: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volumes
    :param window: (int) rolling window used for estimation
    :return: (pd.Series) of Amihud lambda
    """
    returns_abs = np.log(close / close.shift(1)).abs()
    return (returns_abs / dollar_volume).rolling(window=window).mean()



def get_bar_based_hasbrouck_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p.289-290.

    Get Hasbrouck lambda from bars data

    :param close: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volumes
    :param window: (int) Rolling window used for estimation
    :return: (pd.Series) Hasbrouck lambda
    """
    log_ret = np.log(close / close.shift(1))
    log_ret_sign = log_ret.apply(np.sign).replace(0, method='pad')

    signed_dollar_volume_sqrt = log_ret_sign * np.sqrt(dollar_volume)
    return (log_ret / signed_dollar_volume_sqrt).rolling(window=window).mean()



def get_trades_based_kyle_lambda(price_diff: list, volume: list, aggressor_flags: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.286-288.

    Get Kyle lambda from trades data

    :param price_diff: (list) Price diffs
    :param volume: (list) Trades sizes
    :param aggressor_flags: (list) Trade directions [-1, 1]  (tick rule or aggressor side can be used to define)
    :return: (list) Kyle lambda for a bar and t-value
    """
    signed_volume = np.array(volume) * np.array(aggressor_flags)
    X = np.array(signed_volume).reshape(-1, 1)
    y = np.array(price_diff)
    coef, std = get_betas(X, y)
    t_value = coef[0] / std[0] if std[0] > 0 else np.array([0])
    return [coef[0], t_value[0]]


def get_trades_based_amihud_lambda(log_ret: list, dollar_volume: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.288-289.

    Get Amihud lambda from trades data

    :param log_ret: (list) Log returns
    :param dollar_volume: (list) Dollar volumes (price * size)
    :return: (float) Amihud lambda for a bar
    """
    X = np.array(dollar_volume).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    coef, std = get_betas(X, y)
    t_value = coef[0] / std[0] if std[0] > 0 else np.array([0])
    return [coef[0], t_value[0]]


def get_trades_based_hasbrouck_lambda(log_ret: list, dollar_volume: list, aggressor_flags: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.289-290.

    Get Hasbrouck lambda from trades data

    :param log_ret: (list) Log returns
    :param dollar_volume: (list) Dollar volumes (price * size)
    :param aggressor_flags: (list) Trade directions [-1, 1]  (tick rule or aggressor side can be used to define)
    :return: (list) Hasbrouck lambda for a bar and t value
    """
    X = (np.sqrt(np.array(dollar_volume)) * np.array(aggressor_flags)).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    coef, std = get_betas(X, y)
    t_value = coef[0] / std[0] if std[0] > 0 else np.array([0])
    return [coef[0], t_value[0]]