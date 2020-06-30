import numpy as np
import pandas as pd

from scipy.stats import norm

from research.Util.multiprocess import mp_pandas_obj

p = print
sep = ("=" * 55)

def ff_time(target_pos: pd.Series):
    """
    Advances in Financial Machine Learning, Snippet 14.1, page 197
    Derives the timestamps of flattening or flipping trades from a pandas series
    of target positions. Can be used for position changes analysis, such as
    frequency and balance of position changes.
    Flattenings - times when open position is bing closed (final target position is 0).
    Flips - times when positive position is reversed to negative and vice versa.
    :param target_positions: (pd.Series) Target position series with timestamps as indices
    :return: (pd.DatetimeIndex) Timestamps of trades flattening, flipping and last bet
    """

    _positions = target_pos[(target_pos == 0)].index  # Empty positions index
    previous_pos = target_pos.shift(1)  # Timestamps pointing at previous positions

    # Index of positions where previous one wasn't empty
    previous_pos = previous_pos[(previous_pos != 0)].index

    # FLATTENING - if previous position was open, but current is empty
    flat = _positions.intersection(previous_pos)

    # Multiplies current position with value of next one
    multi_pos = target_pos.iloc[1:] * target_pos.iloc[:-1].values

    # FLIPS - if current position has another direction compared to the next
    flips = multi_pos[(multi_pos < 0)].index
    ff = flat.union(flips).sort_values()
    if target_pos.index[-1] not in ff:  # Appending with last bet
        ff = ff.append(target_pos.index[-1:])

    return ff


def ave_hp(target_positions: pd.Series):
    """
    Advances in Financial Machine Learning, Snippet 14.2, page 197
    Estimates the average holding period (in days) of a strategy, given a pandas series
    of target positions using average entry time pairing algorithm.
    Idea of an algorithm:
    * entry_time = (previous_time * weight_of_previous_position + time_since_beginning_of_trade * increase_in_position )
      / weight_of_current_position
    * holding_period ['holding_time' = time a position was held, 'weight' = weight of position closed]
    * res = weighted average time a trade was held
    :param target_positions: (pd.Series) Target position series with timestamps as indices
    :return: (float) Estimated average holding period, NaN if zero or unpredicted
    """

    holding_period = pd.DataFrame(columns=['holding_time', 'weight'])
    entry_time = 0
    position_difference = target_positions.diff()

    # Time elapsed from the starting time for each position
    time_difference = (target_positions.index - target_positions.index[0]) / np.timedelta64(1, 'D')
    for i in range(1, target_positions.size):

        # Increased or unchanged position
        if float(position_difference.iloc[i] * target_positions.iloc[i - 1]) >= 0:
            if float(target_positions.iloc[i]) != 0:  # And not an empty position
                entry_time = (entry_time * target_positions.iloc[i - 1] +
                              time_difference[i] * position_difference.iloc[i]) / target_positions.iloc[i]

        # Decreased
        if float(position_difference.iloc[i] * target_positions.iloc[i - 1]) < 0:
            hold_time = time_difference[i] - entry_time

            # Flip of a position
            if float(target_positions.iloc[i] * target_positions.iloc[i - 1]) < 0:
                weight = abs(target_positions.iloc[i - 1])
                holding_period.loc[target_positions.index[i], ['holding_time', 'weight']] = (hold_time, weight)
                entry_time = time_difference[i]  # Reset entry time

            # Only a part of position is closed
            else:
                weight = abs(position_difference.iloc[i])
                holding_period.loc[target_positions.index[i], ['holding_time', 'weight']] = (hold_time, weight)

    if float(holding_period['weight'].sum()) > 0:  # If there were closed trades at all
        avg_holding_period = float((holding_period['holding_time'] * \
                                    holding_period['weight']).sum() / holding_period['weight'].sum())
    else:
        avg_holding_period = float('nan')

    return avg_holding_period

def _rtn_by_tw(mtx: pd.DataFrame, molecule):
    out = pd.DataFrame(1, index = molecule, columns = np.arange(mtx.shape[0])) #make zeros
    for idx, (t0, t1, rtn) in enumerate(mtx.itertuples()):
        out.loc[t0:t1, idx] = rtn
    return out

def rtn_by_tw(mtx: pd.DataFrame, num_threads: int = 1):
    df0 = pd.DataFrame(index=mtx.index).assign(t1 = mtx.t1)
    mtx['equity_diff'] = mtx['close'].sub(mtx['open'])
    mtx['bond_diff'] = mtx['clean_close'].sub(mtx['clean_open'])
    for idx in mtx.index:
        _close_pos = mtx.loc[idx, 'close_pos']
        _open_pos = mtx.loc[idx, 'open_pos']
        if ((_open_pos) > .0 and (_close_pos) >= .0) or ((_open_pos) < .0 and (_close_pos) >= .0):
            mtx.loc[idx, 't1_pos'] = max(0, float(_close_pos - _open_pos))
            mtx.loc[idx, 't1_vol'] = float(_open_pos - _close_pos)
        if ((_open_pos) > .0 and (_close_pos) <= .0) or ((_open_pos) > .0 and (_close_pos) <= .0):
            mtx.loc[idx, 't1_pos'] = max(0, float(_close_pos - _open_pos)) #shorting doesn't increase your aum
            mtx.loc[idx, 't1_vol'] = float(_open_pos - _close_pos)
            mtx[idx, 'yield'].mul(-1) #if you short; you will pay for the div/ interest instead
        
        #but it does add to your profit.. or losses
        if mtx.loc[idx, 'label'] == 0:
            mtx.loc[idx, 'pnl'] = (mtx.loc[idx, 'bond_diff'] + mtx.loc[idx, 'yield']) * mtx.loc[idx, 't1_vol']
        else:
            mtx.loc[idx, 'pnl'] = (mtx.loc[idx, 'equity_diff'] + mtx.loc[idx, 'yield']) * mtx.loc[idx, 't1_vol']
            
    
    t0_val = mtx['close'].mul(mtx['open_pos'])
    t1_val = mtx['close'].mul(mtx['t1_pos'])
    
    df0['rtn'] = mtx['pnl'].sub(mtx['expense']).div(t0_val.add(t1_val))
    df0['rtn'] = df0['rtn'].add(1) #prepare matrix to be link
    
    _idx = df0.index.union(df0.t1)
    _idx = _idx.sort_values()
    out = mp_pandas_obj(func=_rtn_by_tw,
                        pd_obj = ('molecule', _idx), #we need the index so that during partition they don't go haywire
                        num_threads=num_threads, 
                        mp_batches=1, 
                        lin_mols=True, 
                        axis=0,
                        mtx = df0)
    
    _product = out.prod(axis = 1)
    out[out == 1] = .0
    out[out > 0] = 1.
    _sum = out.sum(axis = 1)
    _tw_rtn = (_product ** (1/_sum)) - 1
    
    return _tw_rtn

def _hh_idx(rtn: pd.Series) -> float:
    """
    Advances in Financial Machine Learning, Snippet 14.3, page 201
    Derives the concentration of returns from given pd.Series of returns.
    Algorithm is based on Herfindahl-Hirschman Index where return weights
    are taken as an input.
    :param returns: (pd.Series) Returns from bets
    :return: (float) Concentration of returns (nan if less than 3 returns)
    """

    if rtn.size <= 2:
        return float('nan')  # If less than 3 bets
    weights = rtn / rtn.sum()  # Weights of each bet
    hhi = (weights ** 2).sum()  # Herfindahl-Hirschman Index for weights
    hhi = (hhi - rtn.shape[0] ** (-1)) / (1. - rtn.shape[0] ** (-1))

    return hhi.squeeze()


def hh_idx(rtn: pd.Series, freq: str = 'M'):
    """
    Advances in Financial Machine Learning, Snippet 14.3, page 201
    Given a pd.Series of returns, derives concentration of positive returns, negative returns
    and concentration of bets grouped by time intervals (daily, monthly etc.).
    If after time grouping less than 3 observations, returns nan.
    Properties or results:
    * low positive_concentration ⇒ no right fat-tail of returns (desirable)
    * low negative_concentration ⇒ no left fat-tail of returns (desirable)
    * low time_concentration ⇒ bets are not concentrated in time, or are evenly concentrated (desirable)
    * positive_concentration == 0 ⇔ returns are uniform
    * positive_concentration == 1 ⇔ only one non-zero return exists
    :param returns: (pd.Series) Returns from bets
    :param frequency: (str) Desired time grouping frequency from pd.Grouper
    :return: (tuple of floats) Concentration of positive, negative and time grouped concentrations
    """

    # Concentration of positive returns per bet
    positive_hhi = _hh_idx(rtn = rtn[rtn >= 0])

    # Concentration of negative returns per bet
    negative_hhi = _hh_idx(rtn = rtn[rtn < 0])

    # Concentration of bets/time period (month by default)
    time_hhi = _hh_idx(rtn = rtn.groupby(pd.Grouper(freq = freq)).count())
    
    p("\nHerfindahl-Hirschman Index ({0} frequency)\n{1}".format(freq,sep))
    p("HHI positive: {0:.6f}\nHHI Negative: {1:.6f}\nHHI Time between: {2:.6f}\n".format(positive_hhi,
                                                                                         negative_hhi,
                                                                                         time_hhi))

    return (positive_hhi, negative_hhi, time_hhi)


def drawdn_period(data: pd.Series, dollars: bool = False, q_param: float = .0):
    """
    Advances in Financial Machine Learning, Snippet 14.4, page 201
    Calculates drawdowns and time under water for pd.Series of either relative price of a
    portfolio or dollar price of a portfolio.
    Intuitively, a drawdown is the maximum loss suffered by an investment between two consecutive high-watermarks.
    The time under water is the time elapsed between an high watermark and the moment the PnL (profit and loss)
    exceeds the previous maximum PnL. We also append the Time under water series with period from the last
    high-watermark to the last return observed.
    Return details:
    * Drawdown series index is the time of a high watermark and the value of a
      drawdown after it.
    * Time under water index is the time of a high watermark and how much time
      passed till the next high watermark in years. Also includes time between
      the last high watermark and last observation in returns as the last element.
    :param returns: (pd.Series) Returns from bets
    :param dollars: (bool) Flag if given dollar performance and not returns.
                    If dollars, then drawdowns are in dollars, else as a %.
    :return: (tuple of pd.Series) Series of drawdowns and time under water
    """
    if isinstance(data, pd.Series):
        df = data.to_frame('pnl')
    else:
        df = pd.DataFrame(index= data.index).assign(pnl = data)
        
    df['hwm'] = data.expanding().max()  # Adding high watermarks as column

    # Grouped as min returns by high watermarks
    high_watermarks = df.groupby('hwm').min().reset_index()
    high_watermarks.columns = ['hwm', 'min']

    # Time high watermark occurred
    high_watermarks.index = df['hwm'].drop_duplicates(keep='first').index

    # Picking ones that had a drawdown after high watermark
    high_watermarks = high_watermarks[high_watermarks['hwm'] > high_watermarks['min']]
    if dollars:
        draw_dn = high_watermarks['hwm'] - high_watermarks['min']
    else:
        draw_dn = 1 - high_watermarks['min'] / high_watermarks['hwm']

    time_under_water = ((high_watermarks.index[1:] - high_watermarks.index[:-1]) / np.timedelta64(1, 'Y')).values

    # Adding also period from last High watermark to last return observed.
    time_under_water = np.append(time_under_water,
                                 (data.index[-1] - high_watermarks.index[-1]) / np.timedelta64(1, 'Y'))

    time_under_water = pd.Series(time_under_water, index=high_watermarks.index)
    
    if q_param != .0 and dollars is False:
        p("\nLoss Metric ({0} quartile)\n{1}\nDrawdown: {2:.6f}%\nTime under water (Based on 252 days): {3:.6f}\n".format(q_param, sep,
                                                                                                                                draw_dn.quantile(q_param) * 100,
                                                                                                                                time_under_water.quantile(q_param) * 252))
    if q_param != .0 and dollars is not False:
        p("\nLoss Metric ({0} quartile)\n{1}\nDrawdown: ${2:.2f}\nTime under water (Based on 252 days): {3:.6f}\n".format(q_param, sep,
                                                                                                                                    draw_dn.quantile(q_param),
                                                                                                                                    time_under_water.quantile(q_param) * 252))
    
    return draw_dn, time_under_water

def _rtn_dist(rtn: pd.Series):
    postive_rtn = rtn[rtn > 0]
    negative_rtn = rtn[rtn <= 0]
    
    p("\nAccuracy Metric Estimate\n{0}".format(sep))
    p("Total postive returns: {0}\nTotal negative returns: {1}\n".format(postive_rtn.count(),
                                                                             negative_rtn.count()))
    p("Average postive returns: {0:.6f}\nAverage negative returns: {1:.6f}\n".format(postive_rtn.mean(), 
                                                                                          negative_rtn.mean()))

def annualized_rtn(data: pd.Series, t_days: int = 252, verbose: bool = False):
    obs_days = (data.index[-1] - data.index[0]) / np.timedelta64(1, 'D')
    cum_rtn = data[-1]/data[0]
    a_rtn = (cum_rtn)**(t_days/obs_days) - 1
    if verbose:
        _rtn_dist(rtn = data.pct_change().fillna(0))
    return a_rtn
    #Using 365 days instead of 252 as days observed are calculated as calendar
    #days between the first observation and the last
    
    
def sharpe_ratio(rtn: pd.Series, rf_param: float = .0, t_days: int = 252):
    """
    Calculates annualized Sharpe ratio for pd.Series of normal or log returns.
    Risk_free_rate should be given for the same period the returns are given.
    For example, if the input returns are observed in 3 months, the risk-free
    rate given should be the 3-month risk-free rate.
    :param returns: (pd.Series) Returns - normal or log
    :param entries_per_year: (int) Times returns are recorded per year (252 by default)
    :param risk_free_rate: (float) Risk-free rate (0 by default)
    :return: (float) Annualized Sharpe ratio
    """

    _sr = (rtn.mean() - rf_param) / rtn.std() * (t_days) ** 0.5

    return _sr


def inform_ratio(rtn: pd.Series, benchmark: float = .0, t_days: int = 252):
    """
    Calculates annualized information ratio for pd.Series of normal or log returns.
    Benchmark should be provided as a return for the same time period as that between
    input returns. For example, for the daily observations it should be the
    benchmark of daily returns.
    It is the annualized ratio between the average excess return and the tracking error.
    The excess return is measured as the portfolio’s return in excess of the benchmark’s
    return. The tracking error is estimated as the standard deviation of the excess returns.
    :param returns: (pd.Series) Returns - normal or log
    :param benchmark: (float) Benchmark for performance comparison (0 by default)
    :param entries_per_year: (int) Times returns are recorded per year (252 by default)
    :return: (float) Annualized information ratio
    """

    excess_rtn = rtn - benchmark
    _ir = sharpe_ratio(excess_rtn, t_days = t_days)

    return _ir


def proba_sr(obs_sr: float,
             benchmark_sr: float,
             num_returns: int,
             skew_returns: float = .0,
             kurt_returns: float = 3.):
    
    nom = (obs_sr - benchmark_sr) * (num_returns - 1) ** 0.5
    denom = (1 - skew_returns * obs_sr + (kurt_returns - 1) / 4 * obs_sr ** 2) ** 0.5
    p_sr = norm.cdf(nom / denom) 

    return p_sr


def deflated_sr(obs_sr: float,
                sr_est: list,
                num_returns: int,
                skew_returns: float = 0, 
                kurt_returns: float = 3,
                est_param: bool = False):

    # Calculating benchmark_SR from the parameters of estimates
    if est_param:
        benchmark_sr = sr_est[0] * \
                   ((1 - np.euler_gamma) * norm.ppf(1 - 1 / sr_est[1]) +
                    np.euler_gamma * norm.ppf(1 - 1 / sr_est[1] * np.e ** (-1)))

    # Calculating benchmark_SR from a list of estimates
    else:
        benchmark_sr = np.array(sr_est).std() * \
                       ((1 - np.euler_gamma) * norm.ppf(1 - 1 / len(sr_est)) +
                        np.euler_gamma * norm.ppf(1 - 1 / len(sr_est) * np.e ** (-1)))

    deflated_sr = proba_sr(obs_sr = obs_sr,
                           benchmark_sr = benchmark_sr,
                           num_returns = num_returns,
                           skew_returns = skew_returns, 
                           kurt_returns = kurt_returns)

    return (deflated_sr, benchmark_sr)


def min_tr_len(obs_sr: float, 
                benchmark_sr: float,
                skew_returns: float = 0,
                kurt_returns: float = 3,
                alpha: float = 0.05):
    """
    min track record length
    default to norma dist
    """

    tr_len = 1 + (1 - skew_returns * obs_sr + (kurt_returns - 1) / 4 * obs_sr ** 2) * (norm.ppf(1 - alpha) / (obs_sr - benchmark_sr)) ** (2)

    return tr_len
