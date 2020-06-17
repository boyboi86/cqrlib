import numpy as np
import pandas as pd

from scipy.stats import percentileofscore

# =============================================================================
# Calculate rolling percentile scoring
# Only Series no Dataframe
# ============================================================================= 
def rolling_percentileofscore(series, window):

    def _percentile(arr):
        score = arr[-1]
        vals = arr[:-1]
        return percentileofscore(vals, score)

    _ = series.dropna()
    if _.empty:
        return pd.Series(np.nan, index=series.index)
    else:
        return _.rolling(window).apply(_percentile, raw=True).reindex(series.index) 