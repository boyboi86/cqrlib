try:
    from research.Features.fractional_diff import fracDiff_FFD, fracDiff, min_value, plot_min_ffd
    #from research.Features.futures_roll import *
    #from research.Features.PCA_weight_dist import *
except:
    from Features.fractional_diff import fracDiff_FFD, fracDiff, min_value, plot_min_ffd
    #from Features.futures_roll import *
    #from Features.PCA_weight_dist import *