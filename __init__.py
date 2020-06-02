# make sure there is no mp func otherwise import individual func instead

from research.Util.multiprocess import mp_pandas_obj, opt_num_threads, process_jobs_, process_jobs
from research.Util.volatility import *
from research.Util.indicator import bband_as_side

from research.Util.bsm_model import *
from research.Util.opt_prob_calculator import *

#from research.Tools.root_methods import *
from research.Tools.stats_rpt import (normality, unit_root, report_matrix, white_random, feat_imp)
from research.Tools.cross_validate import (train_times, embargo_times, PurgedKFold, cv_score)
from research.Tools.metrics import (mdi, mda, sfi, mp_sfi, sample_weight_generator, plot_feat_imp)
from research.Tools.feat_PCA import (o_feat, feat_pca)

from research.Labels.triple_barrier_method import _pt_sl_t1, vert_barrier, tri_barrier, meta_label, drop_label
from research.Labels.percentile_score import *

from research.Filters.filters import cs_filter

from research.Sampling.sample_unique import (wght_by_coevents, num_co_events, av_unique,
                                             idx_matrix, mp_idx_matrix, 
                                             seq_bts, mp_seq_bts, 
                                             MC_seq_bts, MT_MC, 
                                             wght_by_rtn, wght_by_td)

from research.Features.fractional_diff import fracDiff_FFD, fracDiff, min_value, plot_min_ffd
#from research.Features.futures_roll import *
#from research.Features.PCA_weight_dist import *

from research.Ensemble.seq_bts_bagging import BaggingClassifier, BaggingRegressor