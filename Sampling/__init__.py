# to prevent import of non-necessary func, import individual func
try:
    from research.Sampling.sample_unique import (wght_by_coevents, num_co_events, av_unique,
                                                 idx_matrix, mp_idx_matrix, 
                                                 seq_bts, mp_seq_bts, 
                                                 MC_seq_bts, MT_MC, 
                                                 wght_by_rtn, wght_by_td)
except:
    from Sampling.sample_unique import (wght_by_coevents, num_co_events, av_unique,
                                        idx_matrix, mp_idx_matrix, 
                                        seq_bts, mp_seq_bts, 
                                        MC_seq_bts, MT_MC, 
                                        wght_by_rtn, wght_by_td)