try:
    from research.Util.multiprocess import mp_pandas_obj
    from research.Util.volatility import vol, parksinson_vol, garman_class_vol, yang_zhang_vol
    from research.Util.Yahoo_Fin_data import YF_data
    
    from research.Util.root_methods import * #I don't really use it
    
    # Option related
    from research.Util.bsm_model import *
    from research.Util.opt_prob_calculator import * 
except:
    from Util.multiprocess import mp_pandas_obj
    from Util.volatility import vol, parksinson_vol, garman_class_vol, yang_zhang_vol
    from Util.Yahoo_Fin_data import YF_data
    
    from Util.root_methods import * #I don't really use it
    
    # Option related
    from Util.bsm_model import *
    from Util.opt_prob_calculator import * 