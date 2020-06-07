import numpy as np
import pandas as pd
#===================================================
# try OOP
class _weight_base():
    def __init__(self, X, axis):
        
        self.X = X,
        self.axis = axis
         
    def _weight(self):
        X = self.X
        axis = self.axis
        _shape = X.shape[axis]
        _weight = np.ones(_shape)
        print(_shape)
        _weight = pd.Series(_weight, index = X.index).div(_shape) # if not weight assigned equal weight given
        return _weight 
         

class _sample_weight(_weight_base):
    
    def __init__(self, 
                 X: pd.DataFrame =None,
                 axis: int = 0):
        
        super().__init__(
                X, 0)
        
        
class _class_weight(_weight_base):
    
    def __init__(self, 
             X: pd.DataFrame = None,
             axis:int = 1):
        
        super().__init__(
                X, 1)
        