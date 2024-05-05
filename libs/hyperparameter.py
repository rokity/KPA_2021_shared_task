import numpy as np

class HyperParameters:


    def __init__(self,epochs:list,learning_rate:list,batch_size:list,drop_out:list,unit_1:list) -> None:
        self.hyperparams = {
            "epochs":epochs,
            "learning_rate":learning_rate,
            "batch_size":batch_size,
            "drop_out":drop_out,
            "unit_1":unit_1,
        }
    
    def explodeCombination(self):
        # transform hyperparam dictionary to a list of all possible combinations of hyperparameter's values
        mesh = np.array(np.meshgrid(*self.hyperparams.values()))
        self.hyperparams = mesh.T.reshape(-1, len(self.hyperparams))
        
        return 
    
