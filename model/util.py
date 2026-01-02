import numpy as np
from config import d_model, max_seq_length

class Util:
    def sinusoidal(self):
        positionals = []
        for i in range(d_model):            
            if i%2 == 0:
                positionals.append(np.sin(max_seq_length / 10000 ** (i / d_model)))
            else:
                positionals.append(np.cos(max_seq_length / 10000 ** (i / d_model)))
        return np.array(positionals)
             

    
