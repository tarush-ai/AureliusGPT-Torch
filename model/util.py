import numpy as np
from config import d_model, max_seq_length

class Util:
    def sinusoidal(self):
        PE = np.zeros((max_seq_length, d_model))
        
        for pos in range(max_seq_length):
            for i in range(0, d_model, 2):
                div_term = 10000 ** (i / d_model)
                PE[pos, i] = np.sin(pos / div_term)
                if i + 1 < d_model:
                    PE[pos, i + 1] = np.cos(pos / div_term)
                    
        return PE