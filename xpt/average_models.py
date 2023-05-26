import lsqfit
import numpy as np
import gvar as gv
import time
#import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import scipy.stats as stats

import fit_routine as fit
import i_o

class Model_Average():
    def __init__(self,fit_collection):
        self.fit_collection = fit_collection
        self.fit_results = i_o.get_fit_collection()
        
        