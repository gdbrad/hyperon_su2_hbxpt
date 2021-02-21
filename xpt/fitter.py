import lsqfit
import numpy as np
import gvar as gv
import sys
import os

import fitter.special_functions as sf

class fitter(object):

    def __init__(self, prior, fit_data, model_info, observable, ensemble_mapping=None, prior_interpolation=None):
        self.prior = prior
        self.prior_interpolation = prior_interpolation
        self.fit_data = fit_data
        self.model_info = model_info.copy()
        self.observable = observable
        
        # attributes of fitter object to fill later
        self.empbayes_grouping = None
        self._counter = {'iters' : 0, 'evals' : 0} # To force empbayes_fit to converge?
        self._empbayes_fit = None
        self._fit = None
        self._fit_interpolation = None
        self._simultaneous = False
        self._y = None
        self._ensemble_mapping = ensemble_mapping # Necessary for LO t0, w0 interpolations 


    def __str__(self):
        return str(self.fit)
