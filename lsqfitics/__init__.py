import lsqfit
import gvar as gv
import numpy as np
import copy
import matplotlib.pyplot as plt
import vegas
import itertools

class nonlinear_fit(lsqfit.nonlinear_fit):
    def __init__(self, *args, dcut=None, nparams=None, **kwargs):
        """Constructs an lsqfitics.nonlinear_fit. This object is identical to an lsqfit.nonlinear_fit except with the addition of a few information criteria as properties
        Args:
            dcut=None: number of data point excluded from a fit (see arXiv:2208.14983 [stat.ME])
            k=None: dimension of parameter vector; if _None_, estimates `k` from the length of the flattened prior (this will only work if the prior contains no extraneous information!)
            *args, **kwargs: same as for initializaing an lsqfit.nonlinear_fit object
        Returns:
            an lsqfitics.nonlinear_fit
        """
        super(nonlinear_fit, self).__init__(*args, **kwargs)

        self.dcut = dcut
        if nparams is None:
            nparams = self.p0.size # estimate number of parameters 
        self.nparams = nparams

        self._chi2_prior = None
        self._PAIC = None
        self._BPIC = None


    @classmethod
    def from_fit(cls, fit, dcut=None, nparams=None, deepcopy=True):
        """Converts an lsqfit.nonlinear_fit object into an lsqfitics.nonlinear_fit object
        Args:
            fit: an lsqfit.nonlinear_fit object
            dcut=None: number of data point excluded from a fit (see arXiv:2208.14983 [stat.ME])
            k=None: dimension of parameter vector; if _None_, estimates `k` from the length of the flattened prior (this will only work if the prior contains no extraneous information!)
            deepcopy=True: make a deepcopy of `fit`; potentially memory/time could be saved by skipping this step, but the input fit is fundamentally changed
        Returns:
            an lsqfitics.nonlinear_fit object
        """
        if deepcopy:
            fit_copy = copy.deepcopy(fit)
        else:
            fit_copy = fit

        fit_copy.__class__ = nonlinear_fit
        fit_copy.dcut = dcut
        if nparams is None:
            nparams = fit_copy.p0.size
        fit_copy.nparams = nparams

        fit_copy._chi2_prior = None
        fit_copy._PAIC = None
        fit_copy._BPIC = None

        return fit_copy


    @property
    def chi2_prior(self):
        """The prior chi-squared statistic at the posterior mode $a^*$
        $$
        \sum_{i,j} (a^*_i - \tilde a_i) \tilde \Sigma^{-1}_{ij} (a^*_j - \tilde a_j) 
        $$
        Returns:
            prior chi2
        """
        if self._chi2_prior is None:
            self._chi2_prior = self.evalchi2_prior(self.p)
        return self._chi2_prior


    def evalchi2_prior(self, p):
        """Calculates the prior chi-squared statistic for a given set of input parameters
        $$
        \sum_{i,j} (a_i - \tilde a_i) \tilde \Sigma^{-1}_{ij} (a_j - \tilde a_j) 
        $$
        """

        # I think this would also work also, but it doesn't seem appreciably more performant
        # output = np.sum(gv.mean(self._chiv(p.flatten())**2)[:len(p.flatten())])

        output = gv.mean((p.flatten() - self.prior.flatten())
                @ np.linalg.inv(gv.evalcov(self.prior.flatten())) 
                @ gv.mean((p.flatten() - self.prior.flatten())))
        return output


    # # # # # # # ## # # # # # # #
    # Information criteria below #
    # # # # # # # ## # # # # # # #
    @property
    def ABIC(self):
        """Compute the Akaike information criterion from the posterior rather than the likelihood. Unlike the BAIC, this IC has not been shown to be asymptotically unbiased 
        Returns:
            the ABIC
        """
        return self.BAIC + self.chi2_prior


    @property
    def BAIC(self):
        """Compute the Bayesian Akaike information criterion (BAIC) either with or without `dcut`, depending on whether it was specified when creating the lsqfitics.nonlinear_fit
        Returns:
            the BAIC
        """
        output = self.chi2 + 2 *self.nparams 
        if self.dcut is not None:
            output += 2 *self.dcut 

        return output - self.chi2_prior


    @property
    def BPIC(self):
        """Compute the Bayesian predictive information criterion (BPIC)
        Returns:
            the BPIC
        """
        if self._BPIC is None:
            expval = vegas.PDFIntegrator(self.p, pdf=self.pdf, limit=5)
            
            try:
                #expval(self.evalchi2, neval=1000, nitn=10)
                integral = expval(lambda p : gv.mean(self.evalchi2_prior(p)), neval=1000, nitn=10)
            except ValueError:
                return np.inf

            output = self.chi2 - gv.mean(integral) + 3 *self.nparams
            if self.dcut is not None:
                output += 2 *self.dcut

            self._BPIC = output

        return self._BPIC
    

    @property
    def PAIC(self):
        """Compute the posterior averaging information criterion as proposed by Zhou (arXiv:2009.09248 [stat.ME])
        Returns:
            the PAIC
        """
        if self._PAIC is None:
            expval = vegas.PDFIntegrator(self.p, pdf=self.pdf, limit=5)
            
            try:
                #expval(self.evalchi2, neval=1000, nitn=10)
                integral = expval(lambda p : gv.mean(self.evalchi2(p) - self.evalchi2_prior(p)), neval=1000, nitn=10)
            except ValueError:
                return np.inf

            output = gv.mean(integral + 2 *self.nparams)
            if self.dcut is not None:
                output += 2 *self.dcut

            self._PAIC = output

        return self._PAIC


    @property
    def renorm_logGBF(self):
        """A renormalized version of lsqfit's definition of logGBF; rather than compute weights per exp(logGBF - logGBF_max), compute per exp(-(logGBF - logGBF_min)/2)
        Returns:
            -2 *logGBF
        """
        return -2 *self.logGBF


def from_fit(fit, dcut=None):
    """Converts an lsqfit.nonlinear_fit object into an lsqfitics.nonlinear_fit object
    Args:
        fit: an lsqfit.nonlinear_fit object
        dcut: number of data points excluded from fit
    Returns:
        an lsqfitics.nonlinear_fit object
    """
    return nonlinear_fit.from_fit(fit, dcut)


def calculate_weights(fits, ic, dcuts=None):
    """Calculates the weights from a list or dict of fits. 

    Args:
        fits: a list or dict of lsqfit.nonlinear_fit or lsqfitics.nonlinear_fit objects
        ic: information criterion ('logGBF', 'BAIC')
        dcuts: a list or dict specifying the number of data points excluded from each fit
    Returns:
        the weights of each fit as a list or dict, depending of whether `fits` is a list or dict, respectively
    Raise:
        ValueError: if `ic` is not a valid information_criterion
    """
    permitted_ics = ['logGBF', 'ABIC', 'BAIC', 'BPIC', 'PAIC']
    if ic not in permitted_ics:
        raise ValueError("Not a valid information criterion; must be in %s"%permitted_ics)

    convert_to_weight = lambda x, xmin : np.exp(-(x-xmin)/2)

    ic_values = []
    for j, fit in enumerate(fits):
        if isinstance(fits, dict):
            fit = fits[fit]

        if dcuts is None:
            dcut = None
        elif isinstance(dcuts, dict):
            dcut = dcuts[fit]
        else:
            dcut = dcuts[j]

        temp_fit = fit
        if not isinstance(temp_fit, nonlinear_fit):
            temp_fit = from_fit(temp_fit, dcut=dcut)

        if ic == 'logGBF':
            ic_values.append(temp_fit.renorm_logGBF)
        elif ic == 'BAIC':
            ic_values.append(temp_fit.BAIC)
        elif ic == 'ABIC':
            ic_values.append(temp_fit.ABIC)
        elif ic == 'PAIC':
            ic_values.append(temp_fit.PAIC)
        elif ic == 'BPIC':
            ic_values.append(temp_fit.BPIC)
    
    ic_min = np.min(ic_values)
    weights = convert_to_weight(ic_values, ic_min)
    weights = weights / np.sum(weights) # normalize

    if isinstance(fits, dict):
        weights = dict(zip(list(fits), weights))

    return weights


def calculate_average(values, weights):
    """Calculates the model average given a set of gvar variables and their associated weights
    Args:
        values: a list or dict of gvars
        weights: a list or dict of normalized/unnormalized weights
    Returns:
        the model average
    """

    if isinstance(values, dict):
        values = [v for _, v in sorted(values.items())]
    if isinstance(weights, dict):
        weights = [v for _, v in sorted(weights.items())]

    # normalize
    weights = weights / np.sum(weights)

    mean = np.sum(gv.mean(values) *weights)
    process_var = np.sum(gv.sdev(values)**2 *weights)
    means_var = np.sum(gv.mean(values)**2 *weights) - mean**2

    return gv.gvar(mean, np.sqrt(process_var + means_var))


def plot_weights(fits, ax=None, x=None, show_legend=True, ics=None, dcuts=None):
    """Plots the weights of each 
    Args:
        fits: a list or dict of lsqfitics.nonlinear_fit or lsqfit.nonlinear_fit objects
        ax: axis to draw plot on
        x: x coordinates for weights
        show_legend: show labels for information criteria
        ics: list of information criteria to plot weights of
    Returns:
        a matplotlib.pyplot.axis which plots the weights for each information criteria in `ics`
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = itertools.cycle(prop_cycle.by_key()['color'])

    if ax is None:
        fig, ax = plt.subplots()

    if ics is None:
        ic_list = ['logGBF', 'ABIC', 'BAIC', 'BPIC', 'PAIC']
    else:
        ic_list = ics

    for ic in ic_list:
        weights = calculate_weights(fits, ic=ic, dcuts=dcuts)
        if isinstance(weights, dict):
            weights = np.array([v for k, v in weights.items()])
        
        if x is None:
            x = np.array(range(len(weights)))

        color = next(colors)
        ax.plot(x, weights, '--', label=ic, color=color)

        # only show markers for points with prob > 1%
        idx = np.greater(weights, 0.01)
        ax.plot(x[idx], weights[idx], ls='', marker='.', color=color)
        
    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylabel('prob')

    return ax


if __name__ == '__main__':
    pass