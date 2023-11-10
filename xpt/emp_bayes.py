from xi_fit import FitRoutine

import gvar as gv 
import lsqfit
import numpy as np

class EmpBayes(FitRoutine):
    def __init__(self,
                 prior:dict,
                 grouping:str
                 ):
        self.prior = prior
        self.grouping = grouping
    def _empbayes_grouping(self):
            '''
            routine adapted from @millernb
            '''
            zkeys = {}
            if self.grouping =='all':
                for param in self.prior:
                    zkeys[param] = [param]
            # include particle choice xi or xi_st to fill inside bracket
            elif self.grouping =='order':
                # vary all light quark terms together, strange terms together
                zkeys['chiral_llo'] = ['m_{xi,0}', 'm_{xi_st,0}']
                zkeys['chiral_lo'] = ['s_{xi}', 's_{xi,bar}']
                zkeys['chiral_nlo'] = ['g_{xi,xi}',
                                    'g_{xi_st,xi}', 'g_{xi_st,xi_st}']
                zkeys['chiral_n2lo'] = [
                    'b_{xi,4}', 'b_{xi_st,4}', 'a_{xi,4}', 'a_{xi_st,4}']
                zkeys['disc_nlo'] = ['d_{xi,a}',
                                    'd_{xi_st,a}', 'd_{xi,s}', 'd_{xi_st,s}']
                zkeys['disc_n2lo'] = [
                    'd_{xi,aa}', 'd_{xi,al}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}']

            # discretization effects
            elif self.grouping =='disc':
                zkeys['chiral'] = ['m_{xi,0}', 'm_{xi_st,0}', 's_{xi}', 's_{xi,bar}', 'g_{xi,xi}', 'g_{xi_st,xi}', 'g_{xi_st,xi_st}',
                                'b_{xi,4}', 'b_{xi_st,4}', 'a_{xi,4}', 'a_{xi_st,4}']
                zkeys['disc'] = ['d_{xi,a}', 'd_{xi_st,a}', 'd_{xi,s}', 'd_{xi_st,s}',
                                'd_{xi,aa}', 'd_{xi,al}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}']
                
            elif self.grouping =='disc_only':
                zkeys['disc'] = ['d_{xi,a}', 'd_{xi_st,a}', 'd_{xi,s}', 'd_{xi_st,s}',
                                'd_{xi,aa}', 'd_{xi,al}', 'd_{xi,as}', 'd_{xi,ls}', 'd_{xi,ss}',
                                'd_{xi_st,aa}','d_{xi_st,al}' ,'d_{xi_st,as}','d_{xi_st,ls}', 'd_{xi_st,ss}']
                
            all_keys = [k for g in zkeys for k in zkeys[g]]
            prior_keys = list(FitRoutine._make_prior())
            ignored_keys = set(all_keys) - set(prior_keys)

            # Don't determine empirical priors in param not in model
            for group in zkeys:
                for key in ignored_keys:
                    if key in ignored_keys and key in zkeys[group]:
                        zkeys[group].remove(key)

            return zkeys

    def _make_empbayes_fit(self,grouping='disc_only',observable=None):
        grouping = self._empbayes_grouping
        _counter = {'iters' : 0, 'evals' : 0}

        z0 = gv.BufferDict()
        for group in self._empbayes_grouping():
            z0[group] = 1.0

        def analyzer(arg):
            _counter['evals'] += 1
            print("\nEvals: ",_counter['evals'], arg,"\n")
            print(type(arg[0]))
            return None
        models =FitRoutine._make_models()
        fitter = lsqfit.MultiFitter(models=models)

        fit, z = fitter.empbayes_fit(z0, fitargs=self._make_fitargs, maxit=20, analyzer=analyzer,tol=0.1)
        _empbayes_fit = fit

        return _empbayes_fit

    def _make_fitargs(self, z):
        '''
        preparing fit args that will be passed to fitter.empbayes_fit
        '''
        data =data
        prior =FitRoutine._make_prior(z=z)

        # Ideally:
        # Don't bother with more than the hundredth place
        # Don't let z=0 (=> null GBF)
        # Don't bother with negative values (meaningless)
        # But for some reason, these restrictions (other than the last) cause empbayes_fit not to converge
        # multiplicity = {}
        # for key in z:
        #     multiplicity[key] = 0
        #     z[key] = np.abs(z[key])

        # Helps with convergence (minimizer doesn't use extra digits -- bug in lsqfit?)
        def sig_fig(x): return np.around(
            x, int(np.floor(-np.log10(x))+3))  # Round to 3 sig figs

        def capped(x, x_min, x_max): return np.max([np.min([x, x_max]), x_min])

        zkeys =self._empbayes_grouping()
        zmin = 1e-2
        zmax = 1e3
        for group in z.keys():
            for param in prior.keys():
                if param in zkeys[group]:
                    z[group] = sig_fig(capped(z[group], zmin, zmax))
                    prior[param] = gv.gvar(0, 1) * z[group]

        return dict(data=data,prior=prior)