import numpy as np
import gvar as gv
import sys
import os
import lsqfit
import io
import yaml
import matplotlib
import matplotlib.pyplot as plt

def plot_fit(ens=None,fit_out=None,xparam=None, yparam=None):
        if yparam is None:
            yparam = 'eps_proton'

        x = {}
        y = {}
        c = {}
        #fit = {}

        #plt.axes([0.145,0.145,0.85,0.85])
            
        colors = {
            '06' : '#6A5ACD',
            '09' : '#51a7f9',
            '12' : '#70bf41',
            '15' : '#ec5d57',
        }

        for i in  range(len(ens)):
            for j, param in enumerate([xparam, yparam]):
                if param == 'mp':
                    value = fit_out.y['proton'][i]
                    label = '$\epsilon_p$'

                elif param in ['sigma','sigma_st','xi','xi_st','lam']:
                    value = fit_out.y[param][i]
                    label = '$\epsilon_\pi$'
                
                elif param in ['sigma','sigma_st','xi','xi_st','lam']:
                    value = fit_out.y[param][i]
                    label = '$\epsilon_\pi$'

                elif param == 'eps_pi':
                    value = fit_out.p['eps_pi'][i]
                    label = '$\epsilon_\pi$'
                    #min,max linspace

                elif param == 'eps2_a':
                    value = fit_out.p['eps2_a'][i]
                    label = '$\epsilon_a^2$'


                if j == 0:
                    x[i] = value
                    xlabel = label
                elif j == 1:
                    y[i] = value
                    ylabel = label
        for i in range(len(ens)):
            C = gv.evalcov([x[i], y[i]])
            eVe, eVa = np.linalg.eig(C)
            for e, v in zip(eVe, eVa.T):
                plt.plot([gv.mean(x[i])-1*np.sqrt(e)*v[0], 1*np.sqrt(e)*v[0] + gv.mean(x[i])],
                        [gv.mean(y[i])-1*np.sqrt(e)*v[1], 1*np.sqrt(e)*v[1] + gv.mean(y[i])],
                         alpha=1.0, lw=2)
                plt.plot(gv.mean(x[i]), gv.mean(y[i]), 
                          marker='o', mec='w', zorder=3)


        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
            ncol=len(by_label), bbox_to_anchor=(0,1), loc='lower left')
        plt.grid()
        plt.xlabel(xlabel, fontsize = 24)
        plt.ylabel(ylabel, fontsize = 24)
        #plt.axvline(gv.mean(phys_point_data['eps_pi']), ls='--', label='phys. point')

        fig = plt.gcf()
        plt.close()
        return fig