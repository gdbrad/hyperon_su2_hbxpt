import numpy as np
import gvar as gv
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.transforms import Bbox
import sys
from mpl_toolkits.mplot3d.axes3d import Axes3D
import sys
import copy
import textwrap

sys.setrecursionlimit(10000)

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['figure.figsize']  = [6.75, 6.75/1.618034333]
mpl.rcParams['font.size']  = 20
mpl.rcParams['legend.fontsize'] =  16
mpl.rcParams["lines.markersize"] = 5
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['text.usetex'] = True

#internal xpt modules
from xpt.fit_analysis import Xpt_Fit_Analysis
import xpt.fit_routine as fit
import xpt.i_o as i_o
import xpt.priors as priors
import lsqfitics

def get_data_and_prior_for_unit(unit,system):
    prior = priors.get_prior(units=unit)
    input_output = i_o.InputOutput(units=unit, scheme='w0_imp', system=system, convert_data=False)
    
    data = input_output.perform_gvar_processing()
    new_prior = input_output.make_prior(data=data, prior=prior)
    
    if unit == 'fpi':
        phys_point_data = input_output.get_data_phys_point(fpi_units=True)
    else:
        phys_point_data = input_output.get_data_phys_point(fpi_units=False)
    
    return data, new_prior, phys_point_data


class ModelComparsion:
    '''Final analysis class. Generates a pdf with the model average and plots for each model'''

    verbose = True
    extrapolate = True
    svd_test = False
    svd_tol = 0.06
    discard_cov = True
    def __init__(self,
                models:dict,
                units:str,
                **kwargs):
        self.models = models
        self.units = units
        for key, value in kwargs.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Default args for Xpt_Fit_Analysis
        self.default_args = {
            'verbose': self.verbose,
            'extrapolate': self.extrapolate,
            'svd_test': self.svd_test,
            'svd_tol': self.svd_tol,
            'discard_cov': self.discard_cov
        }

    def physical_points(self):
        physical_points = {
            'lambda' : gv.gvar(1115.683, 0.006),
            'sigma' : np.mean([gv.gvar(g) for g in ['1189.37(07)', '1192.642(24)', '1197.449(30)']]),
            'sigma_st' : np.mean([gv.gvar(g) for g in ['1382.80(35)', '1383.7(1.0)', '1387.2(0.5)']]),
            'xi' : np.mean([gv.gvar(g) for g in ['1314.86(20)', '1321.71(07)']]),
            'xi_st' : np.mean([gv.gvar(g) for g in ['1531.80(32)', '1535.0(0.6)']]),
            'proton' : gv.gvar(938.272,.0000058),
            'lam_chi' : 4 *np.pi *gv.gvar('92.07(57)')
        }
        return physical_points

    def add_fit_results_figure(self, fit_results, title):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')
        results_text = str(fit_results) #
        wrapper = textwrap.TextWrapper(width=80)
        results_text = wrapper.fill(results_text)
        textbox = plt.text(0.5, 0.5, results_text, fontsize=8, ha='center', va='center', wrap=True)
        textbox.set_bbox(dict(facecolor='white', alpha=1, edgecolor='black', boxstyle="round,pad=1"))
        # plt.title(title)
        plt.text(0.1, 0.5, title, rotation='vertical', fontsize=16, ha='center', va='center')
        return fig

    def model_plots(self,system=None):
        if self.compare_type == 'combined':
            # Implement logic for combined plotting here
            pass
        else:
            


            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf_pages = PdfPages(f'model_plots_{system}.pdf')
            # Add model_average output.
            if system == 'xi':
                particles = ['xi','xi_st']
            else:
                particles=['lambda','sigma_st','sigma']
            avg_out, weights = self.model_average(particles=particles)

            avg_out_str = '\n'.join([f"{k}: {v}" for k, v in avg_out.items()])
            weights_str = '\n'.join([f"{k}: {v}" for k, v in weights.items()])
            avg_and_weights_str = f"Model averages:\n{avg_out_str}\n\nWeights:\n{weights_str}"
            fig_avg = plt.figure(figsize=(10, 10))
            plt.text(0.5, 0.5, avg_and_weights_str, fontsize=16, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5, boxstyle="round,pad=1"))
            plt.axis('off')
            pdf_pages.savefig(fig_avg, bbox_inches='tight')
            plt.close(fig_avg)

            fig_comp = self.compare_models(particles)
            pdf_pages.savefig(fig_comp, bbox_inches='tight')
            plt.close(fig_comp)
            for mdl_key in self.models:
                xfa_instance = self.models[mdl_key]

                # fit_out = xfa_instance.fit
                
                # fit_results_fig = self.add_fit_results_figure(fit_out, title='Model:'+mdl_key)
                # pdf_pages.savefig(fit_results_fig)
                # plt.close(fit_results_fig)
                plt.figure(figsize=(10, 10))  # create a new figure with specific size
                plt.text(0, 0, str(xfa_instance), fontsize=8)  # add text to figure
                plt.axis('off')  # hide axes
                pdf_pages.savefig()  # save current figure to pdf
                plt.close()  # close current figure
                if system == 'xi':
                    fig1= xfa_instance.plot_params_fit(param='a',observable='xi',eps=False)
                    fig2=xfa_instance.plot_params_fit(param='a',observable='xi_st',eps=False)
                    pdf_pages.savefig(fig1)
                    pdf_pages.savefig(fig2)
                    plt.close(fig1)
                    plt.close(fig2)
                else:
                    fig1= xfa_instance.plot_params_fit(param='a',observable='lambda',eps=False)
                    fig2=xfa_instance.plot_params_fit(param='a',observable='sigma',eps=False)
                    fig3=xfa_instance.plot_params_fit(param='a',observable='sigma_st',eps=False)
                    pdf_pages.savefig(fig1)
                    pdf_pages.savefig(fig2)
                    pdf_pages.savefig(fig3)
                    plt.close(fig1)
                    plt.close(fig2)
                    plt.close(fig3)
            if system == 'xi':
                fig_mpi = xfa_instance.plot_params(xparam='mpi_sq',observables=['xi','xi_st'],show_plot=True)
                fig_eps = xfa_instance.plot_params(xparam='eps2_a',observables=['xi','xi_st'],show_plot=True,eps=False)
                pdf_pages.savefig(fig_mpi)
                pdf_pages.savefig(fig_eps)
                plt.close(fig_mpi)
                plt.close(fig_eps)

            pdf_pages.close()

    def model_average(self,particles):
        avg = {}
        avg_out = {}
        fit_collection = {}
        for mdl_key in self.models:
            xfa_instance= self.models[mdl_key]
            fit_out = xfa_instance.fit
            fit_collection[mdl_key] = fit_out
            for part in particles:
                if self.units == 'phys':
                    avg[part] = fit_out.p[f'm_{{{part},0}}']          
                if self.units =='fpi':
                    avg[part] = fit_out.p[f'm_{{{part},0}}'] * 4 *np.pi *gv.gvar('92.07(57)')          

        weights = lsqfitics.calculate_weights(fit_collection,'logGBF')
        # Sort weights dictionary by values in descending order
        weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}
        for part in particles:
            avg_out[part] = lsqfitics.calculate_average(values=avg[part],weights=weights)

        return avg_out,weights

    def compare_models(self,particles=None):
        models = []
        extrapolated_masses = {particle: [] for particle in particles}
        q = []
        chi2_values_ = []
        weights = []
        _, weight_dict = self.model_average(particles)
        for mdl_key in self.models:
            weight_value = weight_dict[mdl_key]
            weights.append(weight_value)  # assuming th
            xfa_instance = self.models[mdl_key]
            mass = xfa_instance.extrapolation(observables=['mass'])
            # print(mass)
            models.append(mdl_key)
            info = xfa_instance.fit_info
            chi2_ = info['chi2/df']
            q_ =  info['Q']
            
            for particle in particles:
                extrapolated_mass = mass[particle]['mass']
                extrapolated_masses[particle].append(extrapolated_mass)
            chi2_values_.append(chi2_)
            q.append(q_)
        chi2_values = np.array(chi2_values_)
        q_values = np.array(q)

        fig, axs = plt.subplots(nrows=len(particles), ncols=3, figsize=(12, 6*len(particles)))
        # Create y values for scatter plot (this is just a range of integers)
        y_values = range(len(models))

        for idx, particle in enumerate(particles):
            physical_point_ = self.physical_points()
            physical_point = physical_point_[particle]
            means  = [gv.mean(val) for val in extrapolated_masses[particle]]
            stddevs = [gv.sdev(val) for val in extrapolated_masses[particle]]

            # Plot the extrapolated masses for each particle with error bars
            scatter = axs[idx, 0].errorbar(means, y_values, xerr=stddevs, fmt='o', alpha=0.6)
            axs[idx, 0].axvline(x=gv.mean(physical_point), color='r', linestyle='--')
            axs[idx, 0].set_xlabel('Extrapolated Mass')
            axs[idx, 0].set_ylabel('Models')
            axs[idx, 0].set_yticks(y_values)
            axs[idx, 0].set_yticklabels(models)
            axs[idx, 0].set_xlim([min(means)-max(stddevs)*1.5, max(means)+max(stddevs)*1.5])  # adjust this for desired zoom
            axs[idx, 0].set_title(particle)
            axs[idx, 0].legend()

            axs[idx, 1].scatter(chi2_values, y_values, alpha=0.6)
            axs[idx, 1].set_xlabel('Chi2/df')
            axs[idx, 1].set_yticks([])  

            # Weights subplot
            axs[idx, 2].barh(y_values, weights, alpha=0.6)  # Assuming weights is a simple list; adjust as needed
            axs[idx, 2].set_xlabel('Weights')
            # axs[idx, 2].set_xlim([min(weights)-0.2, max(weights)+0.2])  # Adjust the x limits if required
            axs[idx, 2].set_yticks([])  # Remove y-ticks for consistency
        plt.tight_layout()
        return fig