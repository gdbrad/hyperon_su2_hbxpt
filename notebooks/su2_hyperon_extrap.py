import os
import sys
from pathlib import Path
import warnings
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import yaml
import importlib
import numpy as np
import gvar as gv
import platform
import pprint
# import lsqfitics
cwd = Path(os.getcwd())
# Assuming your notebook is in the project root, set the project root as cwd
project_root = cwd.parent
print(project_root)
# If your notebook is in a subdirectory of the project root, you can modify the path accordingly:
# project_root = cwd.parent  # Go up one directory level
# project_root = cwd.parent.parent  # Go up two directory levels
# Add the project root directory to sys.path
sys.path.insert(0, str(project_root))
# sys.path.append('../')

# local imports 
import xpt.fit_analysis as xfa
import xpt.priors as priors
import xpt.i_o as i_o
import xpt.fit_routine as fit
import xpt.plots as plots
warnings.simplefilter(action="default")
warnings.filterwarnings('ignore')
# Define paths and other variables
if platform.system() == 'Darwin':
    base_dir = '/Users/grantdb/lqcd'
else:
    base_dir = '/home/gmoney/lqcd'

data_dir = os.path.join(base_dir, "data")
hyperon_data_file = os.path.join(data_dir,"hyperon_data.h5")


# gv.load('../scale_setting.p')
with open('../xpt/models.yaml', 'r') as f:
    models = yaml.load(f, Loader=yaml.FullLoader)
xi_models = models['models']['xi_system']
lam_sigma_models = models['models']['lam_sigma_system']


prior = priors.get_prior(units='mev')
prior_fpi = priors.get_prior(units='lam_chi')
input_output = i_o.InputOutput(project_path=data_dir)
data, ensembles = input_output.get_data(units='phys')
data_units_fpi,ensembles = input_output.get_data(units='fpi')
new_prior = input_output.make_prior(data=data,prior=prior)
new_prior_fpi = input_output.make_prior(data=data_units_fpi,prior=prior_fpi)
phys_point_data = input_output.get_data_phys_point()




# # Create a figure with two subplots (one for each baryon 'xi' and 'xi_st')
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
# axes[0].set_title("xi")
# axes[1].set_title("xi_st")

for mdl_key in xi_models:
    model_info = xi_models[mdl_key]
    extrap_analysis = xfa.Xpt_Fit_Analysis(verbose=True, phys_point_data=phys_point_data,
                                data=data, model_info=model_info, prior=new_prior,project_path=data_dir)
# extrap_analysis.plot_model_mass_comparison(models=xi_models)
    # Get the extrapolated masses and chi^2 value for the current model
# extrap_analysis.plot_params(xparam='mpi_sq',observable='xi',show_plot=True)
    # extrap_analysis.plot_params(xparam='eps2_a',observables=['xi','xi_st'],show_plot=True,eps=False)
# extrap_analysis.plot_params_fit(param='a',observable='xi',eps=False)
# extrap_analysis.plot_params_fit(param='epi',observable='xi')

'''model average with lsqfitics'''
# y_fit_ = extrap_analysis.extrapolation()
# print(y_fit_)





