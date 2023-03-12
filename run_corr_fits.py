from pathlib import Path
import sys
import argparse


import tqdm
import h5py as h5
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import gvar as gv
import pandas as pd
import os 
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import importlib
import platform

from src.corr_fitter import gmo_fit_analysis as fa
from src.corr_fitter import load_data_priors as ld
from src.corr_fitter import gmo_fitter as fitter
from src.corr_fitter import bs_utils as bs 

matplotlib.rcParams['figure.figsize'] = [10, 8]

importlib.reload(fa)


def main():
    '''
    executable which performs analysis of octet and decuplet correlators using the lsqfit module 
    '''
    parser = argparse.ArgumentParser(description='analysis of simult. fit to the baryons')
    parser.add_argument('fit_params', help='input file to specify fit')
    parser.add_argument('--fit_type',help='specify simultaneous baryon fit or individual fits')
    parser.add_argument('pdf',help='generate a pdf and output plot?',default=True)
    parser.add_argument('--bs',help='perform bootstrapping?',default=False, action='store_true')
    parser.add_argument('--bsn',help='number of bs samples',type=int,default=2000) 

    # parser.add_argument('xpt',help='run gmo xpt analysis?',default=True)

    args = parser.parse_args()
    print(args)
    # add path to the input file and load it
    sys.path.append(os.path.dirname(os.path.abspath(args.fit_params)))
    fp = importlib.import_module(
        args.fit_params.split('/')[-1].split('.py')[0])
    if platform.system() == 'Darwin':
        file = '/Users/grantdb/lqcd/data/c51_2pt_octet_decuplet.h5'
    else:
        file = '/home/gmoney/lqcd/data/c51_2pt_octet_decuplet.h5'

    p_dict = fp.p_dict
    abbr = p_dict['abbr']
    if abbr in['a12m180S','a12m220']:
        nucleon_corr = ld.get_raw_corr(file,p_dict['abbr'],particle='proton')
        prior_nucl = {}
        prior = {}
        states=p_dict['states']
        newlist = [x for x in states]
        for x in newlist:
            path = os.path.normpath("./priors/{0}/{1}/prior_nucl.csv".format(p_dict['abbr'],x))
            df = pd.read_csv(path, index_col=0).to_dict()
            for key in list(df.keys()):
                length = int(np.sqrt(len(list(df[key].values()))))
                prior_nucl[key] = list(df[key].values())[:length]
                # prior_nucl['gmo_E'] = list([np.repeat(gv.gvar('0.0030(27)'),8)])
            prior = gv.gvar(prior_nucl)

    # pull in raw corr data
    nucleon = ld.get_raw_corr(file,p_dict['abbr'],particle='proton')
    lam = ld.get_raw_corr(file,p_dict['abbr'],particle='lambda_z')
    xi = ld.get_raw_corr(file,p_dict['abbr'],particle='xi_z')
    xi_st = ld.get_raw_corr(file,p_dict['abbr'],particle='xi_star_z')
    sigma = ld.get_raw_corr(file,p_dict['abbr'],particle='sigma_p')
    sigma_st = ld.get_raw_corr(file,p_dict['abbr'],particle='sigma_star__p')
    delta = ld.get_raw_corr(file,p_dict['abbr'],particle='delta_pp')
    ncfg = xi['PS'].shape[0]

    model_type = args.fit_type
    # prior = ld.fetch_prior(model_type,p_dict)
    
    if args.fit_type == 'all':
        sim_baryons = fa.fit_analysis(t_range=p_dict
        ['t_range'],simult=True,t_period=64,states=p_dict['all'],p_dict=p_dict, n_states=p_dict['n_states'],prior=prior,
        nucleon_corr_data=nucleon,lam_corr_data=lam, xi_corr_data=xi,sigma_corr_data=sigma,model_type=model_type)
        print(sim_baryons)
        fit_out = sim_baryons.get_fit()
        
        out_path = 'fit_results/{0}/{1}/'.format(p_dict['abbr'],model_type)

        ld.pickle_out(fit_out=fit_out,out_path=out_path,species="baryon")
        print(ld.print_posterior(out_path=out_path))
        if args.pdf:
            plot1 = sim_baryons.return_best_fit_info()
            plot2 = sim_baryons.plot_effective_mass(t_plot_min=0, t_plot_max=40,model_type=model_type, 
            show_plot=True,show_fit=True)
            plot3 = sim_baryons.plot_effective_wf(model_type=model_type, t_plot_min=0, t_plot_max=40, 
            show_plot=True,show_fit=True)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                pp.savefig(plot1)
                pp.savefig(plot2)
                # pp.savefig(plot3)

            output_pdf.close()
    
    


   
        
    # individual correlator fits to form "naive" gmo relation
    elif args.fit_type == 'xi':
        prior_xi = {k:v for k,v in prior.items() if 'xi' in k}
        # print(new_d)
        model_type = 'xi'
        xi_ = fa.fit_analysis(t_range=p_dict
        ['t_range'],p_dict=p_dict,simult=False,states=['xi'], t_period=64, n_states=p_dict['n_states'],prior=prior_xi,
        nucleon_corr_data=None,lam_corr_data=None, xi_corr_data=xi_corr,
        sigma_corr_data=None,gmo_corr_data=None,model_type=model_type)
        # print(xi_)
        fit_out = xi_.get_fit()
        print(fit_out.formatall(maxline=True))
        if args.pdf:
            # plot1 = xi_.return_best_fit_info()
            plot2 = xi_.plot_effective_mass(t_plot_min=5, t_plot_max=30, model_type = model_type,show_plot=True,show_fit=True)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                # pp.savefig(plot1)
                pp.savefig(plot2)
            output_pdf.close()

    elif args.fit_type == 'lam':
        prior_lam = {k:v for k,v in prior.items() if 'lam' in k}
        # print(new_d)
        model_type = 'lam'
        lam_ = fitter.fitter(t_range=p_dict
        ['t_range'],t_period=64, n_states=p_dict['n_states'],states=['lam'],prior=prior_lam,
        nucleon_corr=None,lam_corr=gv.dataset.avg_data(lam_corr), xi_corr=None,
        sigma_corr=None,delta_corr=None,gmo_ratio_corr=None,
        piplus_corr=None,kplus_corr=None,model_type=model_type)
        # print(xi_)
        fit_out = lam_.get_fit()
        print(fit_out.formatall(maxline=True))
        if args.pdf:
            plot1 = lam_.return_best_fit_info()
            plot2 = lam_.plot_effective_mass(tmin=None, tmax=None, ylim=None, show_plot=True,show_fit=True)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                pp.savefig(plot1)
                pp.savefig(plot2)
            output_pdf.close()

    elif args.fit_type == 'proton':
        prior_proton = {k:v for k,v in prior.items() if 'proton' in k}
        # print(new_d)
        model_type = 'proton'
        proton_ = fitter.fitter(t_range=p_dict
        ['t_range'],t_period=64, n_states=p_dict['n_states'],states=['proton'],prior=prior_proton,
        nucleon_corr=gv.dataset.avg_data(nucleon_corr),lam_corr=None, xi_corr=None,
        sigma_corr=None,gmo_ratio_corr=None,model_type=model_type)
        fit_out = proton_.get_fit()
        print(fit_out.formatall(maxline=True))
        print(str(np.exp(fit_out.p['proton_log(dE)'][0])))
        if args.pdf:
            plot1 = proton_.return_best_fit_info()
            plot2 = proton_.plot_effective_mass(tmin=None, tmax=None, ylim=None, show_plot=True,show_fit=True)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                pp.savefig(plot1)
                pp.savefig(plot2)
            # output_pdf.close()

    elif args.fit_type == 'sigma':
        prior_sigma = {k:v for k,v in prior.items() if 'sigma' in k}
        # print(new_d)
        model_type = 'sigma'
        sigma_ = fitter.fitter(t_range=p_dict
        ['t_range'],t_period=64, n_states=p_dict['n_states'],states=['sigma'],prior=prior_sigma,
        nucleon_corr=None,lam_corr=None, xi_corr=None,
        sigma_corr=gv.dataset.avg_data(sigma_corr),delta_corr=None,gmo_ratio_corr=None,
        piplus_corr=None,kplus_corr=None,model_type=model_type)
        # print(xi_)
        fit_out = sigma_.get_fit()
        print(fit_out.formatall(maxline=True))
        if args.pdf:
            plot1 = sigma_.return_best_fit_info()
            plot2 = sigma_.plot_effective_mass(tmin=None, tmax=None, ylim=None, show_plot=True,show_fit=True)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                pp.savefig(plot1)
                pp.savefig(plot2)
            output_pdf.close()

    
    ''' xpt routines 
    '''
    # if args.xpt:
    #     model_info = fp.model_info



    



if __name__ == "__main__":
    main()