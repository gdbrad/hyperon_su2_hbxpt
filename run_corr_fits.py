# import tqdm
import h5py as h5
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import gvar as gv
import pandas as pd
import os 
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import importlib
import argparse
import platform

import corr_fitter.bs_utils as bs 
import corr_fitter.corr_fit_analysis as fa
import corr_fitter.load_data_priors as ld
import corr_fitter.corr_fitter as fitter
matplotlib.rcParams['figure.figsize'] = [10, 8]

importlib.reload(fa)


def main():
    parser = argparse.ArgumentParser(description='analysis of simult. fit to the hyperon spectrum')
    parser.add_argument('fit_params', help='input file to specify fit')
    parser.add_argument('--fit_type',help='specify simultaneous hyperon fit with or without delta,nucleon correlators')
    parser.add_argument('--normalize',help='normalize correlator data?',default=False,action='store_true')
    parser.add_argument('--fix_prior',help='optimize priors after first fit?',default=False,action='store_true')
    parser.add_argument('--pdf',help='generate a pdf and output plot?',default=False,action='store_true')
    parser.add_argument('--bs',help='perform bootstrapping?',default=False, action='store_true') 
    parser.add_argument('--bsn',help='number of bs samples',type=int,default=2000)
    parser.add_argument('--bs_write',help='write bs file?',default=False,action='store_true') 
    parser.add_argument('--post',help='perform a final save of fit posterior to a pickle file',default=False,action='store_true')


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
    if abbr  == 'a12m180S' or abbr == 'a12m220':
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
    nucleon  = ld.get_raw_corr(file,p_dict['abbr'],particle='proton')
    lam      = ld.get_raw_corr(file,p_dict['abbr'],particle='lambda_z')
    xi       = ld.get_raw_corr(file,p_dict['abbr'],particle='xi_z')
    xi_st    = ld.get_raw_corr(file,p_dict['abbr'],particle='xi_star_z')
    sigma    = ld.get_raw_corr(file,p_dict['abbr'],particle='sigma_p')
    sigma_st = ld.get_raw_corr(file,p_dict['abbr'],particle='sigma_star_p')
    delta    = ld.get_raw_corr(file,p_dict['abbr'],particle='delta_pp')

    ncfg     = xi['PS'].shape[0]
    bs_list = bs.get_bs_list(Ndata=ncfg,Nbs=2000)

    # hi = ld.resample_correlator(raw_corr=nucleon,bs_list=bs_list,n=2000)
    # print(hi)


    model_type = args.fit_type
    prior = fp.prior
    # print(prior)
    
    if args.fit_type == 'hyperons':
        hyperons = fa.corr_fit_analysis(t_range=p_dict
        ['t_range'],simult=True,t_period=64,states=p_dict['hyperons'],p_dict=p_dict, 
        n_states=p_dict['n_states'],prior=prior,nucleon_corr_data=None,lam_corr_data=lam, 
        xi_corr_data=xi,xi_st_corr_data=xi_st,sigma_corr_data=sigma,
        sigma_st_corr_data=sigma_st,model_type=model_type)
        print(hyperons)
        fit_out = hyperons.get_fit()
        
        out_path = 'fit_results/{0}/{1}/'.format(p_dict['abbr'],model_type)

        if args.post:

            post = hyperons.posterior()
            ld.pickle_out(fit_out=post,out_path=out_path,species="baryon")
            print(ld.print_posterior(out_path=out_path))
        if args.pdf:
            plot1 = hyperons.return_best_fit_info()
            plot2 = hyperons.plot_effective_mass(t_plot_min=0, t_plot_max=40,model_type=model_type, 
            show_plot=True,show_fit=True)
            plot3 = hyperons.plot_effective_wf(model_type=model_type, t_plot_min=0, t_plot_max=40, 
            show_plot=True,show_fit=True)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                pp.savefig(plot1)
                pp.savefig(plot2)
                # pp.savefig(plot3)

            # accept_fit =  input('is fit ready to be saved ?\nenter Y or hit return for no')
            # if accept_fit: 
            #     # add to manifest directory of final fits 

            
                

            pp.close()

    if args.fit_type == 'all':
        all_baryons = fa.corr_fit_analysis(t_range=p_dict
        ['t_range'],simult=True,t_period=64,states=p_dict['hyperons'],p_dict=p_dict, 
        n_states=p_dict['n_states'],prior=prior, delta_corr_data=delta,
        nucleon_corr_data=nucleon,lam_corr_data=lam, xi_corr_data=xi,xi_st_corr_data=xi_st,
        sigma_corr_data=sigma, sigma_st_corr_data=sigma_st,model_type="all")
        # print(all_baryons)
        fit_out = all_baryons.get_fit()
        out_path = 'fit_results/{0}/{1}/'.format(p_dict['abbr'],model_type)
        ld.pickle_out(fit_out=fit_out,out_path=out_path,species="baryon")
        # print(ld.print_posterior(out_path=out_path))

        if args.pdf:
            plot1 = all_baryons.return_best_fit_info()
            plot2 = all_baryons.plot_effective_mass(t_plot_min=0, t_plot_max=20,model_type=model_type, 
            show_plot=True,show_fit=True)
            # plot3 = all_baryons.plot_stability(model_type='all',corr='proton',t_start=0,t_end=20,show_plot=True)
                # plot3 = all_baryons.plot_effective_wf(model_type=model_type, t_plot_min=0, t_plot_max=40, 
                # show_plot=True,show_fit=True)

            output_dir = 'fit_results/{0}/{1}_{0}'.format(p_dict['abbr'],model_type)
            output_pdf = output_dir+".pdf"
            with PdfPages(output_pdf) as pp:
                pp.savefig(plot1)                
                pp.savefig(plot2)
                # pp.savefig(plot3)
            pp.close()
            # output_pdf.close()

        # if args.bs:

   
        
    ''' xpt routines 
    '''
    # if args.xpt:
    #     model_info = fp.model_info



    



if __name__ == "__main__":
    main()