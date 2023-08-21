import yaml 
import argparse
from pathlib import Path
import sys 
import os


cwd = Path(os.getcwd())
# Assuming your notebook is in the project root, set the project root as cwd
project_root = cwd.parent
print(project_root)
# If your notebook is in a subdirectory of the project root, you can modify the path accordingly:
# project_root = cwd.parent  # Go up one directory level
# project_root = cwd.parent.parent  # Go up two directory levels
# Add the project root directory to sys.path
sys.path.insert(0, str(project_root))
from xpt import i_o
from tests import tests

def main():
    parser = argparse.ArgumentParser(
        description='Run tests of hyperon xpt expressions. Outputs a gvar at each order in the xpt expressions(eg. llo up to n2lo)')
    parser.add_argument('system',    help='test cascade system or lambda/sigma')
    args = parser.parse_args()

    input_output = i_o.InputOutput(units='fpi',scheme='w0_org',system=args.system)
    with open('../xpt/models.yaml', 'r') as f:
        models = yaml.load(f, Loader=yaml.FullLoader)
    xi_models = models['models']['xi_system']
    lam_sigma_models = models['models']['lam_sigma_system']

    # test_out = tests.VerboseFitfcn()
    mdl_key='lam:sigma:sigma_st:l_lo:d_n2lo:s_lo'
    xpt_mdl = 'lam:sigma:sigma_st:l_n2lo:d_n2lo:s_n2lo:x_n2lo'
    xpt_mdl_nlo = 'lam:sigma:sigma_st:l_n2lo:d_n2lo:x_nlo'
    xi_xpt = 'xi:xi_st:d_n2lo:l_n2lo:s_n2lo:x_n2lo'


    _model_info = lam_sigma_models[mdl_key]
    _model_info_xpt = lam_sigma_models[xpt_mdl]
    _model_info_xpt_nlo = lam_sigma_models[xpt_mdl_nlo]
    _model_info_xi = xi_models[xi_xpt]
    check_lam = tests.VerboseFitfcn(model_info=_model_info,phys_point_data=input_output.get_data_phys_point())
    check_lam_xpt = tests.VerboseFitfcn(model_info=_model_info_xpt,phys_point_data=input_output.get_data_phys_point())
    check_lam_xpt_nlo= tests.VerboseFitfcn(model_info=_model_info_xpt_nlo,phys_point_data=input_output.get_data_phys_point())
    check_xi_xpt_n2lo= tests.VerboseFitfcn(model_info=_model_info_xi,phys_point_data=input_output.get_data_phys_point())

    print(check_lam_xpt)
    print(check_xi_xpt_n2lo)
    # print(check_lam_xpt_nlo)

if __name__ == "__main__":
    main()