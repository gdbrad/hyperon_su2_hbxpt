import fit_routine as fit
import numpy as np
import gvar as gv
import matplotlib
import matplotlib.pyplot as plt

# Set defaults for plots
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

class Mass_Combinations(object):
    
    def __init__(self,lam=None, sigma=None, xi=None, delta=None, sigma_st=None, xi_st=None, omega=None, p=None):
        # mev or gev? 
        self.N = 1/2 * (gv.gvar(939.565413,0.00006) + gv.gvar(938.272081,0.00006))
        self.lam = lam
        self.sigma = sigma
        self.xi = xi
        self.delta = delta
        self.sigma_st = sigma_st
        self.xi_st = xi_st
        self.Omega = omega #mev or gev
        self.N_c = 3
        self.p = p
        self.eps = (p['m_k']**2 - p['m_pi']**2) / (p['lam_chi']**2) *1e-6 # since lam_chi ~ 1 gev
        self.eps2 = self.eps**2
        self.eps3 = self.eps**3
        # isolate coefficient
        # superscript: {flavor su(3) representation, spin su(2) representation}  
        self.c = {}
        # 8 isospin multiplets of ground state baryons 
        self.c['c^{1,0}_0'] = self.m1_c
        self.c['c^{1,0}_2'] = self.m2_c   
        self.c['c^{8,0}_1'] = self.m3_c
        self.c['c^{8,0}_2']  = self.m4_c
        self.c['c^{8,0}_3'] = self.m5_c   
        self.c['c^{27,0}_2'] = self.m6_c   
        self.c['c^{27,0}_3'] = self.m7_c    
        self.c['c^{64,0}_3'] = self.m8_c

        # coeffs in jenkins paper not in 1/n_c paper:
        #    
        # self.c['c^{1,0}_1']  = self.mA      
        # self.c['c^{8,0}_35'] = self.mB
        # self.c['c^{8,0}_405'] = self.mC 
        # self.c['c^{27,0}_405'] = self.mD 

        #mass combinations given by coefficient value in table
        self.M = {}
        #su(3) singlets
        self.M['M_1'] = self.m1
        self.M['M_2'] = self.m2
        #flavor-octet
        self.M['M_3'] = self.m3
        self.M['M_4'] = self.m4
        self.M['M_5'] = self.m5
        #flavor-27 mass combs
        self.M['M_6'] = self.m6
        self.M['M_7'] = self.m7
        #supressed by 3 powers of su(3) breaking and 1/nc^2
        self.M['M_8'] = self.m8
        self.M['M_A'] = self.mA
        self.M['M_B'] = self.mB
        self.M['M_C'] = self.mC
        self.M['M_D'] = self.mD

        # scale invariant mass combination
        self.R = {}
        #su(3) singlets
        self.R['R_1'] = self.R_1
        self.R['R_2'] = self.R_2
        #flavor-octet
        self.R['R_3'] = self.R_3
        self.R['R_3_eps'] = self.R_3_eps
        self.R['R_4'] = self.R_4
        self.R['R_4_eps'] = self.R_4_eps
        self.R['R_5'] = self.R_5
        self.R['R_5_eps'] = self.R_5_eps
        #flavor-27 mass combs
        self.R['R_6'] = self.R_6
        self.R['R_6_eps2'] = self.R_6_eps2
        self.R['R_7'] = self.R_7
        self.R['R_7_eps2'] = self.R_7_eps2
        #supressed by 3 powers of su(3) breaking and 1/nc^2
        self.R['R_8'] = self.R_8
        self.R['R_8_eps3'] = self.R_8_eps3
        self.R['R_A'] = self.R_A
        self.R['R_B'] = self.R_B
        self.R['R_C'] = self.R_C
        self.R['R_D'] = self.R_D


    # calc 1/nc expansion of baryon mass operator for perturbative 
    # su3 flavor-symmetry breaking

    # M = M^1,0 + M^8,0 + M^27,0 + M^64,0
    
    @property
    def plot_baryon_mass_mpi(self):
        return self._plot_baryon_mass_mpi()
    
    def _plot_baryon_mass_mpi(self):
        mass = {}
        keys = self.M.keys()
        print(keys)
        y = np.array(self.M.values())
        print(y)
        mpi = np.array(self.p['m_pi'])
        #x = np.arange(mpi)
        print(mpi)
        fig = plt.plot(mpi,y)
        return fig
        # for k in keys:
        #     x = np.arange(keys.shape[0])
        #     print(x)
        #     mass[k] = self.M.values[x]
        #     plt.errorbar(x=x, y=[d.mean for d in mass[k]], yerr=[d.sdev for d in mass[k]], fmt='o',
        #         capsize=3, capthick=2.0, elinewidth=5.0, label=k)
        # if lim is not None:
        #     plt.xlim(lim[0], lim[1])
        #     plt.ylim(lim[2], lim[3])
        # plt.legend()
        # plt.xlabel('$t$')
        # plt.ylabel('$M_{eff}$')

        # fig = plt.gcf()
        # plt.close()
        # return fig

    #define operators


    # @property
    # def 1(self):
    #     return self._1()

    # def _1(self):
    #     output = 


    @property 
    def m1(self):
        return self._m1() * 0.001 
        #print(self.N)  

    def _m1(self):
        #output = 0
        output = 25*(2*self.N + self.lam + 3*self.sigma + 2*self.xi)
        - (4*(4*self.delta + 3*self.sigma_st + 2*self.xi_st + self.Omega))
        return output 
    @property 
    def m2(self):
        return self._m2() * 0.001

    def _m2(self):
        output = 5*(2*self.N + self.lam + 3*self.sigma + 2*self.xi)
        - (4*(4*self.delta + 3*self.sigma_st + 2*self.xi_st + self.Omega))
        return output 
    @property 
    def m3(self):
        return self._m3() * 0.0001

    def _m3(self):
        output = 5*(6*self.N + self.lam - 3*self.sigma - 4*self.xi)
        - (2*(2*self.delta - self.sigma_st - self.Omega))
        return output
    
    @property 
    def m4(self):
        return self._m4() * 0.0001
    def _m4(self):
        output = self.N + self.lam - 3*self.sigma + self.xi
        return output
    
    @property 
    def m5(self):
        return self._m5() * 0.0001
    def _m5(self):
        output = (-2*self.N + 3*self.lam - 9*self.sigma + 8*self.xi)
        + (2*(2*self.delta - self.xi_st - self.Omega))
        return output

    @property 
    def m6(self):
        return self._m6() * 0.00001

    def _m6(self):
        output = 35*(2*self.N - 3*self.lam - self.sigma + 2*self.xi)
        - (4*(4*self.delta - 5*self.sigma_st - 2*self.xi_st + 3*self.Omega))
        return output

    @property 
    def m7(self):
        return self._m7() * 0.00001

    def _m7(self):
        output = 7*(2*self.N - 3*self.lam - self.sigma + 2*self.xi)
        - (2*(4*self.delta - 5*self.sigma_st - 2*self.xi_st + 3*self.Omega))
        return output

    @property 
    def m8(self):
        return self._m8() * 0.00001

    def _m8(self):
        output = self.delta - 3*self.sigma_st + 3*self.xi_st - self.Omega
        return output
    # combinations at order 1/N_c^2
    @property 
    def mA(self):
        return self._mA() *0.0001

    def _mA(self):
        output = (self.sigma_st - self.sigma) - (self.xi_st - self.xi)
        return output 

    @property 
    def mB(self):
        return self._mB() *0.00001

    def _mB(self):
        output = (1/3 * (self.sigma + 2*self.sigma_st)) -  self.lam - (2/3*(self.delta - self.N))
        return output 
    
    @property 
    def mC(self):
        return self._mC() * 0.00001

    def _mC(self):
        output = (-1/4 * (2*self.N - 3*self.lam - self.sigma + 2*self.xi)) + (1/4*(self.delta - self.sigma_st - self.xi_st + self.Omega))
        return output 

    @property 
    def mD(self):
        return self._mD() * 0.00001

    def _mD(self):
        output = -1/2 * (self.delta - 3*self.sigma_st + 3*self.xi_st - self.Omega)
        return output

    # coefficients obtained by dividing out N_c order and scalar 
    @property
    def m1_c(self):
        return self._m1_c() * 0.1
    
    def _m1_c(self):
        c = self.m1 / 160*self.N_c
        return c
    
    @property
    def m2_c(self):
        return self._m2_c() * 0.1
    
    def _m2_c(self):
        c = self.m2 / (-120* 1/self.N_c)
        return c

    @property
    def m3_c(self):
        return self._m3_c() * 0.1
    
    def _m3_c(self):
        c = self.m3 / (20* np.sqrt(3)*self.eps)
        return c
    
    @property
    def m4_c(self):
        return self._m4_c() * 0.1

    def _m4_c(self):
        c = self.m4 / (-5* np.sqrt(3)* 1/self.N_c* self.eps)
        return c
    
    @property
    def m5_c(self):
        return self._m5_c() * 0.1

    def _m5_c(self):
        c = 0
        c = self.m5 / (30* np.sqrt(3)* 1/self.N_c**2 * self.eps)
        return c

    @property
    def m6_c(self):
        return self._m6_c() * 0.01

    def _m6_c(self):
        c = 0
        c = self.m6 / (126* 1/self.N_c * self.eps2)
        return c

    @property
    def m7_c(self):
        return self._m7_c() * 0.01

    def _m7_c(self):
        c = 0
        c =  self.m7 / (-63 * 1/self.N_c**2 * self.eps2)
        return c

    @property
    def m8_c(self):
        return self._m8_c() * 0.001

    def _m8_c(self):
        c = 0
        c = self.m8 / (9* np.sqrt(3) *  1/self.N_c**2 * self.eps3)
        return c

# scale invariant baryon mass combinations dim = mass

# TODO: PLOTS WITH ERROR BARS SUPERIMPOSED FOR /EPS #
    @property
    def R_1(self):
        return self._R_1()
    
    def _R_1(self):
        r = self.m1 * 160/240
        return r

    @property
    def R_2(self):
        return self._R_2()
    
    def _R_2(self):
        r = self.m2 * -80/120
        return r

    @property
    def R_3(self):
        return self._R_3() 
    
    def _R_3(self):
        r = self.m3 * 20*np.sqrt(3)/ 78
        return r

    @property
    def R_3_eps(self):
        return self._R_3() / self.eps

    @property
    def R_4(self):
        return self._R_4()
    
    def _R_4(self):
        r = self.m4* 5*np.sqrt(3)/ 60
        return r

    @property
    def R_4_eps(self):
        return self._R_4() / self.eps

    @property
    def R_5(self):
        return self._R_5()
    
    def _R_5(self):
        r = self.m5* 30*np.sqrt(3)/ 30
        return r

    @property
    def R_5_eps(self):
        return self._R_5() / self.eps

    @property
    def R_6(self):
        return self._R_6()
    
    def _R_6(self):
        r = self.m6* 126/ 336
        return r

    @property
    def R_6_eps2(self):
        return self._R_6() / self.eps2

    @property
    def R_7(self):
        return self._R_7()
    
    def _R_7(self):
        r = self.m7* -63/ 84
        return r

    @property
    def R_7_eps2(self):
        return self._R_7() / self.eps2

    @property
    def R_8(self):
        return self._R_8()
    
    def _R_8(self):
        r = self.m8* 9*np.sqrt(3)/ 8
        return r

    @property
    def R_8_eps3(self):
        return self._R_8() / self.eps3

    @property
    def R_A(self):
        return self._R_A()
    
    def _R_A(self):
        r = self.mA / 4
        return r

    @property
    def R_B(self):
        return self._R_B()
    
    def _R_B(self):
        r = self.mB / (10/3)
        return r

    @property
    def R_C(self):
        return self._R_C()
    
    def _R_C(self):
        r = self.mC / 3
        return r

    @property
    def R_D(self):
        return self._R_D()
    
    def _R_D(self):
        r = self.mD / 4
        return r

    

    

    

    

    

    

    

    


    

    

    

    

    
    
    

    

    

    



    

    


    










    
    
