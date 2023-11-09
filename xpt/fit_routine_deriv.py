# xi

def fitfcn_mass_deriv(self, p, data=None,xdata = None):
        xdata = self.prep_data(p, data, xdata)
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        
        output = 0 #llo
        output += self.fitfcn_lo_deriv(p,xdata)  
        output += self.fitfcn_nlo_xpt_deriv(p,xdata) 
        output += self.fitfcn_n2lo_ct_deriv(p,xdata)
        output += self.fitfcn_n2lo_xpt_deriv(p,xdata)
        if self.model_info['units'] == 'fpi':
            output * xdata['lam_chi']
        return output

def fitfcn_lo_deriv(self,p,xdata):
        '''derivative expansion to O(m_pi^2)'''
        output = 0
        if self.model_info['units'] == 'phys':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['m_{xi,0}'] * (p['d_{xi,a}'] * xdata['eps2_a'])
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['s_{xi}'] *xdata['eps_pi']* (
                            (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['eps_pi']**2)+
                            (2*xdata['lam_chi']*xdata['eps_pi'])
                    )
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{xi,0}']*(p['d_{xi,s}'] *  xdata['d_eps2_s'])
            
        elif self.model_info['units'] == 'fpi':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['d_{xi,a}'] * xdata['eps2_a']
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['s_{xi}'] *xdata['eps_pi']**2
                           
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['d_{xi,s}'] *  xdata['d_eps2_s']

        return output

def fitfcn_nlo_xpt_deriv(self, p, xdata):
        """Derivative expansion XPT expression at O(m_pi^3)"""

        if not self.model_info['xpt']:
            return 0

        def compute_phys_terms():
            term1 = -3/2 * np.pi * p['g_{xi,xi}']**2 * xdata['eps_pi'] * (
                (self.d_de_lam_chi_lam_chi(p, xdata) * xdata['lam_chi']) * xdata['eps_pi']**3 +
                (3 * xdata['lam_chi'] * xdata['eps_pi']**2)
            )
            term2 = p['g_{xi_st,xi}']**2 * xdata['eps_pi'] * (
                (xdata['lam_chi'] * self.d_de_lam_chi_lam_chi(p, xdata)) * naf.fcn_F(xdata['eps_pi'], xdata['eps_delta']) +
                xdata['lam_chi'] * naf.fcn_dF(xdata['eps_pi'], xdata['eps_delta'])
            )
            return term1 - term2

        def compute_fpi_terms():
            term3 = -9/4 * np.pi * p['g_{xi,xi}']**2 * xdata['eps_pi']**3
            term4 = 1/2 * p['g_{xi_st,xi}']**2 * xdata['eps_pi'] * naf.fcn_dF(xdata['eps_pi'], xdata['eps_delta'])
            return term3 - term4

        output = 0

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output += compute_phys_terms()
            elif self.model_info['units'] == 'fpi':
                output += compute_fpi_terms()

        return output

def fitfcn_n2lo_ct_deriv(self, p, xdata):
        ''''derivative expansion to O(m_pi^4) without terms coming from xpt expressions'''
        def compute_order_strange():
            term1 = p['d_{xi,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{xi,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{xi,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1 = p['d_{xi,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{xi,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light(fpi=None): 
            term1 =  p['b_{xi,4}']* xdata['eps_pi']
            term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 
            term3 =  4 * xdata['lam_chi'] * xdata['eps_pi']**3
            if fpi:

                termfpi = p['a_{xi,4}']* xdata['eps_pi']**4 
                termfpi2 = 2 * p['b_{xi,4}']* xdata['eps_pi']**4
                termfpi3 = p['s_{xi}']*(1/4*xdata['eps_pi']**4 - 1/4* p['l3_bar']* xdata['eps_pi']**4)
                return termfpi + termfpi2 + termfpi3
            return term1*(term2+term3)

        def compute_order_chiral(fpi=None):
            if fpi:
                if self.model_info['fv']:
                    return p['a_{xi,4}']* (2*xdata['eps_pi']**4*fv.fcn_I_m(xdata['eps_pi']**2,xdata['L'],xdata['lam_chi'],10))
                else:
                    return p['a_{xi,4}']* (2*xdata['eps_pi']**4*np.log(xdata['eps_pi']**2))
            else:

                term1 =  p['a_{xi,4}']* xdata['eps_pi']
                if self.model_info['fv']:
                    term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 * fv.fcn_I_m(xdata['eps_pi']**2,xdata['L'],xdata['lam_chi'],10)
                    term3 = 4 * xdata['lam_chi'] * xdata['eps_pi']**3 * fv.fcn_I_m(xdata['eps_pi']**2,xdata['L'],xdata['lam_chi'],10)

                else:
                    term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) 
                    term3 = 4 * xdata['lam_chi'] * xdata['eps_pi']**3 * np.log(xdata['eps_pi']**2)
                term4 = 2 * xdata['lam_chi'] * xdata['eps_pi']**3 
            return term1*(term2+term3+term4)
        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{xi,0}'] * compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{xi,0}'] * compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light()

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral()

        elif self.model_info['units'] == 'fpi':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += compute_order_strange()

            if self.model_info['order_disc'] in ['n2lo']:
                output += compute_order_disc()

            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light(fpi=True)

            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral(fpi=True)

        return output

def fitfcn_n2lo_xpt_deriv(self, p, xdata):
        '''xpt expression for xi mass derivative expansion at O(m_pi^4)'''

        def base_term():
            """Computes the base term which is common for both 'phys' and 'fpi' units."""
            return 3/2 * p['g_{xi_st,xi}']** 2 * (p['s_{xi}'] - p['s_{xi,bar}']) * xdata['eps_pi']  

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            term1 = (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi']) * xdata['eps_pi']** 2 * naf.fcn_J(xdata['eps_pi'], xdata['eps_delta'])
            term2 = 2* xdata['lam_chi'] *xdata['eps_pi'] *  naf.fcn_J(xdata['eps_pi'], xdata['eps_delta'])
            term3 = xdata['lam_chi'] * xdata['eps_pi']**2 * naf.fcn_dJ(xdata['eps_pi'], xdata['eps_delta'])
            return  base_term() * (term1+term2+term3)

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return base_term()

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output


# xi st

def fitfcn_mass_deriv(self, p, data=None,xdata = None):
        xdata = self.prep_data(p, data, xdata)
        if data is not None:
            for key in data.keys():
                p[key] = data[key]
        output = 0 #llo
        output += self.fitfcn_lo_deriv(p,xdata)  
        output += self.fitfcn_nlo_xpt_deriv(p,xdata) 
        output += self.fitfcn_n2lo_ct_deriv(p,xdata)
        output += self.fitfcn_n2lo_xpt_deriv(p,xdata)
        if self.model_info['units'] == 'fpi':
            output = output * xdata['lam_chi']
        return output


def fitfcn_lo_deriv(self,p,xdata):
        '''derivative expansion to O(m_pi^2)'''
        output = 0
        if self.model_info['units'] == 'phys':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['m_{xi_st,0}'] * (p['d_{xi_st,a}'] * xdata['eps2_a'])
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['s_{xi,bar}'] *xdata['eps_pi']* (
                            (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['eps_pi']**2)+
                            (2*xdata['lam_chi']*xdata['eps_pi'])
                    )
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['m_{xi_st,0}']*(p['d_{xi_st,s}'] *  xdata['d_eps2_s'])
            
        elif self.model_info['units'] == 'fpi':
            if self.model_info['order_disc'] in ['lo', 'nlo', 'n2lo']:
                output += p['d_{xi_st,a}'] * xdata['eps2_a']
        
            if self.model_info['order_light'] in ['lo', 'nlo', 'n2lo']:
                    output+= p['s_{xi,bar}'] *xdata['eps_pi']**2
                           
            if self.model_info['order_strange'] in ['lo', 'nlo', 'n2lo']:
                output+= p['d_{xi_st,s}'] *  xdata['d_eps2_s']

        return output

def fitfcn_nlo_xpt_deriv(self, p, xdata):
        """Derivative expansion XPT expression at O(m_pi^3)"""

        if not self.model_info['xpt']:
            return 0

        def compute_phys_terms():
            term1 = -5/6 * np.pi * p['g_{xi_st,xi_st}']**2 * xdata['eps_pi'] * (
                (self.d_de_lam_chi_lam_chi(p, xdata) * xdata['lam_chi']) * xdata['eps_pi']**3 +
                (3 * xdata['lam_chi'] * xdata['eps_pi']**2)
            )
            term2 = p['g_{xi_st,xi}']**2 * xdata['eps_pi'] * (
                (xdata['lam_chi'] * self.d_de_lam_chi_lam_chi(p, xdata)) * naf.fcn_F(xdata['eps_pi'], -xdata['eps_delta']) +xdata['lam_chi'] * naf.fcn_dF(xdata['eps_pi'], -xdata['eps_delta'])
            )
            return term1 - term2

        def compute_fpi_terms():
            term3 = -5/4* np.pi * p['g_{xi_st,xi_st}']**2 * xdata['eps_pi']**3
            term4 =  p['g_{xi_st,xi}']**2 * xdata['eps_pi'] * naf.fcn_dF(xdata['eps_pi'], -xdata['eps_delta'])
            return term3 - term4

        output = 0
        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output += compute_phys_terms()
            elif self.model_info['units'] == 'fpi':
                output += compute_fpi_terms()

        return output


def fitfcn_n2lo_ct_deriv(self, p, xdata):
        ''''derivative expansion to O(m_pi^4) without terms coming from xpt expressions'''
        def compute_order_strange():
            term1 = p['d_{xi_st,as}'] * xdata['eps2_a'] * xdata['d_eps2_s']
            term2 = p['d_{xi_st,ls}'] * xdata['d_eps2_s'] * xdata['eps_pi']**2
            term3 = p['d_{xi_st,ss}'] * xdata['d_eps2_s']**2

            return term1 + term2 + term3

        def compute_order_disc():
            term1 = p['d_{xi_st,al}'] * xdata['eps2_a'] * xdata['eps_pi']**2
            term2 = p['d_{xi_st,aa}'] * xdata['eps2_a']**2

            return term1 + term2

        def compute_order_light(fpi=None): 
            term1 =  p['b_{xi_st,4}']* xdata['eps_pi']
            term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 
            term3 =  4 * xdata['lam_chi'] * xdata['eps_pi']**3
            if fpi:
                termfpi = p['a_{xi_st,4}']* xdata['eps_pi']**4 
                termfpi2 = 2 * p['b_{xi_st,4}']* xdata['eps_pi']**4
                termfpi3 = p['s_{xi,bar}']*(1/4*xdata['eps_pi']**4 - 1/4* p['l3_bar']* xdata['eps_pi']**4)
                return termfpi + termfpi2 + termfpi3
            return term1*(term2+term3)

        def compute_order_chiral(fpi=None):
            term1 =  p['a_{xi_st,4}']* xdata['eps_pi']
            if self.model_info['fv']:
                term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 * fv.fcn_I_m(xdata['eps_pi']**2,xdata['L'],xdata['lam_chi'],10) 
                term3 = 4 * xdata['lam_chi'] * xdata['eps_pi']**3 * fv.fcn_I_m(xdata['eps_pi']**2,xdata['L'],xdata['lam_chi'],10) 
            else:
                term2 =  (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi'])*xdata['eps_pi']**4 * np.log(xdata['eps_pi']**2) 
                term3 = 4 * xdata['lam_chi'] * xdata['eps_pi']**3 * np.log(xdata['eps_pi']**2)
            term4 = 2 * xdata['lam_chi'] * xdata['eps_pi']**3 

            if fpi:
                if self.model_info['fv']:
                    return p['a_{xi_st,4}']* (2*xdata['eps_pi']**4*fv.fcn_I_m(xdata['eps_pi']**2,xdata['L'],xdata['lam_chi'],10))
                else:
                    return p['a_{xi_st,4}']* (2*xdata['eps_pi']**4*np.log(xdata['eps_pi']**2))
            else:
                return term1*(term2+term3+term4)
        output = 0

        if self.model_info['units'] == 'phys':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += p['m_{xi_st,0}'] * compute_order_strange()
            if self.model_info['order_disc'] in ['n2lo']:
                output += p['m_{xi_st,0}'] * compute_order_disc()
            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light()
            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral()

        elif self.model_info['units'] == 'fpi':  # lam_chi dependence ON 
            if self.model_info['order_strange'] in ['n2lo']:
                output += compute_order_strange()
            if self.model_info['order_disc'] in ['n2lo']:
                output += compute_order_disc()
            if self.model_info['order_light'] in ['n2lo']:
                output += compute_order_light(fpi=True)
            if self.model_info['order_chiral'] in ['n2lo']:
                output += compute_order_chiral(fpi=True)

        return output

def fitfcn_n2lo_xpt_deriv(self, p, xdata):
        '''xpt expression for xi mass derivative expansion at O(m_pi^4)'''

        def base_term():
            """Computes the base term which is common for both 'phys' and 'fpi' units."""
            return 3/4 * p['g_{xi_st,xi}']** 2 * (p['s_{xi,bar}']-p['s_{xi}']) * xdata['eps_pi']  

        def compute_phys_output():
            """Computes the output for 'phys' units, considering lam_chi dependence."""
            term1 = (self.d_de_lam_chi_lam_chi(p,xdata)*xdata['lam_chi']) * xdata['eps_pi']** 2 * naf.fcn_J(xdata['eps_pi'], -xdata['eps_delta'])
            term2 = 2* xdata['lam_chi'] *xdata['eps_pi'] *  naf.fcn_J(xdata['eps_pi'], -xdata['eps_delta'])
            term3 = xdata['lam_chi'] * xdata['eps_pi']**2 * naf.fcn_dJ(xdata['eps_pi'], -xdata['eps_delta'])
            return  base_term() * (term1+term2+term3)

        def compute_fpi_output():
            """Computes the output for 'fpi' units, not considering lam_chi dependence."""
            return base_term()

        if self.model_info['xpt']:
            if self.model_info['units'] == 'phys':
                output = compute_phys_output()
            elif self.model_info['units'] == 'fpi':
                output = compute_fpi_output()
        else:
            return 0

        return output
