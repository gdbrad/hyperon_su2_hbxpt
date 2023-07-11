import gvar as gv

# currently not implemented prior filtering routines 

def filter_prior_keys(prior, model_info, particles):
    # Extract relevant keys from the particles
    relevant_keys = set()
    for particle in particles:
        relevant_keys.add(f"m_{{{particle},0}}")

        if particle != "proton" and particle != "delta":
            relevant_keys.add(f"s_{{{particle}}}")
        else:
            relevant_keys.add(f"s_{{{particle},bar}}")

        for p in range(1, 5):
            relevant_keys.add(f"a_{{{particle},{p}}}")
            relevant_keys.add(f"b_{{{particle},{p}}}")

        for other_particle in particles:
            relevant_keys.add(f"g_{{{particle},{other_particle}}}")

    filtered_prior = {k: v for k, v in prior.items() if k in relevant_keys}
    return filtered_prior

def filter_relevant_prior_keys(model_info, prior):
    orders = ['llo', 'lo', 'nlo', 'n2lo']
    particles = model_info['particles']
    order_chiral = model_info['order_chiral']
    order_disc = model_info['order_disc']
    order_strange = model_info['order_strange']
    order_light = model_info['order_light']

    highest_order = max([order_chiral, order_disc, order_strange, order_light])

    relevant_prior_keys = []
    for order in orders:
        if orders.index(order) > orders.index(highest_order):
            break
        relevant_prior_keys.extend(get_filtered_prior_keys(particles, order, prior))

    # Filter priors based on the highest order and particles
    filtered_prior = {k: v for order in orders for k, v in prior[order].items()
                      if k in relevant_prior_keys}

    return filtered_prior

def recalibrate_prior(prior, data,fit_result, scale_factor):
    excluded = {
        'm_k', 'm_pi', 'lam_chi', 'eps2_a','m_xi','m_xi_st'
    }
    new_prior = prior.copy()
    for key in fit_result.p:
        if key not in excluded:
        # if key in fit_result.p:
            new_prior[key] = fit_result.p[key], scale_factor * fit_result.psdev[key]
        else:
            new_prior[key] = data[key]
    return new_prior

def get_prior(units=None): 
    if units=='mev':
        gs_baryons={
    
        # not-even leading order 
        'm_{xi,0}' : gv.gvar(1000,400), # MeV
        'm_{xi_st,0}' : gv.gvar(1500,400), # MeV
        'm_{lambda,0}' : gv.gvar(1050,100), 
        'm_{sigma,0}' : gv.gvar(1300,100), 
        'm_{sigma_st,0}' : gv.gvar(1450,100),
        'm_{omega,0}' : gv.gvar(1650,50),
        }
    elif units =='fpi': # ensure these have a gap between means or else a float division error will occur
        gs_baryons = {
        # not-even leading order 
        'm_{xi,0}' : gv.gvar(1,1), 
        'm_{xi_st,0}' :  gv.gvar(1.1,1), 
        'm_{lambda,0}' :  gv.gvar(1.2,1), 
        'm_{sigma,0}' :  gv.gvar(1.3,1), 
        'm_{sigma_st,0}' :  gv.gvar(1.4,1),
        'm_{omega,0}' :  gv.gvar(1,1),
        }

    else:
        raise ValueError(f"Invalid units: {units}")
    prior = {
        **gs_baryons,
        # lo
        's_{xi}' : gv.gvar(0, 2),
        's_{xi,bar}' : gv.gvar(0, 2),
        's_{lambda}' : gv.gvar(0, 5),
        's_{sigma,bar}' : gv.gvar(0, 5),
        's_{sigma}' : gv.gvar(0, 5),
        'l3_bar' : gv.gvar(3.53,2.6),

        
        'l4_bar':gv.gvar(4.02,4.02),

        'F0'    : gv.gvar(85,30),
        'c2_F' : gv.gvar(0,20),
        'c1_F' : gv.gvar(0,20),

        'b_{xi,2}': gv.gvar(2,2),
        'c0': gv.gvar(2,2),
   
        # nlo
        'g_{xi,xi}' : gv.gvar(0.3, 4),
        'g_{xi_st,xi}' : gv.gvar(0.7, 3),
        'g_{xi_st,xi_st}' : gv.gvar(-.75, 2),
        'g_{lambda,sigma}' : gv.gvar(0, 5),
        'g_{lambda,sigma_st}' :gv.gvar(0, 5),
        'g_{sigma,sigma}' : gv.gvar(0, 5),
        'g_{sigma_st,sigma}' : gv.gvar(0, 5),
        'g_{sigma_st,sigma_st}': gv.gvar(0, 5),

        # n2lo
        'a_{xi,4}' : gv.gvar(0, 2),
        'b_{xi,4}' : gv.gvar(0, 2),
        'a_{xi_st,4}' : gv.gvar(0, 2),
        'b_{xi_st,4}' : gv.gvar(0, 5),
        'a_{sigma,4}' : gv.gvar(0, 5),
        'b_{sigma,4}' : gv.gvar(0, 5),
        'a_{sigma_st,4}' : gv.gvar(0, 5),
        'b_{sigma_st,4}' : gv.gvar(0, 5),
        'a_{lambda,4}' : gv.gvar(0, 5),
        'b_{lambda,4}' : gv.gvar(0, 5),

        # note: no lo terms for taylor 
        # latt/strange nlo
        'd_{xi,a}' : gv.gvar(-2,2),
        'd_{xi,s}' : gv.gvar(0,5),
        'd_{xi_st,a}' : gv.gvar(0,2),
        'd_{xi_st,s}' : gv.gvar(0,5), 
        'd_{lambda,s}' : gv.gvar(0,5),
        'd_{lambda,a}' : gv.gvar(0,5),
        'd_{sigma_st,a}' : gv.gvar(0,5), 
        'd_{sigma_st,s}' : gv.gvar(0,5),
        'd_{sigma,s}' : gv.gvar(0,5), 
        'd_{sigma,a}' : gv.gvar(0,5),

        # disc n2lo
        'd_{xi,aa}' : gv.gvar(2,4),
        'd_{xi,al}' : gv.gvar(0,5),
        'd_{xi,as}' : gv.gvar(0,5),
        'd_{xi,ls}' : gv.gvar(0,5),
        'd_{xi,ss}' : gv.gvar(0,5),

        'd_{xi_st,aa}' : gv.gvar(0,5),
        'd_{xi_st,al}' : gv.gvar(0,5), 
        'd_{xi_st,as}' : gv.gvar(0,7),
        'd_{xi_st,ls}' : gv.gvar(0,5), 
        'd_{xi_st,ss}' : gv.gvar(0,5),

        'd_{lambda,aa}' : gv.gvar(0,5),
        'd_{lambda,al}' : gv.gvar(0,5),
        'd_{lambda,as}' : gv.gvar(0,5),
        'd_{lambda,ls}' : gv.gvar(0,5),
        'd_{lambda,ss}' : gv.gvar(0,5),

        'd_{sigma,aa}' : gv.gvar(0,5),
        'd_{sigma,al}' : gv.gvar(0,5),
        'd_{sigma,as}' : gv.gvar(0,5),
        'd_{sigma,ls}' : gv.gvar(0,5),
        'd_{sigma,ss}' : gv.gvar(0,5),

        'd_{sigma_st,aa}' : gv.gvar(0,5),
        'd_{sigma_st,al}' : gv.gvar(0,5), 
        'd_{sigma_st,as}' : gv.gvar(0,5),
        'd_{sigma_st,ls}' : gv.gvar(0,5), 
        'd_{sigma_st,ss}' : gv.gvar(0,5),}
    return prior

