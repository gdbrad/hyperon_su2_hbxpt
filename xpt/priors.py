import gvar as gv

prior = {
    
        # not-even leading order 
        'm_{xi,0}' : gv.gvar(1200,100), # MeV
        'm_{xi_st,0}' : gv.gvar(1300,100), # MeV
        'm_{lambda,0}' : gv.gvar(1000,10000), 
        'm_{sigma,0}' : gv.gvar(1200,10000), 
        'm_{sigma_st,0}' : gv.gvar(1400,10000),
        'm_{omega,0}' : gv.gvar(1650,10000),

        # lo
        's_{xi}' : gv.gvar(0, 10),
        's_{xi,bar}' : gv.gvar(0, 10),
        's_{lambda}' : gv.gvar(0, 5),
        's_{sigma,bar}' : gv.gvar(0, 5),
        's_{sigma}' : gv.gvar(0, 5),
        's_{omega,bar}' : gv.gvar(0, 5),
   
        # nlo
        'g_{xi,xi}' : gv.gvar(0.3, 3),
        'g_{xi_st,xi}' : gv.gvar(0.7, 3),
        'g_{xi_st,xi_st}' : gv.gvar(-.5, 3),
        'g_{lambda,sigma}' : gv.gvar(0, 5),
        'g_{lambda,sigma_st}' :gv.gvar(0, 5),
        'g_{sigma,sigma}' : gv.gvar(0, 5),
        'g_{sigma_st,sigma}' : gv.gvar(0, 5),
        'g_{sigma_st,sigma_st}': gv.gvar(0, 5),

        # n2lo
        'a_{xi,4}' : gv.gvar(0, 5),
        'a_{xi_st,4}' : gv.gvar(0, 5),
        'b_{xi,4}' : gv.gvar(0, 5),
        'b_{xi_st,4}' : gv.gvar(0, 5),
        'a_{sigma,4}' : gv.gvar(0, 5),
        'a_{sigma_st,4}' : gv.gvar(0, 5),
        'b_{sigma,4}' : gv.gvar(0, 5),
        'b_{sigma_st,4}' : gv.gvar(0, 5),
        'a_{lambda,4}' : gv.gvar(0, 5),
        'b_{lambda,4}' : gv.gvar(0, 5),
        'a_{omega,4}' : gv.gvar(0, 5),
        'b_{omega,4}' : gv.gvar(0, 5),
        'a_{omega,6}' : gv.gvar(0, 5),
        'b_{omega,6}' : gv.gvar(0, 5),


        # note: no lo terms for taylor 
        # latt/strange nlo
        'd_{xi,a}' : gv.gvar(0,10),
        'd_{xi_st,a}' : gv.gvar(0,10),
        'd_{xi,s}' : gv.gvar(0,10),
        'd_{xi_st,s}' : gv.gvar(0,10),  
        'd_{lambda,s}' : gv.gvar(0,10),
        'd_{lambda,a}' : gv.gvar(0,10),
        'd_{sigma_st,a}' : gv.gvar(0,10), 
        'd_{sigma_st,s}' : gv.gvar(0,10),
        'd_{sigma,s}' : gv.gvar(0,10), 
        'd_{sigma,a}' : gv.gvar(0,10),
        'd_{omega,s}' : gv.gvar(0,10), 
        'd_{omega,a}' : gv.gvar(0,10),

        # disc n2lo
        'd_{xi,aa}' : gv.gvar(0,10),
        'd_{xi,al}' : gv.gvar(0,10),
        'd_{xi,as}' : gv.gvar(0,10),
        'd_{xi,ls}' : gv.gvar(0,10),
        'd_{xi,ss}' : gv.gvar(0,10),

        'd_{xi_st,aa}' : gv.gvar(0,10),
        'd_{xi_st,al}' : gv.gvar(0,10), 
        'd_{xi_st,as}' : gv.gvar(0,10),
        'd_{xi_st,ls}' : gv.gvar(0,10), 
        'd_{xi_st,ss}' : gv.gvar(0,10),

        'd_{lambda,aa}' : gv.gvar(0,10),
        'd_{lambda,al}' : gv.gvar(0,10),
        'd_{lambda,as}' : gv.gvar(0,10),
        'd_{lambda,ls}' : gv.gvar(0,10),
        'd_{lambda,ss}' : gv.gvar(0,10),

        'd_{sigma,aa}' : gv.gvar(0,10),
        'd_{sigma,al}' : gv.gvar(0,10),
        'd_{sigma,as}' : gv.gvar(0,10),
        'd_{sigma,ls}' : gv.gvar(0,10),
        'd_{sigma,ss}' : gv.gvar(0,10),

        'd_{sigma_st,aa}' : gv.gvar(0,10),
        'd_{sigma_st,al}' : gv.gvar(0,10), 
        'd_{sigma_st,as}' : gv.gvar(0,10),
        'd_{sigma_st,ls}' : gv.gvar(0,10), 
        'd_{sigma_st,ss}' : gv.gvar(0,10),

        'd_{omega,aa}' : gv.gvar(0,10),
        'd_{omega,al}' : gv.gvar(0,10), 
        'd_{omega,as}' : gv.gvar(0,10),
        'd_{omega,ls}' : gv.gvar(0,10), 
        'd_{omega,ss}' : gv.gvar(0,10)
    

}