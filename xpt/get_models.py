class Model:
    def __init__(self, model_info):
        self.model_info = model_info

    @classmethod
    def from_name(cls, name):
        model_info = cls.get_model_info_from_name(name)
        return cls(model_info)

    @staticmethod
    def get_model_info_from_name(name):
        model_info = {'name': name}
        orders = {'nlo': 'nlo', 'n2lo': 'n2lo', 'n3lo': 'n3lo', 'lo': 'lo','llo':'llo'}
        types = ['s', 'd', 'x', 'l']
        particles = ['xi','xi_st','sigma','sigma_st','lam']

        for t in types:
            for k, v in orders.items():
                if f'{t}_{k}' in name:
                    model_info[f'order_{t}'] = v
                    break
            else:
                model_info[f'order_{t}'] = 'lo'

        for p in particles:
            if f'{p}' in name:
                model_info['particles'] = p

        # Definition of eps2_a
        if '_w0orig' in name:
            model_info['eps2a_defn'] = 'w0_orig'
        elif ':w0imp' in name:
            model_info['eps2a_defn'] = 'w0_imp'
        elif '_t0orig' in name:
            model_info['eps2a_defn'] = 't0_orig'
        elif '_t0impr' in name:
            model_info['eps2a_defn'] = 't0_imp'
        elif '_variable' in name:
            model_info['eps2a_defn'] = 'variable'
        else:
            model_info['eps2a_defn'] = None

        model_info['fv'] = bool(':fv' in name)
        model_info['xpt'] = bool(':xpt' in name)

        # You can add more logic here to extract other information

        return model_info

models = {
    'xi_system': [
        {
            'name': 'xi:xi_st:s_n2lo:d_n2lo:x_n2lo:l_n2lo:w0imp:fv:xpt',
            'particles': ['xi', 'xi_st'],
            'eps2a_defn': 'w0_imp',
            'order_chiral': 'n2lo',
            'order_disc': 'n2lo',
            'order_strange': 'n2lo',
            'order_light': 'n2lo',
            'xpt': True,
            'fv': True
        },

        {
            'name': 's_n2lo:d_n2lo:x_n2lo:l_nlo',
            'particles': ['xi', 'xi_st'],
            'eps2a_defn': 'w0_imp',
            'order_chiral': 'n2lo',
            'order_disc': 'n2lo',
            'order_strange': 'n2lo',
            'order_light': 'nlo',
            'xpt': True,
            'fv': True
        },

        {
            'name': 's_n2lo:d_n2lo:x_nlo:l_nlo',
            'particles': ['xi', 'xi_st'],
            'eps2a_defn': 'w0_imp',
            'order_chiral': 'nlo',
            'order_disc': 'n2lo',
            'order_strange': 'n2lo',
            'order_light': 'nlo',
            'xpt': True,
            'fv': True
        },

        {
            'name': 's_n2lo:d_nlo:x_nlo:l_nlo',
            'particles': ['xi', 'xi_st'],
            'eps2a_defn': 'w0_imp',
            'order_chiral': 'nlo',
            'order_disc': 'nlo',
            'order_strange': 'n2lo',
            'order_light': 'nlo',
            'xpt': True,
            'fv': True
        },

        {
            'name': 's_nlo:d_nlo:x_nlo:l_nlo',
            'particles': ['xi', 'xi_st'],
            'eps2a_defn': 'w0_imp',
            'order_chiral': 'nlo',
            'order_disc': 'nlo',
            'order_strange': 'nlo',
            'order_light': 'nlo',
            'xpt': True,
            'fv': True
        },
        # ...
    ],
    # ...
}

        # ...
    ],
    # ...
}
