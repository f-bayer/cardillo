import numpy as np




def Motor(Transmission):
    class _Motor(Transmission):
        def __init__(self, tau, **kwargs):
            if not callable(tau):
                self.tau = lambda t: tau #TODO: I don't like this to much. Maybe tau should be just a System property. Otherwise the implementation self.la_tau = self.tau may be tempting.
            else:
                self.tau = tau
            self.nla_tau = 1
            self.ntau = 1
            super().__init__(**kwargs)

            self.W_tau = self.W_l

        def la_tau(self, t, q, u, tau):
            return tau

    return _Motor
