import numpy
import numericalunits as un

def empty_model():
    return None

config = {
    'v_0'  : 230  * un.km / un.s,
    'v_esc': 544  * un.km / un.s,
    'rho_0': 0.3  * un.GeV / (un.c0**2),
    'm_dm' : 100  * un.GeV / (un.c0**2),
    'k'    : 1
}

# TO Do also have @export?
# @export
class dm_halo:
    """Dark matter halo model. Takes astrophysical parameters and returns the elastic recoil spectrum"""
    
    def __init__(self):
        self.v_0 = None
        self.v_e = None
        self.v_lsr = None
        self.model = None

    def model(self, name):
        # TO DO
        # Add an assertion error here to check if there is a model
        assert True
        model = empty_model
        return model()

