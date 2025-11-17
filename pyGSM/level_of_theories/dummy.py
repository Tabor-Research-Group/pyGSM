
import numpy as np
from .base_lot import LoT

class dummy(LoT):
    energy_units = "kcal/mol"
    distance_units = "Angstrom"

    def run_raw(self, coords, mult, ad_idx, *, runtypes):
        shape = coords.shape
        res = {}
        if 'gradient' in runtypes:
            res['gradient'] = np.zeros(shape)
        if 'energy' in runtypes:
            res['energy'] = 0
        return res