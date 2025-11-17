import numbers

import numpy as np
from .base_lot import LoT

from rdkit.Chem import AllChem

class rdkit(LoT):
    energy_units = "kcal/mol"
    distance_units = "Angstrom"

    def __init__(self,
                 *,
                 atoms,
                 bonds,
                 force_field="mmff",
                 **kwargs
                 ):
        self.mol = self.setup_rdmol([a.symbol for a in atoms], bonds.edges())
        self._conf = None
        self.force_field = force_field
        super().__init__(atoms=atoms, bonds=bonds, **kwargs)

    @classmethod
    def resolve_bond_type(cls, t):
        if abs(t - 1.5) < 1e-2:
            t = AllChem.BondType.names["AROMATIC"]
        elif abs(t - 2.5) < 1e-2:
            t = AllChem.BondType.names["TWOANDAHALF"]
        elif abs(t - 3.5) < 1e-2:
            t = AllChem.BondType.names["TWOANDAHALF"]
        else:
            t = AllChem.BondType.values[int(t)]
        return t

    @classmethod
    def sanitize_mol(self, mol, sanitize_ops=None):
        from rdkit.Chem import rdmolops
        if sanitize_ops is None:
            sanitize_ops = (
                    rdmolops.SANITIZE_ALL
                    ^ rdmolops.SANITIZE_PROPERTIES
                    # ^rdmolops.SANITIZE_ADJUSTHS
                    # ^rdmolops.SANITIZE_CLEANUP
                    ^ rdmolops.SANITIZE_CLEANUP_ORGANOMETALLICS
            )
        AllChem.SanitizeMol(mol, sanitize_ops)
        return mol

    @classmethod
    def setup_rdmol(cls, atoms, bonds):
        mol = AllChem.EditableMol(AllChem.Mol())
        mol.BeginBatchEdit()
        for a in atoms:
            a = AllChem.Atom(a)
            mol.AddAtom(a)
        for b in bonds:
            if len(b) == 2:
                i, j = b
                t = 1
            else:
                i, j, t = b
            if isinstance(t, numbers.Number):
                t = cls.resolve_bond_type(t)
            else:
                t = AllChem.BondType.names[t]
            mol.AddBond(int(i), int(j), t)
        mol.CommitBatchEdit()

        mol = mol.GetMol()
        mol = AllChem.AddHs(mol, explicitOnly=True)
        mol = cls.sanitize_mol(mol)

        return mol

    @classmethod
    def get_force_field_type(cls, ff_type):
        if isinstance(ff_type, str):
            if ff_type == 'mmff':
                ff_type = (AllChem.MMFFGetMoleculeForceField, AllChem.MMFFGetMoleculeProperties)
            elif ff_type == 'uff':
                ff_type = (AllChem.UFFGetMoleculeForceField, None)
            else:
                raise ValueError(f"can't get RDKit force field type from '{ff_type}")

        return ff_type

    def get_force_field(self, coords, force_field_type=None, **extra_props):
        if self._conf is not None:
            self.mol.RemoveConformer(0)
        self._conf = AllChem.Conformer(len(self.atoms))
        self._conf.SetPositions(coords)
        self._conf.SetId(0)
        self.mol.AddConformer(self._conf)

        if force_field_type is None:
            force_field_type = self.force_field
        force_field_type = self.get_force_field_type(force_field_type)
        if isinstance(force_field_type, (list, tuple)):
            force_field_type, prop_gen = force_field_type
        else:
            prop_gen = None

        if prop_gen is not None:
            props = prop_gen(self.mol)
        else:
            props = None

        if props is not None:
            return force_field_type(self.mol, props, confId=0, **extra_props)
        else:
            return force_field_type(self.mol, confId=0, **extra_props)

    def run_raw(self, coords, mult, ad_idx, *, runtypes):
        ff = self.get_force_field(coords)
        res = {}
        if 'gradient' in runtypes:
            res['gradient'] = np.array(ff.CalcGrad())
        if 'energy' in runtypes:
            res['energy'] = ff.CalcEnergy()
        return res