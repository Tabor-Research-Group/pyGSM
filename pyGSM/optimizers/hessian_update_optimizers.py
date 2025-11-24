
import numpy as np
from .. import coordinate_systems as coord_ops
from .base_optimizer import base_optimizer, force_positive_definite

__all__ = [
    "hessian_update_optimizer"
]

class hessian_update_optimizer(base_optimizer):

    def __init__(self,
                 *,
                 update_hess_in_bg=True,
                 HESS_TANG_TOL_TS=0.35,
                 **base_opts):
        super().__init__(**base_opts)


        # Hessian
        self.update_hess_in_bg = update_hess_in_bg

        # Hessian
        self.maxol_good = True
        self.gtse = 100.
        self.HESS_TANG_TOL_TS = HESS_TANG_TOL_TS


        self.Hint = None
        self.dx = 0.
        self.dg = 0.

    def get_state_dict(self):
        return dict(
            super().get_state_dict(),
            update_hess_in_bg=self.update_hess_in_bg,
            HESS_TANG_TOL_TS=self.HESS_TANG_TOL_TS
        )

    def update_Hessian(self, molecule, mode='BFGS'):
        '''
        mode 1 is BFGS, mode 2 is BOFILL
        '''
        assert mode == 'BFGS' or mode == 'BOFILL', "no update implemented with that mode"
        # do this even if mode==BOFILL
        change = self.update_bfgs(molecule)

        if coord_ops.is_dlc(molecule.coord_obj):
            # This really feels like it's in the wrong place
            molecule.update_Primitive_Hessian(change=change)
            self.logger.log_print(
                ["change", "{change}", "updated primitive internals Hessian", "{new_hess}"],
                change=change,
                new_hess=molecule.Primitive_Hessian,
                log_level=self.logger.LogLevel.Debug
            )
            if mode == 'BFGS':
                molecule.form_Hessian_in_basis()
            elif mode == 'BOFILL':
                change = self.update_bofill(molecule)
                molecule.update_Hessian(change)
        # else:
        #    self.Hessian += change
        molecule.newHess -= 1

        return change

    def update_bfgs(self, molecule):
        if not coord_ops.is_cartesian(molecule.coord_obj):
            return self.update_bfgsp(molecule)
        else:
            raise NotImplementedError("updating not implemented for Cartesian coordinate system")

    def update_bofill(self, molecule):
        self.logger.log_print(" in update bofill")

        # return self.update_TS_BFGS(molecule)

        G = np.copy(molecule.Hessian)  # nicd,nicd
        Gdx = np.dot(G, self.dx)  # (nicd,nicd)(nicd,1) = (nicd,1)
        dgmGdx = self.dg - Gdx  # (nicd,1)

        # MS
        dgmGdxtdx = np.dot(dgmGdx.T, self.dx)  # (1,nicd)(nicd,1)
        Gms = np.outer(dgmGdx, dgmGdx)/dgmGdxtdx

        # PSB
        dxdx = np.outer(self.dx, self.dx)
        dxtdx = np.dot(self.dx.T, self.dx)
        dxtdg = np.dot(self.dx.T, self.dg)
        dxtGdx = np.dot(self.dx.T, Gdx)
        dxtdx2 = dxtdx*dxtdx
        dxtdgmdxtGdx = dxtdg - dxtGdx
        Gpsb = np.outer(dgmGdx, self.dx)/dxtdx + np.outer(self.dx, dgmGdx)/dxtdx - dxtdgmdxtGdx*dxdx/dxtdx2

        # Bofill mixing
        dxtE = np.dot(self.dx.T, dgmGdx)  # (1,nicd)(nicd,1)
        EtE = np.dot(dgmGdx.T, dgmGdx)  # E is dgmGdx
        phi = 1. - dxtE*dxtE/(dxtdx*EtE)

        change = (1.-phi)*Gms + phi*Gpsb
        return change


    def update_bfgsp(self, molecule):
        self.logger.log_print(
            [
                "In update bfgsp",
                'dx_prim {dx_prim} ',
                'dg_prim {dg_prim}'
            ],
            dx_prim=self.dx_prim.T,
            dg_prim=self.dg_prim.T,
            log_level=self.logger.LogLevel.Debug
        )

        Hdx = np.dot(molecule.Primitive_Hessian, self.dx_prim)
        dxHdx = np.dot(np.transpose(self.dx_prim), Hdx)
        dgdg = np.outer(self.dg_prim, self.dg_prim)
        dgtdx = np.dot(np.transpose(self.dg_prim), self.dx_prim)
        change = np.zeros_like(molecule.Primitive_Hessian)

        self.logger.log_print(
            [
                "Hdx",
                "{Hdx}",
                "dgtdx: {dgtdx:1.8f} dxHdx: {dxHdx:1.8f}",
                "dgdg",
                "{dgdg}",
            ],
            Hdx=Hdx.T,
            dgtdx=dgtdx[0, 0],
            dxHdx=dxHdx[0, 0],
            dgdg=dgdg,
            log_level=self.logger.LogLevel.Debug
        )

        if dgtdx > 0.:
            if dgtdx < 0.001:
                dgtdx = 0.001
            change += dgdg/dgtdx
        if dxHdx > 0.:
            if dxHdx < 0.001:
                dxHdx = 0.001
            change -= np.outer(Hdx, Hdx)/dxHdx

        return change

    def update_TS_BFGS(self, molecule):
        G = np.copy(molecule.Hessian)  # nicd,nicd
        dk = self.dx
        yk = self.dg

        jk = yk - np.dot(G, dk)
        B = force_positive_definite(G)

        # Scalar 1: dk^T |Bk| dk
        s1 = np.linalg.multi_dot([dk.T, B, dk])
        # Scalar 2: (yk^T dk)^2 + (dk^T |Bk| dk)^2
        s2 = np.dot(yk.T, dk)**2 + s1**2

        # Vector quantities
        v2 = np.dot(yk.T, dk)*yk + s1*np.dot(B, dk)
        uk = v2/s2
        Ek = np.dot(jk, uk.T) + np.dot(uk, jk.T) + np.dot(jk.T, dk) * np.dot(uk, uk.T)

        return Ek