"""

Auxiliary functions for the logit model

    class AuxiliaryFunctionsLogit:
            prox: proximal operator of the penalty
            prox_star: conjugate function of the proximal operator
            p_star: conjugate function of the penalty
            prim_obj: primal object of the minimization problem
            dual_obj: dual object of the minimization problem
            phi_y: function phi(y) defined in the dal algorithm
            grad_phi: gradient of the function phi(y)
            standardize_design_matrix
            plot_selection_criteria
"""


import numpy as np
from numpy import linalg as LA
from scipy.special import logit
from fungcn.ffs.auxiliary_functions_SF import AuxiliaryFunctionsSF


class AuxiliaryFunctionsLogit(AuxiliaryFunctionsSF):

    def prim_obj(self, A, x, b, lam1, lam2, wgts):

        """
        primal object of the minimization problem

        """

        # zz = - A @ x.ravel()
        # yy = b
        # zy = zz * yy
        # z2 = 0.5 * np.array([zy, -zy]).T
        # outmax = np.max(z2, axis=1)
        # sumexp = np.sum(np.exp(z2 - np.array([outmax, outmax]).T), axis=1)
        # logpout = z2 - np.repeat((outmax + np.log(sumexp)), 2).reshape(A.shape[0], 2)
        # prim3 = -np.sum(logpout[:, 0])
        # norms_x = LA.norm(x, axis=1).reshape(x.shape[0], 1)
        # return (prim3
        #         + lam2 / 2 * np.sum(wgts * norms_x ** 2) + lam1 * np.sum(wgts * norms_x))

        norms_x = LA.norm(x, axis=1).reshape(x.shape[0], 1)
        Ax = - A @ x.ravel()
        return (np.sum(np.log(1 + np.exp(- b * Ax)))
                + lam2 / 2 * np.sum(wgts * norms_x ** 2) + lam1 * np.sum(wgts * norms_x))

    def dual_obj(self, y, z, b, lam1, lam2, wgts):

        """
         dual_obj: dual object of the minimization problem

        """

        # b = np.abs(b)
        by = np.abs(b * y)
        # by = np.minimum(by, 0.5)
        # by = b * y
        byI = by[np.where((by > 0) & (by < 1))]

        return - (np.sum((1 - byI) * np.log(1 - byI) + byI * np.log(byI))
                  + self.p_star(z, wgts * lam1, wgts * lam2))

    def grad_phi(self, A, y, x, b, Aty, sgm, lam1, lam2, wgts):

        """
        gradient of the function phi(y)

        """

        # b = np.abs(b)
        by = np.abs(b * y)
        # by = np.minimum(by, 0.5)
        # by = b * y
        I = np.where((by > 0) & (by < 1))
        gloss = np.zeros(y.shape[0])
        gloss[I] = b[I] * logit(by[I])

        return gloss - A @ self.prox(x - sgm * Aty, wgts * sgm * lam1, wgts * sgm * lam2).ravel()

    def phi_y(self, y, x, b, Aty, sgm, lam1, lam2, wgts):
        """
        function phi(y) defined in the dal algorithm

        """

        # b = np.abs(b)
        by = np.abs(b * y)
        # by = np.minimum(by, 0.5)
        # by = b * y
        byI = by[np.where((by > 0) & (by < 1))]

        return (np.sum((1 - byI) * np.log(1 - byI) + byI * np.log(byI)) +
                LA.norm(np.sqrt((1 + wgts * sgm * lam2) / (2 * sgm)) *
                        self.prox(x - sgm * Aty, wgts * sgm * lam1, wgts * sgm * lam2).ravel()) ** 2
                - LA.norm(x) ** 2 / (2 * sgm))

