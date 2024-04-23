"""

Auxiliary functions for the Scalar on Function model

    class AuxiliaryFunctionsSF:
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
from fungcn.ffs.auxiliary_functions_FS import AuxiliaryFunctionsFS


class AuxiliaryFunctionsSF(AuxiliaryFunctionsFS):

    def prim_obj(self, A, x, b, lam1, lam2, wgts):

        """
        primal object of the minimization problem

        """

        norms_x = LA.norm(x, axis=1).reshape(x.shape[0], 1)

        return (0.5 * LA.norm(A @ x.ravel() - b) ** 2
                + lam2 / 2 * np.sum(wgts * norms_x ** 2) + lam1 * np.sum(wgts * norms_x))

    def grad_phi(self, A, y, x, b, Aty, sgm, lam1, lam2, wgts):

        """
        gradient of the function phi(y)

        """

        return y + b - A @ self.prox(x - sgm * Aty, wgts * sgm * lam1, wgts * sgm * lam2).ravel()

    def compute_coefficients_form(self, A, b, k):
        """
        compute coefficient form given the orginal matrices

        """

        n, m, _ = A.shape
        k_suggested = -1
        if not k:
            k = 5

        # find a different basis for each feature
        print('  * performing pca for all features')
        A_coeffs = np.zeros((n, m, k))
        A_basis = np.zeros((n, A.shape[2], k))
        var_exp = 0

        for i in range(n):
            eigvals, eigenfuns = LA.eigh(A[i, :, :].T @ A[i, :, :])
            var_exp += np.cumsum(np.flip(eigvals)) / np.sum(eigvals + 1e-20)
            A_basis[i, :, :] = eigenfuns[:, -k:]
            A_coeffs[i, :, :] = A[i, :, :] @ A_basis[i, :, :]
        var_exp = var_exp / n

        # find other basis
        b_basis_full = None
        x_basis = A_basis

        # find scores
        A = A_coeffs.transpose(1, 0, 2).reshape(m, n * k)

        return A, b, k, k_suggested, b_basis_full, x_basis, var_exp

