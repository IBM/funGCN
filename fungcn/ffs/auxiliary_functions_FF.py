"""

Auxiliary functions for the Function on Function model

    class AuxiliaryFunctionsFF:
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
from sklearn.model_selection import KFold
from fungcn.ffs.auxiliary_functions_FS import AuxiliaryFunctionsFS


class AuxiliaryFunctionsFF(AuxiliaryFunctionsFS):

    def prox(self, v, par1, par2):

        """
        proximal operator of the penalty

        """

        k = v.shape[1]
        n = np.int32(v.shape[0] / k)
        v_line = v.reshape(n, k * k)

        return (v_line / (1 + par2) * np.maximum(0, 1 - par1 / LA.norm(v_line, axis=1).reshape(n, 1))).reshape(n * k, k)

    def p_star(self, v, par1, par2):

        """
        conjugate function of the penalty

        """

        k = v.shape[1]
        n = np.int32(v.shape[0] / k)

        return np.sum(np.maximum(0, LA.norm(v.reshape(n, k * k), axis=1).reshape(n, 1) - par1) ** 2 / (2 * par2))

    def prim_obj(self, A, x, b, lam1, lam2, wgts):

        """
        primal object of the minimization problem

        """

        k = x.shape[1]
        n = np.int32(x.shape[0] / k)
        norms_x = LA.norm(x.reshape(n, k * k), axis=1).reshape(n, 1)

        return 0.5 * LA.norm(A @ x - b) ** 2 + lam2 / 2 * np.sum(wgts * norms_x ** 2) + lam1 * np.sum(wgts * norms_x)

    def phi_y(self, y, x, b, Aty, sgm, lam1, lam2, wgts):

        """
        function phi(y) defined in the dal algorithm

        """

        k = x.shape[1]
        n = np.int32(x.shape[0] / k)

        return (LA.norm(y) ** 2 / 2 + np.sum(b * y) +
                LA.norm(np.sqrt((1 + wgts * sgm * lam2) / (2 * sgm)) *
                        self.prox(x - sgm * Aty, wgts * sgm * lam1, wgts * sgm * lam2).reshape(n, k * k)) ** 2
                - LA.norm(x) ** 2 / (2 * sgm))

    def compute_coefficients_form(self, A, b, k):

        """
        compute coefficient form given the orginal matrices. FPC b are used for A

        """

        n, m, _ = A.shape

        # find b basis and k
        eigvals, b_basis_full = LA.eigh(b.T @ b)
        var_exp = np.cumsum(np.flip(eigvals)) / np.sum(eigvals)
        k_suggested = max(np.argwhere(var_exp > 0.9)[0][0] + 1, 3)
        if not k:
            k = k_suggested

        # find basis
        b_basis = b_basis_full[:, -k:]
        x_basis = b_basis

        # find scores
        b = b @ b_basis
        A = (A @ b_basis).transpose(1, 0, 2).reshape(m, n * k)

        return A, b, k, k_suggested, b_basis_full, x_basis, var_exp

    def compute_coefficients_form_using_FPC_features(self, A, b, k):

        """
        compute coefficient form given the orginal matrices. Each feature uses its own FPC

        """

        n, m, _ = A.shape

        # find b basis and k
        eigvals, b_basis_full = LA.eigh(b.T @ b)
        var_exp = np.cumsum(np.flip(eigvals)) / np.sum(eigvals)
        k_suggested = max(np.argwhere(var_exp > 0.9)[0][0] + 1, 3)
        if not k:
            k = k_suggested
        b_basis = b_basis_full[:, -k:]

        # find A basis
        A_coeffs = np.zeros((n, m, k))
        A_basis = np.zeros((n, A.shape[2], k))

        for i in range(n):
            eigvals, eigenfuns = LA.eigh(A[i, :, :].T @ A[i, :, :])
            A_basis[i, :, :] = eigenfuns[:, -k:]
            A_coeffs[i, :, :] = A[i, :, :] @ A_basis[i, :, :]

        # define X basis
        x_basis = np.zeros((2, n, A.shape[2], k))
        x_basis[0, :, :, :] = A_basis
        x_basis[1, :, :, :] = np.tile(b_basis, (n, 1, 1))

        # find b and A projections
        A = A_coeffs.transpose(1, 0, 2).reshape(m, n * k)
        b = b @ b_basis

        return A, b, k, k_suggested, b_basis_full, x_basis, var_exp

    def select_k_estimation(self, A_full, b_full, b_basis_full, fit):

        """
           computes k for estimation based on cv criterion

        """

        n, m, _ = A_full.shape
        k = fit.x_coeffs.shape[1]

        if k > 6:
            return fit, k
        else:

            rss_cv = []
            for ki in range(max(k, 3), 7):
                # find b basis and k
                bi_basis = b_basis_full[:, -ki:]
                bi = b_full @ bi_basis
                Ai = (A_full @ bi_basis).transpose(1, 0, 2).reshape(m, n * ki)
                AJi = Ai[:, np.repeat(fit.indx, ki)]

                kf = KFold(n_splits=5)
                kf.get_n_splits(AJi)
                rss_folds = []
                for train_index, test_index in kf.split(AJi):
                    A_cv_train, A_cv_test = AJi[train_index], AJi[test_index]
                    b_cv_train, b_cv_test = bi[train_index], bi[test_index]
                    xji = LA.solve(A_cv_train.T @ A_cv_train, A_cv_train.T @ b_cv_train)
                    rss_folds.append(LA.norm(b_cv_test - A_cv_test @ xji) ** 2)

                rss_cv.append(np.mean(rss_folds))

            k_estimation = max(k, 3) + np.argmin(rss_cv)
            if k_estimation > k:
                fit.x_basis = b_basis_full[:, -k_estimation:]
                fit.b_coeffs = b_full @ fit.x_basis
                fit.A_coeffs = (A_full @ fit.x_basis).transpose(1, 0, 2).reshape(m, n * k_estimation)
                AJ = fit.A_coeffs[:, np.repeat(fit.indx, k_estimation)]
                fit.x_coeffs = np.zeros((n * k_estimation, k_estimation))
                fit.x_coeffs[np.repeat(fit.indx, k_estimation), ] = LA.solve(AJ.T @ AJ, AJ.T @ fit.b_coeffs)

        return fit, k_estimation

    def compute_curves(self, fit, b_std):

        """
           Computes the final functional estimates from the x coefficients

        """

        x_basis, indx, x, r = fit.x_basis, fit.indx, fit.x_coeffs, fit.r
        b_std = b_std.reshape(b_std.shape[0], 1)

        if x_basis.ndim > 2:
            x_basis2 = x_basis[1, indx, :, :]
            x_curves = b_std * x_basis[0, indx, :, :] @ x[indx, :, :] @ x_basis2.transpose(0, 2, 1)
        else:
            x_curves = b_std * x_basis @ x[indx, :, :] @ x_basis.T

        return x_curves
