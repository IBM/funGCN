"""

Auxiliary functions for the Function on Scalar model

    class AuxiliaryFunctionsFS:
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
import matplotlib.pyplot as plt


class AuxiliaryFunctionsFS:

    def prox(self, v, par1, par2):

        """
        proximal operator of the penalty

        """
        return v / (1 + par2) * np.maximum(0, 1 - par1 / (LA.norm(v, axis=1).reshape(v.shape[0], 1) + 1e-20))

    def prox_star(self, v, par1, par2, t):

        """
        conjugate function of the proximal operator
        :param v: the argument is already divided by sigma: prox_(p*/sgm)(x/sgm = v)

        """

        return v - self.prox(v * t, par1, par2) / t

    def p_star(self, v, par1, par2):

        """
        conjugate function of the penalty

        """

        return np.sum(np.maximum(0, LA.norm(v, axis=1).reshape(v.shape[0], 1) - par1) ** 2 / (2 * par2))

    def prim_obj(self, A, x, b, lam1, lam2, wgts):

        """
        primal object of the minimization problem

        """

        norms_x = LA.norm(x, axis=1).reshape(x.shape[0], 1)

        return 0.5 * LA.norm(A @ x - b) ** 2 + lam2 / 2 * np.sum(wgts * norms_x ** 2) + lam1 * np.sum(wgts * norms_x)

    def dual_obj(self, y, z, b, lam1, lam2, wgts):

        """
         dual_obj: dual object of the minimization problem

        """

        return - (0.5 * LA.norm(y) ** 2 + np.sum(b * y) + self.p_star(z, wgts * lam1, wgts * lam2))

    def phi_y(self, y, x, b, Aty, sgm, lam1, lam2, wgts):

        """
        function phi(y) defined in the dal algorithm

        """

        return (LA.norm(y) ** 2 / 2 + np.sum(b * y) +
                LA.norm(np.sqrt((1 + wgts * sgm * lam2) / (2 * sgm)) *
                        self.prox(x - sgm * Aty, wgts * sgm * lam1, wgts * sgm * lam2)) ** 2
                - LA.norm(x) ** 2 / (2 * sgm))

    def grad_phi(self, A, y, x, b, Aty, sgm, lam1, lam2, wgts):

        """
        gradient of the function phi(y)

        """

        return y + b - A @ self.prox(x - sgm * Aty, wgts * sgm * lam1, wgts * sgm * lam2)

    def compute_coefficients_form(self, A, b, k):

        """
        compute coefficient form given the orginal matrices

        """

        # find k
        eigvals, b_basis_full = LA.eigh(b.T @ b)
        var_exp = np.cumsum(np.flip(eigvals)) / np.sum(eigvals)
        k_suggested = min(5, max(np.argwhere(var_exp > 0.99)[0][0] + 1, 2))
        if not k:
            k = k_suggested

        # find basis
        b_basis = b_basis_full[:, -k:]
        x_basis = b_basis

        # find scores
        b = b @ b_basis

        return A, b, k, k_suggested, b_basis_full, x_basis, var_exp

    def compute_curves(self, fit, b_std):

        """
        Computes the final functional estimates from the x coefficients

        """

        x_basis, indx, x, r = fit.x_basis, fit.indx, fit.x_coeffs, fit.r
        k = x.shape[1]

        if x_basis.ndim > 2:
            x_curves = b_std * (x[indx, :].reshape(r, 1, k) @
                                x_basis[indx, :, :].transpose(0, 2, 1)).reshape(r, x_basis.shape[1])

        else:
            x_curves = x[indx, :] @ (b_std.reshape(b_std.shape[0], 1) * x_basis).T

        return x_curves

    def standardize_design_matrix(self, A, categorical=False):

        """
        function to standardize the design matrix

        """

        if len(A.shape) == 2:

            if categorical:
                frequencies = np.sum(A, axis=0)
                return (A - frequencies / A.shape[0]) / np.sqrt(frequencies)

            else:
                return (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-32)

        else:
            # for i in range(A.shape[0]):
            #     A[i, :, :] = (A[i, :, :] - A[i, :, :].mean(axis=0)) / (A[i, :, :].std(axis=0) + 1e-32)
            return (A -  A.mean(axis=1)[:, np.newaxis]) / ((A.std(axis=1) + 1e-32)[:, np.newaxis])

    def plot_selection_criterion(self, r, selection_criterion, alpha, grid, main=None):

        """
        plots of: r, ebic, gcv, cv for different values of alpha

        :param r: list of r_vec. Each element of the list is the r_vec values for the respective alpha in alpha_list
        :param selection_criterion: list of selection_criterion_vec. One vector for each value of alpha.
        :param alpha: vec of different value of alpha considered
        :param grid: array of all the c_lam considered (same for all alphas)
        :param main: main for selection criterion plot

        """

        # if the inputs are not list, we create them:
        if type(r) != list:
            r_list, selection_criterion_list = list(), list()
            r_list.append(r)
            selection_criterion_list.append(selection_criterion)
            alpha_vec = np.array([alpha])
            n_alpha = 1
        else:
            r_list, selection_criterion_list, alpha_vec = r, selection_criterion, alpha
            n_alpha = alpha.shape[0]

        fig, ax = plt.subplots(2, 1)

        # r
        for i in range(n_alpha):
            indx = r_list[i] != -1
            t = grid[:r_list[i].shape[0]][indx]
            ax[0].plot(t, r_list[i][indx], label=('alpha = %.2f' % alpha_vec[i]))
        ax[0].legend(loc='best')
        ax[0].set_title('selected features')
        ax[0].set_xlim([grid[0], grid[-1]])

        # selection_criterion_list
        for i in range(n_alpha):
            indx = selection_criterion_list[i] != -1
            t = grid[:selection_criterion_list[i].shape[0]][indx]
            ax[1].plot(t, selection_criterion_list[i][indx], label=('alpha = %.2f' % alpha_vec[i]))
        ax[1].legend(loc='best')
        ax[1].set_title(main)
        ax[1].set_xlim([grid[0], grid[-1]])

        plt.show()
