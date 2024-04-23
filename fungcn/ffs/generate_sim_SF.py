""" 
Definition of classes to generate A, b, and x for the simulations in Scalar on Function Framework

"""

import numpy as np
from sklearn.gaussian_process.kernels import Matern
from fungcn.ffs.generate_sim_FF import GenerateSimFF


class GenerateSimSF(GenerateSimFF):

    def __init__(self, seed):
        self.seed = seed

    def generate_x(self, not0, grid, sd_x, mu_x, l_x, nu_x):
        """
        Generate coefficient matrix x: np.array((n, neval, neval))
        """

        np.random.seed(self.seed + 1)
        print('  * creating features')

        neval = grid.shape[0]

        cov_x = sd_x ** 2 * Matern(length_scale=l_x, nu=nu_x)(grid.reshape(-1, 1))
        x_true = np.random.multivariate_normal(mu_x * np.ones(neval), cov_x, not0)
        return x_true

    def compute_b_plus_eps(self, A, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps):

        """
        Compute the response the errors terms epsilon and the response b
        """

        np.random.seed(self.seed + 2)
        print('  * computing b')

        neval = grid.shape[0]
        m = A.shape[1]
        # x_true_expanded = (np.eye(neval) * x_true.reshape(not0, 1, neval)).reshape(not0 * neval, neval)
        # b = np.sum(A[0:not0, :, :].transpose(1, 0, 2).reshape(m, not0 * neval) @ x_true_expanded, axis=1)
        b = A[0:not0, :, :].transpose(1, 0, 2).reshape(m, not0 * neval) @ x_true.ravel()
        b -= b.mean(axis=0)

        # create the errors -- and their covariance using a matern process
        print('  * creating errors')

        sd_eps = np.std(b) / np.sqrt(snr)
        eps = np.random.normal(0, sd_eps, (m, ))
        eps -= eps.mean(axis=0)

        b += eps

        return b, eps



