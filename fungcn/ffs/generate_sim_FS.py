""" 
Definition of classes to generate A, b, and x for the simulations in Function on Scalar framework

"""

import numpy as np
from sklearn.gaussian_process.kernels import Matern


class GenerateSimFS:

    def __init__(self, seed):
        self.seed = seed

    def generate_A(self, n, m):

        """
        Generate design matrix A: np.array((n, m))
        """

        np.random.seed(self.seed)
        print('  * creating A')

        A = np.random.normal(0, 1, (m, n))
        return (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-32)

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
        m = A.shape[0]
        b = A[:, 0:not0] @ x_true
        b -= b.mean(axis=0)

        # create the errors -- and their covariance using a matern process
        print('  * creating errors')

        sd_eps = np.std(b) / np.sqrt(snr)
        cov_eps = sd_eps ** 2 * Matern(length_scale=l_eps, nu=nu_eps)(grid.reshape(-1, 1))
        eps = np.random.multivariate_normal(mu_eps * np.ones(neval), cov_eps, m)
        eps -= eps.mean(axis=0)

        b += eps

        return b, eps



