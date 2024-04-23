""" 
Definition of classes to generate A, b, and x for the simulations in logit framework

"""

import numpy as np
from scipy.stats import bernoulli
from scipy.special import expit
from fungcn.ffs.generate_sim_SF import GenerateSimSF


class GenerateSimLogit(GenerateSimSF):

    def __init__(self, seed):
        self.seed = seed

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
        Ax = A[0:not0, :, :].transpose(1, 0, 2).reshape(m, not0 * neval) @ x_true.ravel()

        # create the errors -- and their covariance using a matern process
        print('  * creating errors')

        sd_eps = np.std(Ax) / np.sqrt(snr)
        eps = np.random.normal(0, sd_eps, (m,))
        eps -= eps.mean(axis=0)

        # Ax += eps
        b = bernoulli.rvs(expit(Ax), size=m)
        b[b < 1] = -1

        return b, eps



