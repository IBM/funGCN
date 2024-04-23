
import numpy as np
import random
from sklearn.gaussian_process.kernels import Matern
from tqdm import tqdm
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans


class GenerateSimFGCN:
    """
    Class to Generate a simulation for the FunGCN model. Initialize class and then call the get_design_matrix method

    Args:
        nvar: number of variables
        nobs: number of statistical units
        ntimes: number of evaluation for each longitudinal variables
        nfun0: number of longitudinal/functional variables interconnected
        ncat0: number of categorical variables interconnected
        cat0_levels (list): number of categories of each interconnected categorical variable
        nscalar0: number of scalar variables interconnected
        fun_ratio: ratio of longitudinal variables out of nvar
        cat_ratio: ratio of longitudinal variables out of nvar
        scalar_ratio: ratio of scalar variables out of nvar

    get_design_matrix outputs:
        X: design matrix of dimension (nvar, nobs, ntimes). The first nfun0 + ncat0 + nscalar0 variables in the first
            dimensions are the connected variables. For scalar and categorical variables the same values is repeated ntimes
        var_modality: a list of dimension n_var containing the modality of each variable.
            Possible values: 'f', 'c', 's' for longitudinal, categorical, and scalar respectively

    """

    def __init__(self, nvar, nobs, ntimes, nfun0, ncat0, cat0_levels, nscalar0, fun_ratio, cat_ratio, scalar_ratio,
                 min_corr=0.6, max_corr=0.9, mult_factor=10, seed=54):

        self.nvar = nvar
        self.nobs = nobs
        self.ntimes = ntimes
        self.nfun0 = nfun0
        self.ncat0 = ncat0
        self.cat0_levels = cat0_levels
        self.nscalar0 = nscalar0
        self.fun_ratio = fun_ratio
        self.cat_ratio = cat_ratio
        self.scalar_ratio = scalar_ratio
        self.min_corr = min_corr
        self.max_corr = max_corr
        self.mult_factor = mult_factor
        self.seed = seed

    def apply_gaussian_filter(self, col):
        return gaussian_filter(col, sigma=10)

    def divide_integer_with_weights(self, x, w1, w2, w3):

        # Normalize weights
        total_weight = w1 + w2 + w3
        w1_normalized = w1 / total_weight
        w2_normalized = w2 / total_weight

        # Calculate each part
        part1 = round(x * w1_normalized)
        part2 = round(x * w2_normalized)
        part3 = x - part1 - part2

        return part1, part2, part3

    def divide_integer_evenly(self, x, k):
        # Calculate the base size of each part
        base_size = x // k

        # Calculate the remainder
        remainder = x % k

        # Initialize the result list with base_size for each part
        parts = [base_size] * k

        # Distribute the remainder among the first 'remainder' parts
        for i in range(remainder):
            parts[i] += 1

        return parts

    def generate_smooth_correlation_noise(self, nvar, min_corr=0.6, max_corr=0.9, mult_factor=10):

        # Initialize the matrix with zeros
        cov_matrix = np.zeros((nvar, nvar))

        # Fill the upper triangle of the matrix with random values
        upper_triangle_indices = np.triu_indices(nvar, k=1)
        cov_matrix[upper_triangle_indices] = np.random.uniform(low=min_corr, high=max_corr, size=len(upper_triangle_indices[0]))

        # Make the matrix symmetric
        cov_matrix = (cov_matrix + cov_matrix.T) / 2

        # Set diagonal elements to 1
        np.fill_diagonal(cov_matrix, 1)

        # compute multivariate normal correlation noise
        correlation_noise = multivariate_normal.rvs(mean=np.zeros((nvar, )), cov=cov_matrix, size=self.ntimes)

        # smooth correlation noise with Guassian filter
        smooth_correlation_noise = mult_factor * np.apply_along_axis(self.apply_gaussian_filter, 0, correlation_noise)

        return smooth_correlation_noise

    def generate_Xf(self, nvar, nobs, grid, mu_X=0, sd_X=1, l_X=0.25, nu_X=3.5):
        """
        Generate design matrix X for functional features: np.array((n, m, neval))
        """
        # define fixed parameters
        # mu_X mean of features
        # sd_X standard deviation of the A Matern covariance
        # l_X range parameter of A Matern covariance
        # nu_X smoothness of A Matern covariance

        print('  * creating A for functional features')

        ntimes = grid.shape[0]
        X = np.zeros((nvar, nobs, ntimes))

        for var in tqdm(range(nvar)):
            cov_Xi = sd_X ** 2 * Matern(length_scale=l_X, nu=nu_X)(grid.reshape(-1, 1))
            X[var, :, :] = np.random.multivariate_normal(mu_X * np.ones(ntimes), cov_Xi, nobs)

        return X

    def generate_Xc(self, nvar, nobs, ntimes, min_levels=2, max_levels=5):
        """
        Generate design matrix X for categorical features
        """

        # print('  * creating A for categorical features')
        X = np.zeros((nvar, nobs, ntimes))
        for var in range(nvar):

            # find number of levels
            n_levels = random.randint(min_levels, max_levels)

            # find observations per each levels
            nobs_per_level = self.divide_integer_evenly(nobs, n_levels)

            # repeat each levels nobs per levels and shuffle the results
            x_cat = np.repeat(np.arange(0, n_levels), nobs_per_level)
            np.random.shuffle(x_cat)

            # tile cat_vec into X
            X[var, :, :] = np.tile(x_cat, (ntimes, 1)).T

        return X

    def generate_Xs(self, nvar, nobs, ntimes, mean_X=0, std_X=1):
        """
        Generate design matrix X for scalar features
        """

        X = np.zeros((nvar, nobs, ntimes))
        for var in range(nvar):
            # generate values from random normal
            x_vec = np.random.normal(mean_X, std_X, nobs)

            # tile x_vec into X
            X[var, :, :] = np.tile(x_vec, (ntimes, 1)).T

        return X

    def transform_function_in_scalar(self, Xin):

        Xout = np.zeros(Xin.shape)
        for var in range(Xin.shape[0]):

            # find vector taking mean
            x_vec = np.mean(Xin[var], axis=1)

            # tile x_vec into X
            Xout[var, :, :] = np.tile(x_vec, (self.ntimes, 1)).T

        return Xout

    def transform_scalar_in_categorical(self, Xin, cat_levels):

        Xout = np.zeros(Xin.shape)
        for var in range(Xin.shape[0]):

            # make cluster out of scalars
            x = Xin[var, :, 0].reshape(-1, 1)
            kmeans = KMeans(n_clusters=cat_levels[var], random_state=np.random.randint(0,1000), n_init='auto').fit(x)
            x_cat = 1. * kmeans.labels_

            # tile cat_vec into X
            Xout[var, :, :] = np.tile(x_cat, (self.ntimes, 1)).T

        return Xout

    def get_design_matrix(self):

        # define evalution grid of the fun variables
        grid = np.linspace(0, 1, self.ntimes)

        # find total number of correlated variables
        nvar0 = self.nfun0 + self.ncat0 + self.nscalar0

        # compute not correlated variables
        nfun, ncat, nscalar = self.divide_integer_with_weights(self.nvar - nvar0, self.fun_ratio, self.cat_ratio, self.scalar_ratio)

        # compute var modality vector
        var_modality = np.array(
            self.nfun0 * ['f'] + self.ncat0 * ['c'] + self.nscalar0 * ['s'] + nfun * ['f'] + ncat * ['c'] + nscalar * ['s'])

        # define var modality mask, where the first nvar0 variable are functional
        var_modality_mask = var_modality.copy()
        var_modality_mask[:nvar0] = 'f'

        # initialize design matrix
        X = np.zeros((self.nvar, self.nobs, self.ntimes))

        # generate functional variables
        X[var_modality_mask == 'f'] = self.generate_Xf(nvar=nvar0 + nfun, nobs=self.nobs, grid=grid)

        # generate categorical variables
        X[var_modality_mask == 'c'] = self.generate_Xc(nvar=ncat, nobs=self.nobs, ntimes=self.ntimes)

        # generate scalar variables
        X[var_modality_mask == 's'] = self.generate_Xs(nvar=nscalar, nobs=self.nobs, ntimes=self.ntimes)

        # compute smooth correlation noise between the nvar0 var and add it to X
        smooth_correlation_noise = self.generate_smooth_correlation_noise(nvar0, min_corr=self.min_corr, max_corr=self.max_corr, mult_factor=self.mult_factor)
        X[:nvar0] += smooth_correlation_noise.T[:, np.newaxis, :]

        # transform functions in scalar
        X[self.nfun0:nvar0] = self.transform_function_in_scalar(X[self.nfun0:nvar0])

        # transform scalar in categorical
        X[self.nfun0:(self.nfun0 + self.ncat0)] = self.transform_scalar_in_categorical(X[self.nfun0:(self.nfun0 + self.ncat0)], self.cat0_levels)

        return X, var_modality
