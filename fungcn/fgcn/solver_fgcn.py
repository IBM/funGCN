"""
FunGCN class

"""

import numpy as np
import warnings
import scipy.sparse as sp
import torch
from tqdm.auto import tqdm
from numpy import linalg as LA
from fungcn.ffs.enum_classes import AdaptiveScheme, RegressionType, SelectionCriteria
from fungcn.ffs.solver_path import FASTEN
from fungcn.fgcn.gcn_model import GCNModel
from fungcn.fgcn.output_classes import OutputPrediction
from fungcn.fgcn.early_stopper import EarlyStopper
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import splrep, splev, BSpline
import pandas as pd
import matplotlib.pyplot as plt


class FunGCN(torch.nn.Module):
    """
    Class to define the Functional Graph Convolutional Network models

    Methods:
        standardize_X: standardizes design matrix
        compute_categories_embeddings: computes embeddings for categorical data
        compute_graph_embeddings: computes embeddigns for the the knowledge graph creation
        find_forecast_domain: divides the time and the basis coefficients into past and horizon
        compute_gcn_embeddings: computes embeddigns for the GCN model
        update_embeddings: compute and update (if necessary) all the embeddings, calling compute_graph_embeddings and
            compute_gcn_embeddings
        split_data_train_validation_test: splits the data into train, validation, and test set
        standardize_data: standardizes data train, validation, and test sets
        isolate_target_from_data: remove the target information Y from the design matrix X
        preprocess_data: calls all the functions above to preprocess the data
        graph_estimation: implements a node-wise feature selection approach to estimate the graph. FASTEN feature
            selection is performed
        preprocess_adjacency_matrix: takes the output of graph_estimation and preprocess it for the GCN module
        initialize_gcn_module: initialization of the graph convolutional network model
        train_model: trains the GCN
        retrieve_original_form: starting from the embedded forms, recovers the original form for the GCN outcomes
        compute_evaluation_metrics: computes the evaluation metrics on the original feature space
        predict: predict new observations (or the test set) using the trained model
        print_prediction_metrics: print the evaluation metrics computed on the new observations (or the test set)
        plot_true_vs_predicted: plots the truth and predicted longitudinal features

    Attributes:
        X_gcn: (nobs, nvar, k_gcn) embedded array for GCN
        X_gcn_test: (nobs_test, nvar, k_gcn) embedded array for the GCN testing
        X_gcn_test: (nobs_train, nvar, k_gcn) embedded array for the GCN training
        X_gcn_test: (nobs_val, nvar, k_gcn) embedded array for the GCN validation
        X_graph: (nvar, nobs, k_graph) embedded array for graph estimation
        X_graph_train: (nvar_train, nobs, k_graph) embedded array with observations used for graph_estimation
        accuracy_test (dict): keys are categorical targets and values are accuracy on the test sets
        adjacency (nvar, nvar): adjancency matrix
        adjusted_start_forecast: position of the first basis in the forecasting domain
        basis_gcn: basis used to represents the longitudinal features for gcn
        basis_graph: basis used to represents the longitudinal features for graph estimation
        data: input data, array of dimensions (nvar, nobs, ntimes)
        edge_index: list of the graph edges
        edge_weigths: list of the graph weights
        forecast_ratio: portion of prediction with respect to the total domain if the task is forecasting
        k_gcn_in: number of coefficients of the GCN input
        k_gcn_out: number of coefficients of the GCN output
        k_graph: number of coefficients for graph estimation
        loss_train: loss on the training set
        loss_val: loss on the validation set
        mean_vec_gcn: vector with the original features means, used to reconstruct the orginal features
        model: the gcn model
        nedges: number of edges for each node of the graph
        rmse_test (dict): keys are longitudinal/scalar targets and values are the rmse on the test sets
        start_forecsat: time at which the forecasting starts
        std_rmse_test (dict): keys are longitudinal/scalar targets and values are the standardized rmse on the test sets
        std_vec_gcn: vector with the original features standard deviations, used to reconstruct the orginal features
        test_ind: index of the test set
        train_ind: index of the training set
        val_ind: index of the validation set
        var_modality: list of modalities for each variables; possible values: 'f', 'c', 's'
        verbose: if true, the output is printed
        y_gcn_train: targets for the gcn restricted on the train set
        y_gcn_val: targets for the gcn restricted on the validation set
        y_ind: list containing the indeces of the target variables
        y_test_hat_embedded: estimated features on the test set and in the embedded space
        y_test_hat_os: estimated features on the test set and retransformed in the original space
        y_train_hat_embedded: estimated features on the trian set and in the embedded space
        y_val_hat_embedded: estimated features on the validation set and in the embedded space

    """

    def __init__(self, data, y_ind, var_modality=None, verbose=1):
        super().__init__()

        if var_modality is None:
            var_modality = np.repeat('f', data.shape[0])

        self.data = data
        self.y_ind = y_ind
        self.var_modality = var_modality
        self.verbose = verbose

        # attributes to initialize to None
        # TODO: add to this list
        self.model = None
        self.X_graph_train = None
        self.X_gcn_train = None
        self.X_gcn_test = None
        self.X_gcn_val = None
        self.k_graph = None
        self.k_gcn_in = None
        self.k_gcn_out = None
        self.forecast_ratio = None
        self.start_forecast = 0
        self.adjusted_start_forecast = 0
        self.adjacency = None
        self.train_ind = None
        self.test_ind = None
        self.val_ind = None
        self.mean_vec_gcn = None
        self.std_vec_gcn = None
        self.loss_train = None
        self.loss_val = None
        self.rmse_test = None
        self.std_rmse_test = None
        self.accuracy_test = None
        self.y_test_hat_os = None
        # self.mask_f = None

    def standardize_X(self, X):
        """
        Standardizes the input matrix X by subtracting the mean and dividing by the standard deviation of each feature.

        Parameters:
        X (numpy.ndarray): A 2D numpy array where each row represents a feature and each column represents an
            observation.

        Returns:
        tuple:
            Xstd (numpy.ndarray): The standardized data matrix.
            mean_vec (numpy.ndarray): The vector of means of each feature.
            std_vec (numpy.ndarray): The vector of standard deviations of each feature.

        Description:
        This function computes the mean and standard deviation for each feature (i.e., row) of the input matrix X.
        It then standardizes each feature by subtracting its mean and dividing by its standard deviation.
        A small constant (1e-32) is added to the standard deviation before division to prevent division by zero.
        """

        # Calculate the mean for each feature
        mean_vec = X.mean(axis=1)[:, np.newaxis]

        # Calculate the standard deviation for each feature, add a small constant to avoid division by zero
        std_vec = (X.std(axis=1) + 1e-32)[:, np.newaxis]

        # Standardize the data
        Xstd = (X - mean_vec) / (std_vec + 1e-32)

        return Xstd, mean_vec, std_vec
        # return X, np.zeros(mean_vec.shape), np.ones(std_vec.shape)

    def compute_categories_embeddings(self, x, k):

        n_val = np.max(np.round(x)).astype(int) + 2
        embedding_layer = torch.nn.Embedding(num_embeddings=n_val, embedding_dim=k)
        return embedding_layer(torch.LongTensor(x)).detach().numpy()

    def compute_graph_embeddings(self, k, data=None, var_modality=None):
        """
        Computes graph embeddings for data based on variable modalities.

        Parameters:
        k (int): The number of dimensions for the embeddings.
        data (numpy.ndarray, optional): A 3D array where the first dimension represents features, the second dimension
            represents observations, and the third dimension represents data points. If None, defaults to self.data.
        var_modality (list, optional): A list indicating the modality of each variable ('c' for categorical).
            If None, defaults to self.var_modality.

        Returns:
        tuple: A tuple containing:
            X (numpy.ndarray): The computed embeddings for each feature across all observations.
            basis_x (numpy.ndarray): The basis used for non-categorical data embeddings.
            k (int): The embedding dimensionality, returned for reference.

        Description:
        This method handles different types of variable modalities. For categorical data, it computes embeddings using
        a dedicated method. For other types, it uses eigen-decomposition to create basis functions and project the
        original data into a lower-dimensional space defined by the top k eigenvectors.
        """

        if data is None:
            data = self.data  # Use default data if none provided

        if var_modality is None:
            var_modality = self.var_modality  # Use default variable modalities if none provided

        nfeat, nobs, npoints = data.shape  # Extract dimensions of the data

        X = np.zeros((nfeat, nobs, k))  # Initialize the array for embeddings
        basis_x = np.zeros((nfeat, npoints, k))  # Initialize the basis array for non-categorical data

        for i in range(nfeat):  # Iterate over each feature
            if var_modality[i] == 'c':
                # Compute embeddings for categorical data
                X[i, :, :] = self.compute_categories_embeddings(data[i, :, 0], k)
            else:
                # Compute embeddings for non-categorical data using eigen-decomposition
                _, eigenfuns = LA.eigh(data[i, :, :].T @ data[i, :, :])
                basis_x[i, :, :] = eigenfuns[:, -k:]
                X[i, :, :] = data[i, :, :] @ basis_x[i, :, :]

        return X, basis_x, k  # Return the embeddings, basis, and embedding dimension

    def find_forecast_domain(self, k, forecast_ratio, npoints, degree):
        """
        Calculates the domains for historical and forecast data based on the provided forecast ratio.

        Parameters:
        k (int): Total number of knots.
        forecast_ratio (float): The proportion of the point series to be used for forecasting.
        npoints (int): Total number of points in the data series.
        degree (int): Degree of the splines used in the model.

        Returns:
        tuple: Contains various parameters calculated for historical and forecast data including:
            - t (array): Combined knots for both historical and forecast data.
            - k1 (int): Number of effective knots for historical data.
            - k2 (int): Number of effective knots for forecast data.
            - grid1 (array): Grid points for historical data.
            - grid2 (array): Grid points for forecast data.
            - knots1 (array): Knots for historical data.
            - knots2 (array): Knots for forecast data.

        Description:
        This function splits the data into historical and forecast domains based on the forecast ratio.
        It calculates separate sets of knots, grid points, and effective knot counts for both domains.
        Raises an exception if the number of knots is too few based on the spline degree required.
        """

        # Calculate the starting index for the forecast based on the forecast ratio
        start_forecast = int(np.round(npoints * (1 - forecast_ratio)))

        # Define the full grid of points
        grid = np.arange(npoints)
        # Calculate the number of interior knots
        nknots = k - 2 * degree - 2
        # Generate knots evenly spaced within the data range
        knots = np.linspace(1, npoints - 2, nknots)

        # Separate knots for historical data (before forecast start)
        knots1 = knots[knots < start_forecast]
        # Grid points for historical data
        grid1 = grid[grid < (start_forecast + 1)]
        # Prepare historical knots for basis functions
        t1 = np.append(np.zeros(degree + 1), knots1[:-degree])
        k1 = t1.shape[0]

        # Ensure there are enough knots for spline basis
        if knots1.shape[0] < degree + 1:
            raise Exception('k history is too low: increase k_gcn or reduce forecast ratio')

        # Separate knots for forecast data (after historical data)
        adjusted_start = int(t1[-1] + 1)
        knots2 = knots[knots >= adjusted_start]
        grid2 = grid[grid >= adjusted_start]
        t2 = knots2.copy()
        k2 = t2.shape[0]

        # Ensure there are enough knots for spline basis in forecast
        if knots2.shape[0] < degree + 1:
            raise Exception('k horizon is too low: increase k_gcn or increase forecast ratio')

        # Final combined knots for full basis
        t = np.concatenate([t1, t2, np.full(degree + 1, npoints - 1)])

        return t, k1, k2, grid1, grid2, knots1, knots2

    def compute_gcn_embeddings(self, k, forecast_ratio=0., data=None, var_modality=None):

        if data is None:
            data = self.data

        if var_modality is None:
            var_modality = self.var_modality

        nfeat, nobs, npoints = data.shape
        degree = 3
        k1, k2 = k, k

        if forecast_ratio > 0:
            # prepare data for forecast task

            # divide domain
            t, k1, k2, grid1, grid2, knots1, knots2 = self.find_forecast_domain(k=k, forecast_ratio=forecast_ratio, npoints=npoints, degree=degree)

            # initialize variables
            X = np.zeros((nfeat, nobs, k1 + k2))
            data_hist = data[:, :, grid1]
            data_horizon = data[:, :, grid2]
            stop_c2 = k2 + degree + 1

            # update start_forecast attribute
            self.start_forecast = grid1[-1]
            self.adjusted_start_forecast = grid2[0]

            for i in range(nfeat):

                if var_modality[i] == 'c':
                    X[i, :, :k1] = self.compute_categories_embeddings(data[i, :, 0], k1)

                elif var_modality[i] == 's':
                    for obs in range(nobs):
                        _, c1, _ = splrep(x=grid1, y=data_hist[i, obs, :], k=degree, task=-1, t=knots1)
                        X[i, obs, :k1] = c1[:k1]

                elif var_modality[i] == 'f':
                    for obs in range(nobs):
                        _, c1, _ = splrep(x=grid1, y=data_hist[i, obs, :], k=degree, task=-1, t=knots1)
                        _, c2, _ = splrep(x=grid2, y=data_horizon[i, obs, :], k=degree, task=-1, t=knots2)
                        X[i, obs, :k1] = c1[:k1]
                        X[i, obs, k1:] = c2[(degree + 1):stop_c2]
        else:

            # prepare data for regression/classification task
            X = np.zeros((nfeat, nobs, k))
            grid = np.arange(npoints)
            nknots = k - degree - 1
            knots = np.linspace(1, npoints - 2, nknots)

            for i in range(nfeat):
                if var_modality[i] == 'c':

                    X[i, :, :] = self.compute_categories_embeddings(data[i, :, 0], k)
                # elif var_modality[i] == 's':
                #     X[i, :, :] = np.tile(data[i, :, 0], (k, 1)).T

                else:
                    for obs in range(nobs):
                        t, c, _ = splrep(x=grid, y=data[i, obs, :], k=degree, task=-1, t=knots)
                        X[i, obs, :] = c[:k]

        # standardize X
        # X, mean_vec, std_vec = self.standardize_X(X)

        return X, t, k1, k2

    def update_embeddings(self, k_graph, k_gcn):

        """
        Check when update embeddings is necessary

        """

        # embedding for graph
        if k_graph != self.k_graph:
            # update embedding for graph
            self.X_graph, self.basis_graph, self.k_graph = self.compute_graph_embeddings(k=k_graph)

        # embedding for gcn
        if k_gcn != self.k_gcn_in:
            # update embedding for gcn
            self.X_gcn, self.basis_gcn, self.k_gcn_in, self.k_gcn_out = self.compute_gcn_embeddings(k=k_gcn, forecast_ratio=self.forecast_ratio)

        # printing
        if self.verbose and self.forecast_ratio > 0.:
            print('')
            print('k history:', self.k_gcn_in)
            print('k horizon:', self.k_gcn_out)
            print('starting forecasting at', self.start_forecast)

    def split_data_train_validation_test(self, test_size=0., val_size=0., random_state=1):

        # find all the indexes
        self.train_ind = np.arange(self.data.shape[1])

        # initialize graph and gcn train
        self.X_graph_train = self.X_graph.copy()
        X = self.X_gcn.copy()

        if test_size > 0:
            # update train ind and find test ind
            self.train_ind, self.test_ind = train_test_split(self.train_ind, test_size=test_size, random_state=random_state)

            # save graph train
            self.X_graph_train = self.X_graph[:, self.train_ind, :]

            # save gcn test
            self.X_gcn_test = X[:, self.test_ind, :]

        if val_size > 0:
            # update train ind and find val ind
            self.train_ind, self.val_ind = train_test_split(self.train_ind, test_size=val_size, random_state=random_state)

            # save gcn val
            self.X_gcn_val = X[:, self.val_ind, :]

        # save gcn train
        self.X_gcn_train = X[:, self.train_ind, :]
    
    def standardize_data(self, test_size=0., val_size=0.):

        # standardize and create torch tensor for graph
        self.X_graph_train, _, _ = self.standardize_X(self.X_graph_train)

        if test_size > 0:
            # standardize and create torch tensor for gcn test
            self.X_gcn_test, _, _ = self.standardize_X(self.X_gcn_test)
            self.X_gcn_test = torch.FloatTensor(self.X_gcn_test.transpose(1, 0, 2))

        if val_size > 0:

            # you have to standardize train and val together, merge them:
            X_big = np.concatenate((self.X_gcn_train, self.X_gcn_val), 1)
            X_big, self.mean_vec_gcn, self.std_vec_gcn = self.standardize_X(X_big)

            # re-extract val and train
            self.X_gcn_train = X_big[:, :self.train_ind.shape[0], :]
            self.X_gcn_val = X_big[:, self.train_ind.shape[0]:, :]

            # create tensors
            self.X_gcn_train = torch.FloatTensor(self.X_gcn_train.transpose(1, 0, 2))
            self.X_gcn_val = torch.FloatTensor(self.X_gcn_val.transpose(1, 0, 2))

        else:
            # standardize train and create tensor
            self.X_gcn_train, self.mean_vec_gcn, self.std_vec_gcn = self.standardize_X(self.X_gcn_train)
            self.X_gcn_train = torch.FloatTensor(self.X_gcn_train.transpose(1, 0, 2))

    def isolate_target_from_data(self, k_gcn_in):

        # forecast task
        if self.forecast_ratio > 0:

            # define X and y for gcn train
            self.y_gcn_train = self.X_gcn_train[:, self.y_ind, k_gcn_in:].clone()
            self.X_gcn_train = self.X_gcn_train[:, :, :k_gcn_in]

            # define X and y for gcn val
            if self.X_gcn_val is not None:
                self.y_gcn_val = self.X_gcn_val[:, self.y_ind, k_gcn_in:].clone()
                self.X_gcn_val = self.X_gcn_val[:, :, :k_gcn_in]

            # define X and y for gcn test
            # you don't have to define self.y_gcn_test, because for the test you use the original form
            if self.X_gcn_test is not None:
                self.X_gcn_test = self.X_gcn_test[:, :, :k_gcn_in]

        # regression/classification task
        else:

            # define X and y for gcn train
            self.y_gcn_train = self.X_gcn_train[:, self.y_ind, :].clone()
            self.X_gcn_train[:, self.y_ind, :] = 0

            # define X and y for gcn val
            if self.X_gcn_val is not None:
                self.y_gcn_val = self.X_gcn_val[:, self.y_ind, :].clone()
                self.X_gcn_val[:, self.y_ind, :] = 0

            # define X and y for gcn test
            # you don't have to define self.y_gcn_test, because for the test you use the original form
            if self.X_gcn_test is not None:
                self.X_gcn_test[:, self.y_ind, :] = 0

    def preprocess_data(self, k_graph, k_gcn=None, forecast_ratio=0., test_size=0., val_size=0., random_state=None):

        if self.verbose:
            print('')
            print('Preprocessing data')

        if k_gcn is None:
            k_gcn = k_graph

        if random_state is None:
            random_state = np.random.randint(1, 1e10, 1)

        if forecast_ratio > 0. and any(var_mod_target != 'f' for var_mod_target in self.var_modality[self.y_ind]):
            raise Exception('Forecast ratio is > 0, the targets must be longitudinal variables')

        # save forecast_ratio
        self.forecast_ratio = forecast_ratio

        # compute embeddings form if necessary
        self.update_embeddings(k_graph=k_graph, k_gcn=k_gcn)

        # split data in test and validation
        self.split_data_train_validation_test(test_size=test_size, val_size=val_size, random_state=random_state)

        # standardize data
        self.standardize_data(test_size=test_size, val_size=val_size)

        # divide X and Y
        self.isolate_target_from_data(k_gcn_in=self.k_gcn_in)

    def graph_estimation(self, max_selected=10, graph_path=None):

        # initialize solver and parameters
        solver = FASTEN()
        nfeat, _, k = self.X_graph_train.shape
        adjacency = np.zeros((nfeat, nfeat))

        # TODO: pass as parameters?
        # adaptive_scheme = AdaptiveScheme.NONE
        # selection_criteria = SelectionCriteria.EBIC
        # alpha = 0.2

        if self.verbose:
            print('')
            print('solving FASTEN feature selection for each node variable')

        for i in tqdm(range(nfeat)):

            # TODO: now you treat categorical as vectors of dimension k and you run function-on-function
            #  for scalar, when they are the responses, you run scalar-on-functoin
            # functional response
            if self.var_modality[i] == 'f' or self.var_modality[i] == 'c':
                regression_type = RegressionType.FF
                y_i = self.X_graph_train[i, :, :]
                # print(self.basis_graph.shape)
                # basis_i = np.delete(self.basis_graph, i, 1)
                basis_i = self.basis_graph.copy()

            # scalar response
            else:
                regression_type = RegressionType.SF
                y_i = self.X_graph_train[i, :, 0]
                # standardize scalar response
                y_i = (y_i - y_i.mean()) / (y_i.std() + 1e-32)
                # basis_i = np.delete(self.basis_graph, i, 0)
                basis_i = self.basis_graph.copy()

            # delete response variable from A and A_basis
            X_i = np.delete(self.X_graph_train, i, 0)

            out_i = solver.solver(
                regression_type=regression_type,
                A=X_i, b=y_i, k=k, wgts=1,
                selection_criterion=SelectionCriteria.EBIC, n_folds=10,
                adaptive_scheme=AdaptiveScheme.NONE,
                coefficients_form=True, x_basis=basis_i,
                c_lam_vec=np.geomspace(1, 0.005, num=100),
                c_lam_vec_adaptive=None,
                max_selected=max_selected, check_selection_criterion=False,
                alpha=0.2, lam1_max=None,
                x0=None, y0=None, z0=None, Aty0=None,
                relaxed_criteria=False, relaxed_estimates=False,
                compute_curves=False, select_k_estimation=False,
                tol_nwt=1e-5, tol_dal=1e-6,
                plot=False, print_lev=0)

            adjacency[i, np.arange(nfeat) != i] = out_i.c_lam_entry_value

        # make theta symmetric and upper triangular
        self.adjacency = (adjacency + adjacency.T) / 2

        # save Theta
        if graph_path:
            adjacency_saved = np.triu(self.adjacency)
            np.savetxt(graph_path, adjacency_saved, fmt='%f', delimiter=',')

    def preprocess_adjacency_matrix(self, pruning=0.5):

        """
        Symmetrically normalize adjacency matrix, create torch object, and extract edges information

        """
        norm_adj = self.adjacency.copy()

        # prune matrix
        norm_adj[norm_adj < pruning] = 0

        # save n edges for each node
        self.nedges = (norm_adj > 0).sum(1)

        norm_adj += np.identity(norm_adj.shape[0])

        # normalize matrix
        degrees = np.power(np.array(norm_adj.sum(1)) + 1e-10, -0.5).ravel()
        degrees[np.isinf(degrees)] = 0.0
        D = np.diag(degrees)
        norm_adj = sp.coo_matrix((norm_adj @ D).T @ D)

        # extract degrees, weights and index information
        self.edge_weights = torch.FloatTensor(norm_adj.data)
        self.edge_index = torch.LongTensor(np.stack([norm_adj.row, norm_adj.col]))

    def initialize_gcn_model(self, pruning=0.5, nhid=None, dropout=0., kernel_size=0):

        if nhid is None:
            nhid = [64, 32]

        # preprocess adjacency matrix
        self.preprocess_adjacency_matrix(pruning=pruning)

        # check n_edges for target nodes
        if self.verbose:
            # print(self.nedges)
            print(' ')
            print('Target nodes edges:', self.nedges[self.y_ind])
            print(' ')
            if min(self.nedges[self.y_ind]) == 0:
                warnings.warn('WARNING: One or more target nodes have 0 edges: decrease pruning parameters or discard target', UserWarning)

        # initialize GCN model
        self.model = GCNModel(y_ind=self.y_ind, n_nodes=self.X_gcn_train.shape[1], edge_index=self.edge_index,
                              edge_weights=self.edge_weights, dim_input=self.k_gcn_in, dim_output=self.k_gcn_out,
                              nhid=nhid, dropout=dropout, kernel_size=kernel_size)

    def train_model(self, lr=0.005, epochs=100, batch_size=1, patience=5, min_delta=0.1):

        # call GCN model and stopper
        stopper = EarlyStopper(patience=patience, min_delta=min_delta)

        # setting model for training
        self.model.train()

        # define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        # optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, alpha=0.9)

        # define loss criterion
        # criterion = F.mse_loss

        # define output matrix
        self.y_train_hat_embedded = np.zeros(self.y_gcn_train.shape)

        # train the model
        for epoch in range(epochs):

            # # decrease learning rate
            # if (epoch + 1) % 10 == 0:
            #     if self.verbose:
            #         print('decreasing Learning Rate')
            #     lr *= 2
            #     optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            # initialize loss
            loss = 0
            loss_train = 0

            for obs, x_obs in enumerate(self.X_gcn_train):

                # compute y_hat
                y_hat = self.model(x_obs)
                self.y_train_hat_embedded[obs] = y_hat.detach()
                loss += torch.mean((y_hat - self.y_gcn_train[obs]) ** 2)

                # loss += torch.mean((y_hat[:, 0] - y_train[obs][:, 0]) ** 2)
                # loss += criterion(y_hat, snapshot.y.reshape(k, ))

                # TODO: get rid of batch_size if useless
                # train depending on batch
                if (obs + 1) % batch_size == 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_train += loss
                    loss = 0

            if self.X_gcn_val is not None:

                # define outputs
                loss_val = 0
                self.y_val_hat_embedded = np.zeros(self.y_gcn_val.shape)

                with torch.no_grad():
                    for obs, x_obs in enumerate(self.X_gcn_val):
                        y_hat = self.model(x_obs)
                        self.y_val_hat_embedded[obs] = y_hat.detach()
                        loss_val += torch.mean((y_hat - self.y_gcn_val[obs]) ** 2)
                        # loss_val += torch.mean((y_hat[:, 0] - y_val[obs][:, 0]) ** 2)

                    # update loss val and check early stop criterion
                    self.loss_val = loss_val / self.val_ind.shape[0]
                    if stopper.early_stop(self.loss_val):
                        if self.verbose:
                            print("Early stopping")
                        break

            # save loss train
            self.loss_train = loss_train / self.train_ind.shape[0]

            if self.verbose:
                print(f'Epoch: {epoch + 1}, Loss Train: {self.loss_train}, Loss Val: {self.loss_val}')

    def retrieve_original_form(self, y, X=None, basis_x=None, mean_vec_y=None, std_vec_y=None):
        """
        function to pass from the estimated y in the coefficient form to the estimated y in the original space (os)
        param: y --> y in the gcn_coefficient form: (nobs, nfeat, k_gcn_in)
        """

        basis_x = self.basis_gcn if basis_x is None else basis_x
        std_vec_y = self.std_vec_gcn[self.y_ind] if std_vec_y is None else std_vec_y
        mean_vec_y = self.mean_vec_gcn[self.y_ind] if mean_vec_y is None else mean_vec_y

        # define output matrix
        npoints = self.data.shape[2]
        grid = np.arange(npoints)
        degree = 3
        y_train_hat = None
        y_os = np.zeros((y.shape[1], y.shape[0], npoints))

        if self.forecast_ratio > 0:
            # add history coefficient to y_hat if forecasting
            y = np.concatenate([X[:, self.y_ind, :], y], axis=2)
        else:
            # prepare data for classification task
            y_train_hat = self.y_train_hat_embedded.transpose(1, 0, 2) * std_vec_y + mean_vec_y

        # standardize back and go back to the original shape
        y = y.transpose(1, 0, 2) * std_vec_y + mean_vec_y

        # prepare y for longitudinal variables: add 0 to the estimated bsplines coefficients
        y = np.concatenate([y, np.zeros((y.shape[0], y.shape[1], degree + 1))], axis=2)

        for i in range(len(self.y_ind)):

            # for categorical variables
            if self.var_modality[self.y_ind[i]] == 'c':
                # compute categories with the neareast neighbour method
                nearest_neighbors = NearestNeighbors(n_neighbors=1).fit(y_train_hat[i])

                # do not consider the last 0 coefficients
                _, indices = nearest_neighbors.kneighbors(y[i, :, :self.k_gcn_in])

                # recover original training set for y_i
                y_train = self.data[self.y_ind[i], self.train_ind, 0]

                # use estimated indexes to find the predicted categories
                y_os[i, :, :] = np.tile(y_train[indices.flatten()], (npoints, 1)).T

            # for scalar variable
            elif self.var_modality[self.y_ind[i]] == 's':

                # y_os[i, :, :] = np.tile(y[i, :, 0], (npoints, 1)).T  # if you used constant representation
                for obs in range(y.shape[1]):
                    # we take the average value of the estimated 'constant' as the scalar estimation
                    const_value = np.mean(splev(x=grid, tck=BSpline(basis_x, y[i, obs, :], degree), der=0))
                    y_os[i, obs, :] = np.tile(const_value, (npoints, 1)).T

            # for longitudinal variables
            else:
                for obs in range(y.shape[1]):
                    # bspline evaluation
                    y_os[i, obs, :] = splev(x=grid, tck=BSpline(basis_x, y[i, obs, :], degree), der=0)
                    # adjust extrema
                    y_os[i, obs, 0] = 0.5 * (y_os[i, obs, 1] + y_os[i, obs, 0])
                    y_os[i, obs, -1] = 0.5 * (y_os[i, obs, -1] + y_os[i, obs, -2])

        return y_os

    def compute_evaluation_metrics(self, y_hat=None, y_true=None):

        # initialize output dictionary
        rmse_dict, std_rmse_dict, accuracy_dict = {}, {}, {}

        for i in range(len(self.y_ind)):
            if self.var_modality[self.y_ind[i]] == 'c':
                # compute accuracy for categorical
                accuracy_dict[self.y_ind[i]] = np.mean(y_hat[i, :, 0] == y_true[i, :, 0])
            else:
                # compute standardized rmse for functions
                rmse = np.sqrt(np.mean((y_hat[i] - y_true[i]) ** 2))

                # TODO: decide which standardization for RMSE (std_dev o min_max)
                #   std_dev makes more sense since the training loss is computed on data standardized by std_dev
                # std_dev = np.max(y_true[i]) - np.min(y_true[i])

                std_dev = np.std(y_true[i])
                rmse_dict[self.y_ind[i]] = rmse
                std_rmse_dict[self.y_ind[i]] = rmse / std_dev if std_dev != 0 else 0

        return rmse_dict, std_rmse_dict, accuracy_dict

    def predict(self, new_data=None, y_true=None):

        """
        function to predict targets given new observation. If new data is None, it predicts the test observations.
        If y_true is given, it also computes the evaluation metrics.
        :param X: new observations to predict in the form of X (nfeat, nobs, npoints)
            For forecast, it contains just the past, so one has to compute embeddings using forecast_ratio=0.
            For regression/classification task, the targets node should be filled with 0
        :param y_true: true value of the predictions in the form of (nfeat[y_ind], nobs, npoints)
        """

        if new_data is None:
            if self.X_gcn_test is None:
                raise Exception('Test set has not been initialized, rerun preprocess_data with test_size > 0 or pass X as input')
            else:
                # prediction on the test set
                X = self.X_gcn_test.clone()
                y_true = self.data[self.y_ind]
                y_true = y_true[:, self.test_ind, :]
                basis_x = self.basis_gcn
                std_vec_y, mean_vec_y = self.std_vec_gcn[self.y_ind], self.mean_vec_gcn[self.y_ind]

        else:
            # find embeddings of the new data: forecast_ratio=0. even if the task is forecast
            X, basis_x, _, _ = self.compute_gcn_embeddings(k=self.k_gcn_in, data=new_data, forecast_ratio=0.)
            X, mean_vec, std_vec = self.standardize_X(X)
            X = torch.FloatTensor(X.transpose(1, 0, 2))

            if self.forecast_ratio > 0:
                # if forecast, we do not know the horizon std_vec_y, mean_vec_y: we use the saved ones
                std_vec_y, mean_vec_y = self.std_vec_gcn[self.y_ind], self.mean_vec_gcn[self.y_ind]
            else:
                # for regression/classification task, we use the std and mean computed on the new data
                std_vec_y, mean_vec_y = std_vec[self.y_ind], mean_vec[self.y_ind]

                # fill targets with 0
                X[:, self.y_ind, :] = 0

        # set model in evaluation mode
        self.model.eval()

        # define output matrix
        y_hat = np.zeros((X.shape[0], len(self.y_ind), self.k_gcn_out))
        rmse, std_rmse, accuracy = None, None, None

        # estimate y_hat_test in the coefficient form
        with torch.no_grad():
            for obs, x_obs in enumerate(X):
                y_hat[obs] = self.model(x_obs).detach()

        # go back to the original space
        y_hat_os = self.retrieve_original_form(y=y_hat, X=X, basis_x=basis_x, mean_vec_y=mean_vec_y, std_vec_y=std_vec_y)

        # compute evaluation metrics if we have y_true
        if y_true is not None:

            # We compute the evaluation criterion just on the horizon. Remember adjusted_start_forecast is 0 by default
            rmse, std_rmse, accuracy = self.compute_evaluation_metrics(y_hat_os[:, :, self.adjusted_start_forecast:], y_true[:, :, self.adjusted_start_forecast:])

            if self.verbose:
                print('')
                print('Prediction Evaluation Metrics')
                if len(rmse) > 0:
                    print(' RMSE:', np.round(sum(rmse.values()) / len(rmse), 4))
                    print(' STD RMSE:', np.round(sum(std_rmse.values()) / len(std_rmse), 4))
                if len(accuracy) > 0:
                    print(' ACCURACY:', sum(accuracy.values())/len(accuracy))

        if new_data is None:
            self.y_test_hat_os, self.y_test_hat_embedded = y_hat_os, y_hat
            self.rmse_test, self.std_rmse_test, self.accuracy_test = rmse, std_rmse, accuracy
        else:
            return OutputPrediction(y_hat_os, y_hat, rmse, std_rmse, accuracy)

    def print_prediction_metrics(self, rmse=None, std_rmse=None, accuracy=None, names_var=None):

        rmse = rmse or self.rmse_test
        std_rmse = std_rmse or self.std_rmse_test
        accuracy = accuracy or self.accuracy_test
        if names_var is None:
            names_var = range(0, self.data.shape[0])

        print('')
        if rmse is not None and len(rmse) > 0:
            col_names = ['RMSE', 'std RMSE']
            index_names = []
            rmse_values = []
            std_rmse_values = []
            for key, value in rmse.items():
                index_names.append(names_var[key])
                rmse_values.append(round(value, 3))
                std_rmse_values.append(round(std_rmse[key], 3))

            print(pd.DataFrame(np.array([rmse_values, std_rmse_values]).T, columns=col_names, index=index_names))

        if accuracy is not None and len(accuracy) > 0:
            col_names = ['accuracy']
            index_names = []
            accuracy_values = []
            for key, value in accuracy.items():
                index_names.append(names_var[key])
                accuracy_values.append(round(value, 3))

            print(pd.DataFrame(accuracy_values, columns=col_names, index=index_names))

    def plot_true_vs_predicted(self, target_to_plot=0, curves_to_plot=range(0,10), plot_test=1, names_var=None):

        """
        this function plot the predicted curves vs the true ones. You cannot pass external inputs
        """

        # target to plot is the position of the element in the y_ind list that we want to plot
        y_i = target_to_plot

        if plot_test:
            if self.X_gcn_test is None:
                raise Exception('Test size is 0: plot train instead (set plot_test=0)')
            if self.y_test_hat_os is None:
                raise Exception('Prediction on the test has not done yet: run predict() with no inputs')
            indx = self.test_ind
            y_hat = self.y_test_hat_os

        else:  # plot train
            indx = self.train_ind
            y_hat = self.y_train_hat_embedded
            # y_hat = self.y_gcn_train.numpy()
            y_hat = self.retrieve_original_form(y_hat, self.X_gcn_train)

        # identify observed y
        y = self.data[self.y_ind]
        y_true = y[:, indx, :]

        # set y limits
        y_lim_min = 0.9 * np.min(y_true[y_i, curves_to_plot, :])
        y_lim_max = 1.1 * np.max(y_true[y_i, curves_to_plot, :])
        y_lim_min = y_lim_min if y_lim_min != 0 else 0 - 0.1 * y_lim_max
        y_lim_max = y_lim_max if y_lim_max != 0 else 0 + 0.1 * y_lim_min

        # define plot title
        if names_var is None:
            main = 'observed vs predicted (:)'
        else:
            main = names_var[self.y_ind[y_i]] + ': observed vs predicted (:)'

        # plot all the selected set of observed and predicted curves
        plt.plot(y_true[y_i, curves_to_plot, :].T, lw=1)
        plt.gca().set_prop_cycle(None)
        plt.plot(y_hat[y_i, curves_to_plot, :].T, '-.')
        plt.ylim(y_lim_min, y_lim_max)
        plt.title(main)
        plt.show()

        # plot the observed and predicted curves 1 by 1
        if self.forecast_ratio > 0:
            grid = np.arange(y_true.shape[2])
            start_forecasting = self.adjusted_start_forecast
            grid_horizon = grid[start_forecasting:]
            grid_hist = grid[:start_forecasting]
            for c_i in curves_to_plot:
                plt.plot(grid_horizon, y_true[y_i, c_i, start_forecasting:].T, lw=2, color='tomato')
                plt.plot(grid_horizon, y_hat[y_i, c_i, start_forecasting:].T, '-.', lw=2, color='royalblue')
                plt.plot(grid_hist, y_true[y_i, c_i, :start_forecasting].T, lw=1, color='tomato')
                plt.ylim(y_lim_min, y_lim_max)
                plt.title(main)
                plt.show()

        else:
            for c_i in curves_to_plot:
                plt.plot(y_true[y_i, c_i, :].T, lw=1, color='tomato')
                plt.plot(y_hat[y_i, c_i, :].T, '-.', lw=1.5, color='royalblue')
                plt.ylim(y_lim_min, y_lim_max)
                plt.title(main)
                plt.show()
