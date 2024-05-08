
"""

code to test fgcn on synthetic data

"""

import torch
import numpy as np
from fungcn.fgcn.solver_fgcn import FunGCN
from fungcn.fgcn.generate_sim_fgcn import GenerateSimFGCN
import os
from pathlib import Path
path = os.path.dirname(Path(__file__))


def set_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':

    # set seed
    seed = 56

    # choose task
    task = 'regression'

    # sim parameters
    nobs = 300
    nvar = 500

    # simulation parameters
    nfun0 = 4  # number of correlated functional variables
    ncat0 = 4  # number of correlated categorical variables
    cat0_levels = [2, 2, 3, 4]  # number of levels of each cat0
    nscalar0 = 2  # number of correlated scalar variables

    ntimes = 100  # number of evaluations of each curve
    fun_ratio = 0.6  # ratio of fun var not correlated
    cat_ratio = 0.2  # ratio of categorical var not correlated
    scalar_ratio = 0.2  # ratio of scalar var not correlated

    # fgcn parameters
    if task == 'classification':
        y_ind = [4]  # y_ind_list = [[4], [5], [6], [7]]
        forecast_ratio = 0.
        pruning = 0.7
        k_gcn = 5
        lr = 1e-4  # 5e-4
    elif task == 'regression':
        y_ind = [0]  # y_ind_list = [[0], [1], [2], [3]]
        forecast_ratio = 0.
        pruning = 0.7
        k_gcn = 10
        lr = 5e-5  # 1e-4
    else:
        y_ind = [0]  # y_ind_list = [[0], [1], [2], [3]]
        forecast_ratio = 0.3
        pruning = 0.7
        k_gcn = 20
        lr = 1e-4

    max_selected = 5
    k_graph = 3
    nhid = [32, 32]  # [128, 128] [128, 64] [64, 64] [64, 32] [32, 32]
    epochs = 50
    batch_size = 1
    dropout = 0.
    kernel_size = 0
    patience = 5
    min_delta = 0

    # train, val, and test
    val_size = 0.2
    test_size = 0.25

    # save graph
    save_graph_name = None
    # save_graph_name = 'kg_share/test1'

    # create data
    gen_sim = GenerateSimFGCN(nvar, nobs, ntimes, nfun0, ncat0, cat0_levels, nscalar0, fun_ratio, cat_ratio,
                              scalar_ratio, min_corr=0.9, max_corr=1, mult_factor=10, seed=seed)
    X, var_modality = gen_sim.get_design_matrix()
    # np.savetxt('kg_share/modalities.txt', var_modality, fmt='%s', delimiter=',')

    # initialize functional gcn object
    fun_gcn = FunGCN(data=X, y_ind=y_ind, var_modality=var_modality, verbose=1)

    # preprocess the data
    fun_gcn.preprocess_data(k_graph=k_graph, k_gcn=k_gcn, forecast_ratio=forecast_ratio,
                            test_size=test_size, val_size=val_size, random_state=seed)

    # create graph
    fun_gcn.graph_estimation(max_selected=max_selected, graph_path=save_graph_name)

    # initialize GCN model
    fun_gcn.initialize_gcn_model(pruning=pruning, nhid=nhid, dropout=dropout, kernel_size=kernel_size)

    # train GCN model
    fun_gcn.train_model(lr=lr, epochs=epochs, batch_size=batch_size, patience=patience, min_delta=min_delta)

    # predict on test set
    fun_gcn.predict()

    # print prediction metrics
    fun_gcn.print_prediction_metrics()

    # plot a sample of the estimated curve
    fun_gcn.plot_true_vs_predicted(target_to_plot=0, curves_to_plot=range(0, 30), plot_test=1)


