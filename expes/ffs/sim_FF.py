"""code to run the FF fungcn on synthetic data."""


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from fungcn.ffs.solver_path import FASTEN
from fungcn.ffs.enum_classes import RegressionType, SelectionCriteria, AdaptiveScheme, FPCFeatures
from fungcn.ffs.auxiliary_functions_FF import AuxiliaryFunctionsFF
from fungcn.ffs.generate_sim_FF import GenerateSimFF


if __name__ == '__main__':

    # seed = np.random.randint(1, 2**30, 1)
    # np.random.seed(seed)
    seed = 10

    # ------------------------ #
    #  choose simulation type  #
    # ------------------------ #

    regression_type = RegressionType.FF  # FF, FS, SF
    gen_sim = GenerateSimFF(seed)
    af = AuxiliaryFunctionsFF()
    fpc_features = FPCFeatures.response  # features, response

    selection_criterion = SelectionCriteria.CV  # CV, GCV, or EBIC
    n_folds = 5  # number of folds if cv is performed
    adaptive_scheme = AdaptiveScheme.FULL  # type of adaptive scheme: FULL, SOFT, NONE

    easy_x = True  # if the features are easy or complex to estimate
    relaxed_criteria = True  # if True a linear regression is fitted on the features to select the best lambda
    relaxed_estimates = True  # if True a linear regression is fitted on the features before returning them
    select_k_estimation = False  # if True we allow k to change k for estimation (chosen based on CV)

    # ----------------------------- #
    #  other simulation parameters  #
    # ----------------------------- #

    m = 300  # number of samples
    n = 500  # number of features
    not0 = 5  # number of non 0 features

    domain = np.array([0, 1])  # domains of the curves
    neval = 100  # number of points to construct the true predictors and the response

    mu_A = 0  # mean of features
    sd_A = 1  # standard deviation of the A Matern covariance
    l_A = 0.25  # range parameter of A Matern covariance
    nu_A = 3.5  # smoothness of A Matern covariance

    mu_eps = 0  # mean of errors
    snr = 10  # signal to noise ratio to determine sd_eps
    l_eps = 0.25  # range parameter of eps Matern covariance
    nu_eps = 2.5  # smoothness of eps Matern covariance

    # ----------------------- #
    #  set fungcn parameters  #
    # ----------------------- #

    k = None  # number of FPC scores, if None automatically selected 0.048 0.039

    # c_lam_vec = 0.3  # if we chose to run for just one value of lam1 = lam1 = c_lam * lam1_max
    c_lam_vec = np.geomspace(1, 0.01, num=100)  # grid of lam1 to explore, lam1 = c_lam * lam1_max
    c_lam_vec_adaptive = np.geomspace(1, 0.0001, num=50)  # grid of lam1 to explore in the adaptive FULL path

    max_selected = 50  # max number of selected features
    # max_selected = 100
    check_selection_criterion = True  # if True and the selection criterion has a discontinuity, we stop the search

    wgts = np.ones((n, 1))  # individual penalty weights
    alpha = 0.2  # lam2 = (1-alpha) * c_lam * lam1_max

    sgm = 0.001  # starting value of sigma
    sgm_increase = 5  # sigma increasing factor
    sgm_change = 1  # number of iteration that we wait before increasing sigma

    use_cg = True  # decide if you want to use conjugate gradient
    r_exact = 2000  # number of features such that we start using the exact method

    maxiter_nwt = 40  # nwt max iterations
    maxiter_dal = 100  # dal max iterations
    tol_nwt = 1e-6  # nwt tolerance
    tol_dal = 1e-6  # dal tolerance

    plot = True  # plot selection criteria
    print_lev = 2  # decide level of printing

    # ------------------ #
    #  create variables  #
    # ------------------ #

    if easy_x:
        # Easy x
        x_npeaks_set = np.array([1])  # number of possible peaks of the features
        x_sd_min = 0.2  # minimum sd of the features peaks
        x_sd_max = 0.3  # max sd of the features peaks
        x_range = np.array([-1, 1])  # max and min values of the x
    else:
        # Difficult x
        x_npeaks_set = np.array([2, 3])  # number of possible peaks of the features
        x_sd_min = 0.01  # minimum sd of the features peaks
        x_sd_max = 0.15  # max sd of the features peaks
        x_range = np.array([-1, 1])  # max and min values of the x

    # create equispaced grid where the curves are evaluated at
    grid = np.linspace(domain[0], domain[1], neval)
    grid_expanded = np.outer(grid, np.ones(neval))

    # generate design matrix A
    A, A_test = gen_sim.generate_A(n, m, grid, mu_A, sd_A, l_A, nu_A, test=True)

    # generate coefficient matrix x
    x_true = gen_sim.generate_x(not0, grid, x_npeaks_set, x_sd_min, x_sd_max, x_range)

    # compute errors and response
    b, eps = gen_sim.compute_b_plus_eps(A, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps)
    b_test, eps_test = gen_sim.compute_b_plus_eps(A_test, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps)

    # --------------- #
    #  standardize A  #
    # --------------- #
    print('  * standardizing A')
    A = af.standardize_design_matrix(A)

    # --------------- #
    #  launch fungcn  #
    # --------------- #
    print('')
    print('  * start fgen')
    print('  * sgm = %.4f' % sgm)

    # -------- #
    #  FASTEN  #
    # -------- #

    solver = FASTEN()
    out_path_FF = solver.solver(
        regression_type=regression_type,
        A=A, b=b, k=k, wgts=wgts,
        selection_criterion=selection_criterion, n_folds=n_folds,
        adaptive_scheme=adaptive_scheme, fpc_features=fpc_features,
        coefficients_form=False, x_basis=None, b_std=None,
        c_lam_vec=c_lam_vec, c_lam_vec_adaptive=c_lam_vec_adaptive,
        max_selected=max_selected, check_selection_criterion=check_selection_criterion,
        alpha=alpha, lam1_max=None,
        x0=None, y0=None, z0=None, Aty0=None,
        relaxed_criteria=relaxed_criteria, relaxed_estimates=relaxed_estimates,
        select_k_estimation=select_k_estimation,
        sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
        tol_nwt=tol_nwt, tol_dal=tol_dal,
        maxiter_nwt=maxiter_nwt, maxiter_dal=maxiter_dal,
        use_cg=use_cg, r_exact=r_exact,
        plot=plot, print_lev=print_lev)

    # ------------------ #
    #  model evaluation  #
    # ------------------ #

    out_FF = out_path_FF.best_model

    # compute false positive and false negatives
    indx = out_FF.indx
    r = out_FF.r
    pos_curves = np.where(indx * 1 > 0)[0]
    false_negatives = np.sum(1-indx[0:not0])
    false_positives = np.sum(indx[not0:])
    true_positive = np.sum(indx[0:not0])
    x_hat_true_positive = out_FF.x_curves[0:true_positive, :, :]
    x_true_sub = x_true[indx[0:not0], :, :]

    # MSE for y
    xj_curves = out_FF.x_curves
    AJ = A[indx, :, :].transpose(1, 0, 2).reshape(m, r * neval)
    b_hat = AJ @ xj_curves.reshape(r * neval, neval) + b.mean(axis=0)
    MSEy = np.mean(LA.norm(b - b_hat, axis=1) ** 2 / LA.norm(b, axis=1) ** 2)

    # MSE for x
    resx = (x_hat_true_positive - x_true_sub).reshape(true_positive, neval ** 2)
    MSEx = np.mean(LA.norm(resx, axis=1) ** 2 / LA.norm(x_true_sub.reshape(true_positive, neval ** 2), axis=1) ** 2)

    # MSE for y out-of-sample
    m_test = b_test.shape[0]
    xj_curves = out_FF.x_curves
    AJ = A_test[indx, :, :].transpose(1, 0, 2).reshape(m_test, r * neval)
    b_hat_out = AJ @ xj_curves.reshape(r * neval, neval)
    MSEy_out = np.mean(LA.norm(b_test - b_hat_out, axis=1) ** 2 / LA.norm(b_test, axis=1) ** 2)

    print('MSEy = %f' % MSEy)
    print('MSEy_out = %f' % MSEy_out)
    print('MSEx = %f' % MSEx)
    print('false negatives = ', false_negatives)
    print('false positives = ', false_positives)
    # print('features order of entry:')
    # print(np.argsort(-out_path_FF.c_lam_entry_value[out_path_FF.c_lam_entry_value > 0]))

    # ------------ #
    #  some plots  #
    # ------------ #

    if plot:
        # plot b and b_hat (first 5 response)
        plt.plot(grid, b[0:5, :].T, lw=1)
        plt.gca().set_prop_cycle(None)
        plt.plot(grid, b_hat[0:5, :].T, '--')
        plt.title('observed (-) vs fitted (--) responses')
        plt.show()

        # plot b and b_hat (first 5 response)
        ind_curve = 0
        for i in range(not0):
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            ax.set_zlim(x_range)
            if indx[i]:
                ax.plot_wireframe(grid_expanded, grid_expanded.T, out_FF.x_curves[ind_curve, :, :],
                                  color='salmon', alpha=0.3)
                ax.set_title('estimated surface')
                ind_curve += 1

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            ax.set_zlim(x_range)
            ax.plot_wireframe(grid_expanded, grid_expanded.T, x_true[i, :, :],
                              alpha=0.3)
            ax.set_title('true surface')
            plt.show()

        for i in range(r):
            if pos_curves[i] > not0:
                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1, projection='3d')
                ax.set_zlim(x_range)
                ax.plot_wireframe(grid_expanded, grid_expanded.T, out_FF.x_curves[i, :, :],
                                  color='salmon', alpha=0.3)
                ax.set_title('estimated surface')
                ax = fig.add_subplot(1, 2, 2, projection='3d')
                ax.set_zlim(x_range)
                plt.show()



