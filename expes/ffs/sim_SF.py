"""code to run the SF fungcn on synthetic data"""


import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from fungcn.ffs.solver_path import FASTEN
from fungcn.ffs.enum_classes import RegressionType, SelectionCriteria, AdaptiveScheme
from fungcn.ffs.auxiliary_functions_SF import AuxiliaryFunctionsSF
from fungcn.ffs.generate_sim_SF import GenerateSimSF

if __name__ == '__main__':

    # seed = np.random.randint(1, 2**30, 1)
    # np.random.seed(seed)
    seed = 54

    # ------------------------ #
    #  choose simulation type  #
    # ------------------------ #

    regression_type = RegressionType.SF  # FF, FS, SF
    gen_sim = GenerateSimSF(seed)
    af = AuxiliaryFunctionsSF()

    selection_criterion = SelectionCriteria.CV  # CV, GCV, or EBIC
    n_folds = 5  # number of folds if cv is performed
    adaptive_scheme = AdaptiveScheme.SOFT  # type of adaptive scheme: FULL, SOFT, NONE

    easy_x = True  # if the features are easy or complex to estimate
    relaxed_criteria = True  # if True a linear regression is fitted on the features to select the best lambda
    relaxed_estimates = True  # if True a linear regression is fitted on the features before returning them

    # --------------------------- #
    #  set simulation parameters  #
    # --------------------------- #

    m = 300  # number of samples
    n = 500  # number of features
    not0 = 10  # number of non 0 features

    domain = np.array([0, 1])  # domains of the curves
    neval = 100  # number of points to construct the true predictors and the response

    mu_A = 0  # mean of features
    sd_A = 1  # standard deviation of the A Matern covariance
    l_A = 0.25  # range parameter of A Matern covariance
    nu_A = 3.5  # smoothness of A Matern covariance

    mu_eps = 0  # mean of errors
    snr = 100  # signal to noise ratio to determine sd_eps
    l_eps = 0.25  # range parameter of eps Matern covariance
    nu_eps = 1.5  # smoothness of eps Matern covariance

    # ----------------------- #
    #  set fungcn parameters  #
    # ----------------------- #

    k = None  # number of FPC scores, if None automatically selected

    # c_lam_vec = 0.8  # if we chose to run for just one value of lam1 = lam1 = c_lam * lam1_max
    c_lam_vec = np.geomspace(1, 0.01, num=100)  # grid of lam1 to explore, lam1 = c_lam * lam1_max
    c_lam_vec_adaptive = np.geomspace(1, 0.0001, num=50)

    max_selected = max(50, 2 * not0)  # max number of selected features
    # max_selected = 100
    check_selection_criterion = False  # if True and the selection criterion has a discontinuity, we stop the search

    wgts = 1  # individual penalty weights
    alpha = 0.2  # lam2 = (1-alpha) * c_lam * lam1_max

    sgm = not0 / n  # starting value of sigma
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
        mu_x = 0  # mean of the true predictors
        sd_x = 1  # standard deviation of the x Matern covariance
        l_x = 0.25  # range parameter of x Matern covariance
        nu_x = 3.5  # smoothness of x Matern covariance
    else:
        # Difficult x
        mu_x = 0  # mean of the true predictors
        sd_x = 2  # standard deviation of the x Matern covariance
        l_x = 0.25  # range parameter of x Matern covariance
        nu_x = 2.5  # smoothness of x Matern covariance

    # create equispaced grid where the curves are evaluated at
    grid = np.linspace(domain[0], domain[1], neval)

    # generate design matrix A
    A = gen_sim.generate_A(n, m, grid, mu_A, sd_A, l_A, nu_A)

    # generate coefficient matrix x
    x_true = gen_sim.generate_x(not0, grid, sd_x, mu_x, l_x, nu_x)

    # compute errors and response
    b, eps = gen_sim.compute_b_plus_eps(A, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps)

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
    out_path_SF = solver.solver(
        regression_type=regression_type,
        A=A, b=b, k=k, wgts=wgts,
        selection_criterion=selection_criterion, n_folds=n_folds,
        adaptive_scheme=adaptive_scheme,
        coefficients_form=False, x_basis=None, b_std=None,
        c_lam_vec=c_lam_vec, c_lam_vec_adaptive=c_lam_vec_adaptive,
        max_selected=max_selected, check_selection_criterion=check_selection_criterion,
        alpha=alpha, lam1_max=None,
        x0=None, y0=None, z0=None, Aty0=None,
        relaxed_criteria=relaxed_criteria, relaxed_estimates=relaxed_estimates,
        sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
        tol_nwt=tol_nwt, tol_dal=tol_dal,
        maxiter_nwt=maxiter_nwt, maxiter_dal=maxiter_dal,
        use_cg=use_cg, r_exact=r_exact,
        plot=plot, print_lev=print_lev)

    # ------------------ #
    #  model evaluation  #
    # ------------------ #

    out_SF = out_path_SF.best_model

    # MSE false positive and false negatives
    indx = out_SF.indx
    r = out_SF.r
    pos_curves = np.where(indx * 1 > 0)[0]
    false_negatives = np.sum(1 - indx[0:not0])
    false_positives = np.sum(indx[not0:])
    true_positive = np.sum(indx[0:not0])
    x_hat_true_positive = out_SF.x_curves[0:true_positive, :]
    x_true_sub = x_true[indx[0:not0], :]

    # MSE for y
    xj_curves = out_SF.x_curves
    AJ = A[indx, :, :].transpose(1, 0, 2).reshape(m, r * neval)
    xj_curves_expanded = (np.eye(neval) * xj_curves.reshape(r, 1, neval)).reshape(r * neval, neval)
    b_hat = np.sum(AJ @ xj_curves_expanded, axis=1)
    MSEy = np.mean(LA.norm(b - b_hat, axis=0) ** 2 / LA.norm(b, axis=0) ** 2)

    # MSE for x
    resx = x_hat_true_positive - x_true_sub
    MSEx = np.mean(LA.norm(resx, axis=1) ** 2 / LA.norm(x_true_sub, axis=1) ** 2)

    print('MSEy = %.4f' % MSEy)
    print('MSEx = %.4f' % MSEx)
    print('false negatives = ', false_negatives)
    print('false positives = ', false_positives)

    # ----------------------- #
    #  plot estimated curves  #
    # ----------------------- #

    if plot:
        # plot b and b_hat
        plt.scatter(range(m), b, marker=".")
        plt.gca().set_prop_cycle(None)
        plt.scatter(range(m), b_hat, marker=".", color='salmon')
        plt.title('observed (blue) vs fitted (salmon) responses')
        plt.show()

        # plot x and x_hat all together
        plt.plot(grid, out_SF.x_curves[0:, :].T, lw=1)
        plt.gca().set_prop_cycle(None)
        plt.plot(grid, x_true[0:, :].T, '--')
        plt.title('true (-) vs estimated (--) x')
        plt.show()

        # # plot x and x_hat one at a time
        # ind_curve = 0
        # for i in range(not0):
        #     if indx[i]:
        #         plt.plot(grid, out_SF.x_curves[ind_curve, :].T, '--', color='green')
        #         ind_curve += 1
        #     plt.plot(grid, x_true[i, :].T)
        #     plt.title('true (-) vs estimated (--) x')
        #     plt.show()
        #
        # for i in range(r):
        #     if pos_curves[i] > not0:
        #         plt.plot(grid, out_SF.x_curves[i, :].T, '--', color='green')
        #         plt.title('true (-) vs estimated (--) x')
        #         plt.show()


