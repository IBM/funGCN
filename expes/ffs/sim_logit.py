"""code to run the logisitc fungcn on synthetic data"""


import numpy as np
from numpy import linalg as LA
from scipy.special import expit
from fungcn.ffs.solver_path_logit import FASTENLOGIT
from fungcn.ffs.enum_classes import RegressionType, SelectionCriteria, AdaptiveScheme
from fungcn.ffs.auxiliary_functions_logit import AuxiliaryFunctionsLogit
from fungcn.ffs.generate_sim_logit import GenerateSimLogit
import os
print(os.getcwd())

if __name__ == '__main__':

    seed = 54
    # seed = 218359804
    # seed = np.random.randint(1, 2 ** 30, 1)
    gen_sim = GenerateSimLogit(seed)
    print(' seed = ', seed)

    # simulation type
    regression_type = RegressionType.Logit  # FF, FS, SC
    selection_criterion = SelectionCriteria.CV  # CV or EBIC
    af = AuxiliaryFunctionsLogit()
    n_folds = 5  # number of folds if cv is performed
    adaptive_scheme = AdaptiveScheme.SOFT  # type of adaptive scheme: FULL, SOFT, NONE
    easy_x = True  # if the features are easy or complex to estimate

    # simulation parameters
    m = 500  # number of samples
    n = 800  # number of features
    not0 = 20  # number of non 0 features
    domain = np.array([0, 1])  # domains of the curves
    neval = 100  # number of points to construct the true predictors and the response
    grid = np.linspace(domain[0], domain[1], neval)

    # parameters for functional features
    mu_A = 0  # mean of features
    sd_A = 1  # standard deviation of the A Matern covariance
    l_A = 0.25  # range parameter of A Matern covariance
    nu_A = 3.5  # smoothness of A Matern covariance
    mu_eps = 0  # mean of errors
    l_eps = 0.25  # range parameter of eps Matern covariance
    nu_eps = 1.5  # smoothness of eps Matern covariance

    # sigma to noise ratio
    snr = 100

    # parameters for curves coefficients
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

    # fungcn parameters
    k = 5  # number of FPC scores, if None automatically selected
    # c_lam_vec = 0.8  # if we chose to run for just one value of lam1 = lam1 = c_lam * lam1_max
    c_lam_vec = np.geomspace(1, 0.05, num=100)  # grid of lam1 to explore, lam1 = c_lam * lam1_max
    c_lam_vec_adaptive = np.geomspace(1, 0.001, num=50)
    max_selected = max(1000, 2 * not0)  # max number of selected features
    check_selection_criterion = True  # if True and the selection criterion has a discontinuity, we stop the search
    wgts = 1  # individual penalty weights
    alpha = 0.2  # lam2 = (1-alpha) * c_lam * lam1_max
    use_cg = True  # decide if you want to use conjugate gradient
    r_exact = 2000  # number of features such that we start using the exact method
    maxiter_nwt = 40  # nwt max iterations
    maxiter_dal = 100  # dal max iterations
    tol_nwt = 1e-4  # nwt tolerance
    tol_dal = 1e-4  # dal tolerance
    plot = False  # plot selection criteria
    print_lev = 1  # decide level of printing

    # generate design matrix A
    A, A_test = gen_sim.generate_A(n, m, grid, mu_A, sd_A, l_A, nu_A, test=True)

    # generate coefficient matrix x
    x_true = gen_sim.generate_x(not0, grid, sd_x, mu_x, l_x, nu_x)

    # compute errors and response
    b, eps = gen_sim.compute_b_plus_eps(A, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps)
    b_test, eps_test = gen_sim.compute_b_plus_eps(A_test, x_true, not0, grid, snr, mu_eps, l_eps, nu_eps)

    # standardize A
    A = af.standardize_design_matrix(A)
    A_test = af.standardize_design_matrix(A_test)

    # FASTEN
    solver = FASTENLOGIT()
    out_path_logit = solver.solver(
        regression_type=regression_type,
        A=A, b=b, k=k, wgts=wgts,
        selection_criterion=selection_criterion, n_folds=n_folds,
        adaptive_scheme=adaptive_scheme,
        coefficients_form=False, x_basis=None,
        c_lam_vec=c_lam_vec, c_lam_vec_adaptive=c_lam_vec_adaptive,
        max_selected=max_selected, check_selection_criterion=check_selection_criterion,
        alpha=alpha, lam1_max=None,
        x0=None, y0=None, z0=None, Aty0=None,
        sgm=None, sgm_increase=None, sgm_change=1,
        tol_nwt=tol_nwt, tol_dal=tol_dal,
        maxiter_nwt=maxiter_nwt, maxiter_dal=maxiter_dal,
        use_cg=use_cg, r_exact=r_exact,
        plot=plot, print_lev=print_lev)

    convergence = out_path_logit.best_model.convergence

    if not convergence:
        MSEy, MSEy_out = np.NAN, np.NAN
        false_negatives, false_positives, recall, precision = np.NAN, np.NAN, np.NAN, np.NAN
        time = np.NAN

    else:

        # Model evaluation
        time = np.round(out_path_logit.time_total, 3)
        out = out_path_logit.best_model

        # compute false positive and false negatives
        indx = out.indx
        r = out.r
        pos_curves = np.where(indx * 1 > 0)[0]
        false_negatives = np.sum(1 - indx[0:not0])
        false_positives = np.sum(indx[not0:])
        true_positive = np.sum(indx[0:not0])
        x_hat_true_positive = out.x_curves[0:true_positive, :]
        x_true_sub = x_true[indx[0:not0], :]
        precision = true_positive / (true_positive + false_positives)
        recall = true_positive / (true_positive + false_negatives)

        # MSE for y
        xj_curves = out.x_curves
        AJ = A[indx, :, :].transpose(1, 0, 2).reshape(m, r * neval)
        b_hat = np.round(expit(AJ @ xj_curves.ravel()))
        b[b < 1] = 0
        MSEy = 1 - np.mean(np.abs(b - b_hat))

        # MSE for x
        resx = x_hat_true_positive - x_true_sub
        MSEx = np.mean(LA.norm(resx, axis=1) ** 2 / LA.norm(x_true_sub, axis=1) ** 2)

        # MSE for y out-of-sample
        m_test = b_test.shape[0]
        AJ_test = A_test[indx, :, :].transpose(1, 0, 2).reshape(m_test, r * neval)
        b_hat_out = np.round(expit(AJ_test @ xj_curves.ravel()))
        b_test[b_test < 1] = 0
        MSEy_out = 1 - np.mean(np.abs(b_test - b_hat_out))

        # printing results
        print('')
        print(' MODEL EVALUATION')
        print('   - prediction accuracy train = %f' % MSEy)
        print('   - prediction accuracy test = %f' % MSEy_out)
        print('   - MSEx = %f' % MSEx)
        print('   - false negatives = ', false_negatives)
        print('   - false positives = ', false_positives)
        print('   - recall = ', recall)
        print('   - precision = ', precision)
        print('   - time = ', time)

    # # plot x and x_hat all together
    # # plt.plot(grid, x_true[0:, :].T, lw=1)
    # # plt.gca().set_prop_cycle(None)
    # plt.plot(grid, xj_curves[0:, :].T, '--')
    # plt.title('true (-) vs estimated (--) x')
    # plt.show()


