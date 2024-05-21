""" class FASTEN
        fungcn for different values of c_lam and different regression type

    solver_core: solve the problem for one value of c_lam (lambda)
    fungcn: call several times solver_core -- one for each value of lambda -- and performs adaptive step

    INPUT PARAMETERS:
    --------------------------------------------------------------------------------------------------------------------
    :param regression_type: RegressionType object
        FS: function on scalar
        SF: scalar on function
        FF: function on function
    :param A: design matrix
        FS: np.array((m, n))
        FF, SF: np.array((n, m, neval))
    :param b: response matrix/vector
        FS, FF: np.array((m, neval))
        SF: np.array(m, )
    :param k: number of basis function. If k is not passed to the function, it is selected automatically such that:
        FF: more than 90% of response variability
        FS: min(5, more than 99% of response variability)
        SF: k = 5
    :param wgts: individual weights for the penalty. 1 (default) or np.array with shape (n, 1)
    :param use_bFPC_for_A: just for the FF model, if True FPC of b are used also for A. If False each feature uses its
        own FPC
    :param selection_criterion: an object of class SelectionCriteria, it can be CV, GCV, EBIC.
        The output of the fungcn will contain the best model according to the chosen criterion.
    :param n_folds: if selection_criterion is CV, number of folds to compute it. Default = 10
    :param adaptive_scheme: an object of class AdaptiveScheme. It can be NONE, SOFT, FULL.
        NONE: no adaptive step is performed
        SOFT (default): just one adaptive iteration is performed
        FULL: a new path is investigated starting from the weights obtained at the previous path
    ::param coefficients_form: If TRUE the inputs A and b are already in the coefficients form and x_basis MUST be given.
        Deafult is False. The coefficient form has be obtained as follows:
        (remember, if g_basis and f_basis orthogonal, then g_basis.T @ f_basis = I)
            For b - with b_scores = b @ b_basis, we have:
                FS: b_coeff = b_scores @ b_basis.T @ x_basis
                SF: b_coeff = b
                FF: b_coeff = b_scores @ b_basis.T @ x_basis2
            For A - with A_scores = A @ A_basis, we have:
                FS: A_coeff = A
                SF: A_coeff = A_scores @ A_basis.T @ x_basis
                FF: A_coeff = A_scores @ A_basis.T @ x_basis1
        If FALSE the coefficients form is automatically computed using the following basis
            FS: b_basis = x_basis = FPC of b
            SF: A_basis (feat j) = x_basis (feat j) = FPC of feature j
            FF: A_basis = x_basis1 = x_basis2 = FPC of b
    :param x_basis: Default is False. if coefficient_form = TRUE, you have to pass the basis function of x.
        If you use the same basis for all the features then:
            x_basis: (neval x k).
        If you use different basis for each feature, then:
            function-on-function: x_basis is a (2, n, neval, k) tensor with:
                first dimension: x_basis1 and x_basis2, second dimension: basis of each features, and it has to be:
                x_basis1 = A_basis, x_basis2 = b_basis
            All other models: x_basis is an (n, neval, k) tensor
    :param b_std if coefficient form is True, you have to standardize b before computing coefficients and pass the
        standard deviation as input: you need it to reconstruct the final model coefficients (curves or surfaces) and
        the predicted responses
    :param c_lam_vec: np.array to determine the path of lambdas. Default: np.geomspace(1, 0.01, num=100)
        If just one number, a single run is performed and the output is in best_models.single_run
        Different regression model and different alpha, may requires longer/shorter grid. We reccomend the
        user to investigate a long grid and maybe use max_selected to stop the search.
    :param c_lam_vec_adaptive: np.array to determine the path of lambdas in the adaptive step.
        Used if adaptive_scheme = FULL. DEfault: np.geomspace(1, 0.0001, num=50)
    :param max_selected: if given, the algorithm stops when a number of features > max_selected is selected
        Default is None
    :param check_selection_criterion: if True and the selection criterion has  a strong discontinuity,
        we stop the search. If max selected is None or bigger than 80, we suggest to set
        check_selection_criterion = True. Default is False.
    :param alpha: we have lam1 = alpha * c_lam * lam1_max, lam2 = (1 - alpha) * c_lam * lam1_max
    :param lam1_max: smallest values of lam1 that selects 0 features. If it is None, it is computed inside the function
    :param x0: initial value for the variable of the primal problem -- vector 0 if not given
        FF: np.array((n * k, k))
        all the others: np.array((n, k))
    :param y0: initial value fot the first variable of the dual problem -- vector of 0 if not given
        FF, FS: np.array((m, k))
        SF: np.array((m))
    :param z0: initial value for the second variable of the dual problem -- vector of 0 if not given
        FF: np.array((n * k, k))
        all the others: np.array((n, k))
    :param Aty0: A.T @ y0
    :param select_k_estimation: used just if regression_type = FF. Default is True.
        If true, k can change after the selection and before the surfaces' estimation (chosen based on CV)
    :param relaxed_criteria: if True a linear regression is fitted on the selected features before computing
        the selection criterion. Default is True
    :param relaxed_estimates: if True a linear regression is fitted on the features to produce the final estimates.
        Default is True. We suggest to set relaxed_criteria = relaxed_estimates = True
        If adaptive_scheme = FULL, relaxed_estimates and relaxed_criteria are forced to be False
        (the weights already are a relaxation of the estimates)
    WARNING: if relaxed_criteria or relaxed_estimate = True, for some CV folds LA.solve(AJ.T @ AJ, AJ.T @ b)
        may be singular: relaxed_criteria or relaxed_estimate or do not use CV
    :param sgm: starting value of the augmented lagrangian parameter sigma. Default is 5e-3
    :param sgm_increase: increasing factor of sigma.  Default is 5.
    :param sgm_change: we increase sgm -- sgm *= sgm_increase -- every sgm_change iterations. Default is 1
    :param tol_nwt: tolerance for the nwt algorithm. Default is 1e-5
    :param tol_dal: global tolerance of the dal algorithm. Default is 1e-6
    :param maxiter_nwt: maximum number of iterations for nwt. Default is 50
    :param maxiter_dal: maximum number of global iterations. Default is 100
    :param use_cg: True/False. If true, the conjugate gradient method is used to find the direction of the nwt.
        Dfault is False
    :param r_exact: number of features such that we start using the exact method. Default is 2e4
    :param plot: True/False. If true a plot of r, gcv, extended bic and cv (if cv == True) is displayed.
    :param print_lev: different level of printing (0, 1, 2, 3, 4)
    --------------------------------------------------------------------------------------------------------------------

    OUTPUT: OutputPath object with the following attributes
    --------------------------------------------------------------------------------------------------------------------
    :attribute best_model:
        an OutputSolver object containing the best model according to the chosen selection criterion. It has the
        following attributes:
            ------------------------------------------------------------------------------------------------------------
            :attribute x_curves: curves computed just for the not 0 estimated coefficients
                FF: x_basis1 @ x_scores @ x_basis2.T, np.array((r, neval, neval))
                FS, SF: x_curves = x_scores @ x_basis.T,  np.array((r, neval))
                They are returned as None by this function then and computed for the best models in path_solver
            :attribute x_coeffs: standardized estimated coefficients. They are estimated based on:
                (b - b.mean) / b.std()
                FS, SF: np.array((n, k))
                FF: np.array((n, k, k))
            :attribute x_basis: they are returned as None by this function then inserted in path_solver
            :attribute b_coeffs: coefficient form of b
                FS, FF: np.array((m, k))
                SF: np.array(m, ), same as b, but standardized
            :attribute A_coeffs: coefficient form of A
                FS: np.array((m, n))
                FF, SF: np.array((n, m, k))
            :attribute y: optimal value of the first dual variable
            :attribute z: optimal value of the second dual variable
            :attribute r: number of selected features (after adaptive step if it is performed)
            :attribute r_no_adaptive: number of selected features before adaptive step. None if adaptive is not performed
            :attribute indx: position of the selected features
            :attribute selection_criterion_value: value of the chosen selected criterion
            :attribute sgm: final value of the augmented lagrangian parameter sigma
            :attribute c_lam: specific c_lam value used for the returned model
            :attribute alpha: same as input
            :attribute lam1_max: same as input
            :attribute lam1: specific lasso penalization value used for the returned model
            :attribute lam2: specific ridge penalization value used for the returned model
            :attribute time: total time of dal
            :attribute iters: total dal's iteration
            :attribute Aty: np.dot(A.T(), y) computed at the optimal y. Useful to implement warmstart
            :attribute convergence: True/False. If false the algorithm has not converged
            ------------------------------------------------------------------------------------------------------------
    :attribute k_selection: k used fot the feature selection
    :attribute k_estimation: k used fot the feature estimation. It can be different from k_selection, only in the FF
        regression model if select_k_estimation = TRUE
    :attribute r_vec: np.array, number of selected features for each value of c_lam
    :attribute selection_criterion_vec: np.array, value of the selection criterion for each value of c_lam
    :attribute c_lam_entry_value: np.array, contains the c_lam value for which each selected feature entered the model
    :attribute c_lam_vec: np.array, vector containing all the values of c_lam
    :attribute alpha: same as input
    :attribute lam1_vec: np.array, lasso penalization for each value of c_lam
    :attribute lam2_vec: np.array, ridge penalization for each value of c_lam
    :attribute lam1_max: same as input
    :attribute time_total: total time of FAStEN
    :attribute time_path: time to compute the solution path
    :attribute time_cv: time to perform cross validation
    :attribute time_adaptive: time to perform the adaptive step
    :attribute time_curves: time to compute the final estimated curves/surfaces from the basis coefficients
    :attribute iters_vec: array, iteration to converge for each value of c_lam
    :attribute times_vec: array, time to compute the solution for each value of c_lam
    --------------------------------------------------------------------------------------------------------------------

"""

import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from numpy import linalg as LA
from sklearn.model_selection import KFold
from fungcn.ffs.solver_FF import SolverFF
from fungcn.ffs.solver_SF import SolverSF
from fungcn.ffs.solver_FS import SolverFS
from fungcn.ffs.enum_classes import RegressionType, SelectionCriteria, AdaptiveScheme, FPCFeatures
from fungcn.ffs.output_classes import OutputPath, OutputPathCore
from fungcn.ffs.auxiliary_functions_FS import AuxiliaryFunctionsFS
from fungcn.ffs.auxiliary_functions_FF import AuxiliaryFunctionsFF
from fungcn.ffs.auxiliary_functions_SF import AuxiliaryFunctionsSF


class FASTEN:

    def solver_core(self, solver, regression_type,
                    A, b, k, wgts=1,
                    c_lam_vec=None, alpha=None, lam1_max=None,
                    x0=None, y0=None, z0=None, Aty0=None,
                    selection_criterion=SelectionCriteria.GCV,
                    relaxed_criteria=False, relaxed_estimates=False,
                    n_folds=10, max_selected=None,
                    check_selection_criterion=False,
                    sgm=1e-2, sgm_increase=5, sgm_change=1,
                    tol_nwt=1e-5, tol_dal=1e-6,
                    maxiter_nwt=50, maxiter_dal=100,
                    use_cg=False, r_exact=2e4,
                    print_lev=1):

        convergence = True
        sgm0 = sgm
        found_1st_model = False
        best_model = None

        selection_criterion_full_path = selection_criterion
        if selection_criterion_full_path == SelectionCriteria.CV:
            selection_criterion_full_path = SelectionCriteria.GCV

        # ------------------------ #
        #    recover dimensions    #
        # ------------------------ #

        if regression_type == RegressionType.FS:
            m, n = A.shape

        else:
            m, nk = A.shape
            n = int(nk / k)

        # -------------------------- #
        #    create output arrays    #
        # -------------------------- #

        n_lam1 = c_lam_vec.shape[0]
        n_iter = n_lam1
        selection_criterion_vec, r_vec = - np.ones([n_lam1]), - np.ones([n_lam1])
        times_vec, iters_vec = - np.ones([n_lam1]), - np.ones([n_lam1])
        c_lam_entry_value = np.zeros(n)

        # ---------------------- #
        #    solve full model    #
        # ---------------------- #

        if print_lev > 0 and n_lam1 > 1:
            print('-----------------------------------------------------------------------')
            print(' * solving path * ')
            print('-----------------------------------------------------------------------')

        start_path = time.time()

        for i in range(n_lam1):

            if print_lev > 2:
                print('-----------------------------------------------------------------------')
                print(' FULL MODEL:  c_lam = %.2f  |  sigma0 = %.2e' % (c_lam_vec[i], sgm0))
                print('-----------------------------------------------------------------------')

            # ----------------- #
            #    perform dal    #
            # ----------------- #

            fit = solver.solver(
                A=A, b=b, k=k, wgts=wgts,
                c_lam=c_lam_vec[i], alpha=alpha, lam1_max=lam1_max,
                x0=x0, y0=y0, z0=z0, Aty0=Aty0,
                selection_criterion=selection_criterion_full_path,
                relaxed_criteria=relaxed_criteria, relaxed_estimates=relaxed_estimates,
                sgm=sgm0, sgm_increase=sgm_increase, sgm_change=sgm_change,
                tol_nwt=tol_nwt, tol_dal=tol_dal,
                maxiter_nwt=maxiter_nwt, maxiter_dal=maxiter_dal,
                use_cg=use_cg, r_exact=r_exact,
                print_lev=print_lev - 3)

            # ----------------------- #
            #    check convergence    #
            # ----------------------- #

            if not fit.convergence:
                convergence = False
                break

            # ---------------------------- #
            #    update starting values    #
            # ---------------------------- #

            x0, y0, z0, Aty0, sgm0 = fit.x_coeffs, fit.y, fit.z, fit.Aty, fit.sgm

            # -------------------------- #
            #    update output arrays    #
            # -------------------------- #

            times_vec[i], r_vec[i], iters_vec[i] = fit.time, fit.r, fit.iters

            if r_vec[i] > 0:

                selection_criterion_vec[i] = fit.selection_criterion_value
                c_lam_entry_value[fit.indx] = np.maximum(c_lam_vec[i], c_lam_entry_value[fit.indx])

                # ---------------------- #
                #    update best model   #
                # ---------------------- #

                if found_1st_model:

                    # if the jump between two steps is too high, stop the search
                    if (check_selection_criterion and selection_criterion != SelectionCriteria.CV and fit.r > 5 and
                            np.abs((fit.selection_criterion_value - selection_criterion_vec[i-1]) /
                                   min(selection_criterion_vec[i], selection_criterion_vec[i-1])) > 5):
                        n_iter = i - 1
                        break

                    if fit.selection_criterion_value < best_model.selection_criterion_value:
                        best_model = fit

                else:
                    best_model = fit
                    found_1st_model = True

            # --------------------------------------- #
            #    check number of selected features    #
            # --------------------------------------- #

            if r_vec[i] > max_selected - 1:
                n_iter = i + 1
                # reached_max = True
                break

        # ------------------- #
        #    end full model   #
        # ------------------- #

        time_path = time.time() - start_path

        # -------------- #
        #    start cv    #
        # -------------- #

        time_cv = 0

        if selection_criterion == SelectionCriteria.CV and convergence:

            print('-----------------------------------------------------------------------')
            print(' * performing cv *  ')
            print('-----------------------------------------------------------------------')
            time.sleep(0.005)

            cv_mat = - np.ones([n_iter, n_folds])

            if regression_type == RegressionType.FF:
                x0_cv, z0_cv = np.zeros((n * k, k)), np.zeros((n * k, k))
            else:
                x0_cv, z0_cv = np.zeros((n, k)), np.zeros((n, k))

            Aty0_cv = None
            sgm_cv = sgm
            fold = 0

            start_cv = time.time()

            # ------------- #
            #    split A    #
            # ------------- #

            kf = KFold(n_splits=n_folds)
            kf.get_n_splits(A)

            # -------------------- #
            #    loop for folds    #
            # -------------------- #

            for train_index, test_index in kf.split(A):

                A_train, A_test = A[train_index], A[test_index]
                b_train, b_test = b[train_index], b[test_index]

                if regression_type == RegressionType.FS or regression_type == RegressionType.FF:
                    y0_cv = np.zeros((np.shape(train_index)[0], k))
                else:
                    y0_cv = np.zeros(np.shape(train_index)[0])

                # ------------------------ #
                #    loop for lam_ratio    #
                # ------------------------ #

                for i_cv in tqdm(range(n_iter)):

                    # ----------------- #
                    #    perform dal    #
                    # ----------------- #

                    fit_cv = solver.solver(
                        A=A_train, b=b_train, k=k, wgts=wgts,
                        c_lam=c_lam_vec[i_cv], alpha=alpha, lam1_max=lam1_max,
                        x0=x0_cv, y0=y0_cv, z0=z0_cv, Aty0=Aty0_cv,
                        relaxed_criteria=relaxed_criteria, relaxed_estimates=relaxed_criteria,
                        sgm=sgm_cv, sgm_increase=sgm_increase, sgm_change=sgm_change,
                        tol_nwt=tol_nwt, tol_dal=tol_dal,
                        maxiter_nwt=maxiter_nwt, maxiter_dal=maxiter_dal,
                        use_cg=use_cg, r_exact=r_exact,
                        print_lev=0)

                    # ------------------- #
                    #    update cv mat    #
                    # ------------------- #

                    if regression_type == RegressionType.FS or regression_type == RegressionType.FF:
                        cv_mat[i_cv, fold] = LA.norm(A_test @ fit_cv.x_coeffs - b_test) ** 2
                    else:
                        cv_mat[i_cv, fold] = LA.norm(A_test @ fit_cv.x_coeffs.reshape(n * k) - b_test) ** 2

                    # ---------------------------- #
                    #    update starting values    #
                    # ---------------------------- #

                    if i_cv == n_iter:
                        x0_cv, y0_cv, z0_cv, Aty0_cv, sgm_cv = None, None, None, None, sgm

                    else:
                        x0_cv, y0_cv, z0_Cv, Aty0_cv, sgm_cv = fit_cv.x_coeffs, fit_cv.y, fit_cv.z, fit_cv.Aty, fit_cv.sgm

                # ----------------------- #
                #    end loop for lam1    #
                # ----------------------- #

                fold += 1

            # ------------------- #
            #    best model cv    #
            # ------------------- #

            selection_criterion_vec = cv_mat.mean(1) / m

            # check there are jumps too large in the cv criterion
            if check_selection_criterion:
                for i in range(1, n_iter):
                    if np.abs((selection_criterion_vec[i] - selection_criterion_vec[i-1]) /
                              min(selection_criterion_vec[i], selection_criterion_vec[i-1])) > 1.5:
                        n_iter = i - 1
                        selection_criterion_vec = selection_criterion_vec[:n_iter]
                        break

            # find first best r different from 0
            ok_pos = r_vec[0:n_iter] > 0
            c_lam_cv = c_lam_vec[0:n_iter][ok_pos][np.argmin(selection_criterion_vec[ok_pos])]

            x0, y0, z0, Aty0, sgm0 = best_model.x_coeffs, best_model.y, best_model.z, best_model.Aty, best_model.sgm
            best_model = solver.solver(A=A, b=b, k=k, wgts=wgts,
                                       c_lam=c_lam_cv, alpha=alpha, lam1_max=lam1_max,
                                       # x0=None, y0=None, z0=None, Aty0=None,
                                       x0=x0, y0=y0, z0=z0, Aty0=Aty0,
                                       relaxed_criteria=relaxed_criteria, relaxed_estimates=relaxed_estimates,
                                       sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
                                       tol_nwt=tol_nwt, tol_dal=tol_dal,
                                       maxiter_nwt=maxiter_nwt, maxiter_dal=maxiter_dal,
                                       use_cg=use_cg, r_exact=r_exact,
                                       print_lev=0)

            time_cv = time.time() - start_cv

        # if not convergence:
        #     print('-----------------------------------------------------------------------')
        #     print(' dal has not converged for c_lam = %.2f' % c_lam_vec[i])
        #     print('-----------------------------------------------------------------------')

        return OutputPathCore(best_model, time_path, time_cv, r_vec[:n_iter], c_lam_entry_value,
                              times_vec[:n_iter], iters_vec[:n_iter], selection_criterion_vec[:n_iter], convergence)

    def solver(self, regression_type,
               A, b, k=None, wgts=1,
               selection_criterion=SelectionCriteria.GCV, n_folds=10,
               adaptive_scheme=AdaptiveScheme.SOFT, fpc_features=FPCFeatures.response,
               coefficients_form=False, x_basis=None, b_std=None,
               c_lam_vec=None, c_lam_vec_adaptive=None,
               max_selected=None, check_selection_criterion=False,
               alpha=0.2, lam1_max=None,
               x0=None, y0=None, z0=None, Aty0=None,
               compute_curves=True, select_k_estimation=True,
               relaxed_criteria=True, relaxed_estimates=True,
               sgm=5e-3, sgm_increase=5, sgm_change=1,
               tol_nwt=1e-6, tol_dal=1e-6,
               maxiter_nwt=50, maxiter_dal=100,
               use_cg=False, r_exact=2e4,
               plot=False, print_lev=1):

        # ----------------------------------------------------- #
        #   initialize fungcn and auxiliary functions objects   #
        # ----------------------------------------------------- #
        # (we set a different k fot estimation just for the FF model)

        if regression_type == RegressionType.FS:
            solver = SolverFS()
            af = AuxiliaryFunctionsFS()
            select_k_estimation = False

        elif regression_type == RegressionType.FF:
            solver = SolverFF()
            af = AuxiliaryFunctionsFF()

        else:
            solver = SolverSF()
            af = AuxiliaryFunctionsSF()
            select_k_estimation = False

        # ---------------------------- #
        #   dimension of the problem   #
        # ---------------------------- #

        if regression_type == RegressionType.FS:
            m, n = A.shape

        else:
            n, m, _ = A.shape

        # --------------------------------------------- #
        #    computing coefficient form if necessary    #
        # --------------------------------------------- #

        if print_lev > 0:
            print('')

        start_fasten = time.time()

        if coefficients_form and x_basis is None:
            print('')
            print('-----------------------------------------------------------------------')
            print(' ERROR: X basis are not given')
            print('-----------------------------------------------------------------------')

            return -1

        # b_std = np.array([1])  # we need to define b_sdt if coefficients form
        A_full, b_full, b_basis_full = None, None, None  # define variables if not coefficient forms

        if not coefficients_form:

            if print_lev > 1:
                print('-----------------------------------------------------------------------')
                print(' * computing coefficients form *')
                print('-----------------------------------------------------------------------')

            # standardize b
            b_std = b.std(axis=0) + 1e-32
            b = (b - b.mean(axis=0)) / b_std
            # b = (b - b.mean(axis=0))

            if select_k_estimation:
                A_full, b_full = np.copy(A), np.copy(b)

            if regression_type == RegressionType.FF and fpc_features == FPCFeatures.features:
                A, b, k, k_suggested, b_basis_full, x_basis, var_exp = af.compute_coefficients_form_using_FPC_features(A, b, k)

            else:
                A, b, k, k_suggested, b_basis_full, x_basis, var_exp = af.compute_coefficients_form(A, b, k)

        else:

            if b_std is None:
                raise Exception('If coefficient form is true, you have to standardize the response before computing '
                                'the coefficients and pass the standard deviation as input')

            k_suggested = -1
            var_exp = None

            if regression_type == RegressionType.FS:
                _, k = b.shape

            else:
                _, _, k = A.shape
                A = A.transpose(1, 0, 2).reshape(m, n * k)

        # -----------------------#
        #    compute lam1 max    #
        # ---------------------- #

        if lam1_max is None:

            if regression_type == RegressionType.FS:
                lam1_max = np.max(LA.norm(A.T @ b, axis=1) / wgts) / alpha

            elif regression_type == RegressionType.FF:
                lam1_max = np.max(LA.norm((A.T @ b).reshape(n, k * k), axis=1) / wgts) / alpha

            else:
                lam1_max = np.max(LA.norm((A.T @ b).reshape(n, k), axis=1) / wgts) / alpha

        # -------------------------- #
        #    initialize variables    #
        # -------------------------- #

        if regression_type == RegressionType.FS:

            if x0 is None:
                x0 = np.zeros((n, k))
            if y0 is None:
                y0 = np.zeros((m, k))
            if z0 is None:
                z0 = np.zeros((n, k))

        elif regression_type == RegressionType.FF:

            if x0 is None:
                x0 = np.zeros((n * k, k))
            if y0 is None:
                y0 = np.zeros((m, k))
            if z0 is None:
                z0 = np.zeros((n * k, k))

        else:

            if x0 is None:
                x0 = np.zeros((n, k))
            if y0 is None:
                y0 = np.zeros(m)
            if z0 is None:
                z0 = np.zeros((n, k))

        if max_selected is None:
            max_selected = n

        if c_lam_vec is None:
            c_lam_vec = np.geomspace(1, 0.01, num=100)

        elif np.isscalar(c_lam_vec):
            c_lam_vec = np.array([c_lam_vec])
            selection_criterion = SelectionCriteria.GCV

        lam1_vec = alpha * c_lam_vec * lam1_max
        lam2_vec = (1 - alpha) * c_lam_vec * lam1_max

        # ---------------------- #
        #    call output core    #
        # ---------------------- #
        if print_lev > 0:
            print('')
            print('-----------------------------------------------------------------------')
            print(' FASTEN ')
            print('-----------------------------------------------------------------------')

        fit = self.solver_core(solver, regression_type,
                               A, b, k, wgts,
                               c_lam_vec=c_lam_vec, alpha=alpha, lam1_max=lam1_max,
                               x0=x0, y0=y0, z0=z0, Aty0=Aty0,
                               selection_criterion=selection_criterion,
                               relaxed_criteria=relaxed_criteria,
                               relaxed_estimates=relaxed_estimates,
                               n_folds=n_folds, max_selected=max_selected,
                               check_selection_criterion=check_selection_criterion,
                               sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
                               tol_nwt=tol_nwt, tol_dal=tol_dal,
                               maxiter_nwt=maxiter_nwt, maxiter_dal=maxiter_dal,
                               use_cg=use_cg, r_exact=r_exact,
                               print_lev=print_lev)

        best_model = fit.best_model

        # --------------------- #
        #    adaptive scheme    #
        # --------------------- #

        time_adaptive = 0

        if adaptive_scheme != AdaptiveScheme.NONE and best_model.r > 1:

            print('')
            print('-----------------------------------------------------------------------')
            print(' ADAPTIVE STEP ')
            print('-----------------------------------------------------------------------')

            start_adaptive = time.time()

            # ------------------------------------------- #
            #    compute weights and reduced variables    #
            # ------------------------------------------- #

            if regression_type == RegressionType.FF:
                indx = np.repeat(best_model.indx, k)
                xj = best_model.x_coeffs[indx, :]
                AJ = A[:, indx]

                # relax if needed
                if not relaxed_estimates and m > best_model.r * k:
                    xj = LA.solve(AJ.T @ AJ, AJ.T @ b)
                normsxj = LA.norm(xj.reshape(best_model.r, k * k), axis=1)

            elif regression_type == RegressionType.FS:
                indx = best_model.indx
                xj = best_model.x_coeffs[indx, :]
                AJ = A[:, indx]

                # relax if needed
                if not relaxed_estimates and m > best_model.r:
                    xj = LA.solve(AJ.T @ AJ, AJ.T @ b)
                normsxj = LA.norm(xj, axis=1)

            else:
                indx = best_model.indx
                xj = best_model.x_coeffs[indx, :]
                AJ = A[:, np.repeat(indx, k)]

                # relax if needed
                if not relaxed_estimates and m > best_model.r:
                    xj = LA.solve(AJ.T @ AJ, AJ.T @ b).reshape(best_model.r, k)
                normsxj = LA.norm(xj, axis=1)

            # ------------------- #
            #    full adaptive    #
            # ------------------- #

            if adaptive_scheme == AdaptiveScheme.FULL:

                selection_criterion_adaptive = selection_criterion
                relaxed_criteria_adaptive, relaxed_estimates_adaptive = False, False
                wgts = (1 / normsxj).reshape(best_model.r, 1)

                # ------------------------ #
                #    recompute lam1 max    #
                # ------------------------ #

                if regression_type == RegressionType.FS:
                    lam1_max_adaptive = np.max(LA.norm(AJ.T @ b, axis=1) / wgts) / alpha

                elif regression_type == RegressionType.FF:
                    lam1_max_adaptive = np.max(LA.norm((AJ.T @ b).reshape(best_model.r, k * k), axis=1) / wgts) / alpha

                else:
                    lam1_max_adaptive = np.max(LA.norm((AJ.T @ b).reshape(best_model.r, k), axis=1) / wgts) / alpha

                if c_lam_vec_adaptive is None:
                    c_lam_vec_adaptive = np.geomspace(1, 0.0001, num=50)

            # ------------------- #
            #    soft adaptive    #
            # ------------------- #

            else:

                selection_criterion_adaptive = SelectionCriteria.GCV
                relaxed_criteria_adaptive, relaxed_estimates_adaptive = True, True
                lam1_max_adaptive = lam1_max
                wgts = (normsxj.std() / normsxj).reshape(best_model.r, 1)
                c_lam_vec_adaptive = np.array([best_model.c_lam])

            # ---------------------------- #
            #    set initial parameters    #
            # ---------------------------- #

            y0, z0, sgm0 = best_model.y, best_model.z[indx, :], best_model.sgm

            if regression_type != regression_type.SF:
                AJty0 = AJ.T @ y0
            else:
                AJty0 = (AJ.T @ y0).reshape(best_model.r, k)

            fit_adaptive = self.solver_core(solver, regression_type,
                                            AJ, b, k, wgts,
                                            c_lam_vec=c_lam_vec_adaptive , alpha=alpha, lam1_max=lam1_max_adaptive,
                                            x0=xj, y0=y0, z0=z0, Aty0=AJty0,
                                            selection_criterion=selection_criterion_adaptive,
                                            relaxed_criteria=relaxed_criteria_adaptive,
                                            relaxed_estimates=relaxed_estimates_adaptive,
                                            n_folds=n_folds, max_selected=max_selected,
                                            check_selection_criterion=False,
                                            sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
                                            tol_nwt=tol_nwt, tol_dal=tol_dal,
                                            maxiter_nwt=maxiter_nwt, maxiter_dal=maxiter_dal,
                                            use_cg=use_cg, r_exact=r_exact,
                                            print_lev=print_lev)

            best_model_adaptive = fit_adaptive.best_model

            # ------------------------------- #
            #    update original variables    #
            # ------------------------------- #

            best_model.r_no_adaptive = best_model.r
            best_model.r = best_model_adaptive.r
            best_model.selection_criterion_value = best_model_adaptive.selection_criterion_value
            indx_prev = np.copy(best_model.indx)
            best_model.indx[np.where(best_model.indx * 1 > 0)] = best_model_adaptive.indx  # update original index

            if regression_type == RegressionType.FF:
                    best_model.x_coeffs[np.repeat(indx_prev, k), :] = best_model_adaptive.x_coeffs
            else:
                best_model.x_coeffs[indx_prev, :] = best_model_adaptive.x_coeffs

            time_adaptive = time.time() - start_adaptive

        # -------------------------------- #
        #    increment k for estimation    #
        # -------------------------------- #

        k_estimation = k
        best_model.x_basis = x_basis
        time_k_estimation = 0

        if select_k_estimation and not coefficients_form and adaptive_scheme != AdaptiveScheme.FULL:
            start_k_estimation = time.time()
            best_model, k_estimation = af.select_k_estimation(A_full, b_full, b_basis_full, best_model)
            time_k_estimation = time.time() - start_k_estimation

        # -------------------------------------- #
        #    compute curves from coefficients    #
        # -------------------------------------- #

        start_curves = time.time()

        if compute_curves:
            if regression_type == RegressionType.FF:
                best_model.x_coeffs = best_model.x_coeffs.reshape(n, k_estimation, k_estimation)

            best_model.x_curves = af.compute_curves(best_model, b_std)

        time_curves = time.time() - start_curves

        # ------------------------- #
        #    print final results    #
        # ------------------------- #

        time_total = time.time() - start_fasten
        time.sleep(0.005)

        n_iter = fit.r_vec.shape[0]
        if print_lev > 1 and c_lam_vec.shape[0] > 1:

            # --------------------------- #
            #    printing final matrix    #
            # --------------------------- #

            if selection_criterion == SelectionCriteria.GCV:
                selection_criterion_print = 'gcv'
            elif selection_criterion == SelectionCriteria.EBIC:
                selection_criterion_print = 'ebic'
            else:
                selection_criterion_print = 'cv'

            print()
            print('-----------------------------------------------------------------------')
            print(' Path Matrix')
            print('-----------------------------------------------------------------------')

            print_matrix1 = np.stack((c_lam_vec[:n_iter], fit.r_vec, fit.selection_criterion_vec)).T  #
            df1 = pd.DataFrame(print_matrix1, columns=['c_lam', 'r', selection_criterion_print])
            pd.set_option('display.max_rows', df1.shape[0] + 1)
            print(df1.round(3))

            r_not0 = (fit.r_vec > 0)[:n_iter]
            nzeros = n_iter - np.sum(r_not0)
            argmin_selection_criterion = np.argmin(fit.selection_criterion_vec[r_not0]) + nzeros

            if adaptive_scheme == AdaptiveScheme.SOFT:

                print_matrix2 = np.array([argmin_selection_criterion, np.min(fit.selection_criterion_vec[r_not0]),
                                          c_lam_vec[argmin_selection_criterion], best_model.r_no_adaptive, best_model.r])
                df2 = pd.DataFrame(print_matrix2,  columns=['summary'], index=['argmin', selection_criterion_print,
                                                                               'c_lam', 'r', 'r_soft_adaptive']).T
            else:

                print_matrix2 = np.array([argmin_selection_criterion, np.min(fit.selection_criterion_vec[r_not0]),
                                          c_lam_vec[argmin_selection_criterion], best_model.r])
                df2 = pd.DataFrame(print_matrix2,columns=['summary'],  index=['argmin', selection_criterion_print,
                                                                              'c_lam', 'r'],).T

            print('-----------------------------------------------------------------------')
            print(df2.round(3))
            print('-----------------------------------------------------------------------')

            if adaptive_scheme == AdaptiveScheme.FULL:

                print()
                print('-----------------------------------------------------------------------')
                print(' Adaptive Path Matrix')
                print('-----------------------------------------------------------------------')

                n_iter_adaptive = fit_adaptive.r_vec.shape[0]
                print_matrix1 = np.stack((c_lam_vec_adaptive[:n_iter_adaptive], fit_adaptive.r_vec,
                                          fit_adaptive.selection_criterion_vec)).T  #
                df1 = pd.DataFrame(print_matrix1, columns=['c_lam', 'r', selection_criterion_print])
                pd.set_option('display.max_rows', df1.shape[0] + 1)
                print(df1.round(3))

                r_not0 = (fit_adaptive.r_vec > 0)[:n_iter_adaptive]
                nzeros = n_iter_adaptive - np.sum(r_not0)
                argmin_selection_criterion = np.argmin(fit_adaptive.selection_criterion_vec[r_not0]) + nzeros

                print_matrix2 = np.array([argmin_selection_criterion, np.min(fit_adaptive.selection_criterion_vec[r_not0]),
                                          c_lam_vec_adaptive[argmin_selection_criterion], best_model.r_no_adaptive,
                                          best_model.r])
                df2 = pd.DataFrame(print_matrix2, columns=['summary'], index=['argmin', selection_criterion_print,
                                                                              'c_lam', 'r', 'r_full_adaptive']).T

                print('-----------------------------------------------------------------------')
                print(df2.round(3))
                print('-----------------------------------------------------------------------')

        if print_lev > 0 and c_lam_vec.shape[0] < 2:
            print()
            print('-----------------------------------------------------------------------')
            print(' One lambda explored:  r=%d  |  c_lam=%.3f' % (best_model.r, best_model.c_lam))
            print('-----------------------------------------------------------------------')

        if print_lev > 0:

            print('')
            print('-----------------------------------------------------------------------')
            if regression_type == RegressionType.FF:
                print(' k suggested = %.d   |   k selection = %d   |   k estimation = %d' % (k_suggested, k, k_estimation))
            else:
                print(' k suggested = %.d   |   k used = %d ' % (k_suggested, k))
            print('-----------------------------------------------------------------------')
            if var_exp is not None:
                print(' variance explained = %.4f' % var_exp[k - 1])
                print('-----------------------------------------------------------------------')
            print('')
            print('-----------------------------------------------------------------------')
            print(' total time:  %.4f' % time_total)
            print('-----------------------------------------------------------------------')
            print('       path:  %.4f' % fit.time_path)
            print('-----------------------------------------------------------------------')
            if selection_criterion == SelectionCriteria.CV:
                print('         cv:  %.4f' % fit.time_cv)
                print('-----------------------------------------------------------------------')
            if adaptive_scheme != adaptive_scheme.NONE:
                print('   adaptive:  %.4f' % time_adaptive)
                print('-----------------------------------------------------------------------')
            if select_k_estimation:
                print('   change k:  %.4f' % time_k_estimation)
                print('-----------------------------------------------------------------------')
            print('     curves:  %.4f' % time_curves)
            print('-----------------------------------------------------------------------')
            print('')

        # -------------------- #
        #    plot if needed    #
        # -------------------- #

        if plot and c_lam_vec.shape[0] > 1:
            if selection_criterion == SelectionCriteria.GCV:
                main = 'GCV'
            elif selection_criterion == SelectionCriteria.CV:
                main = 'CV'
            else:
                main = 'ebic'
            af.plot_selection_criterion(fit.r_vec, fit.selection_criterion_vec, alpha, c_lam_vec[:n_iter], main)

        # ------------------- #
        #    create output    #
        # ------------------- #

        return OutputPath(best_model, k, k_estimation, fit.r_vec, fit.selection_criterion_vec, fit.c_lam_entry_value,
                          c_lam_vec[:n_iter], alpha, lam1_vec[:n_iter], lam2_vec[:n_iter], lam1_max,
                          time_total, fit.time_path, fit.time_cv, time_adaptive, time_curves,
                          fit.iters_vec, fit.times_vec)



