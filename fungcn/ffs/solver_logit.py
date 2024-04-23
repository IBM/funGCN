""" class SolverLogit
        fungcn for logit regression with functional predictors

    solver_core: carries out the Dual Augmented Lagrangian minimization algorithm
    fungcn: pre-process --> solver_core --> post-process

    INPUT PARAMETERS:
    --------------------------------------------------------------------------------------------------------------------
    :param A: design matrix
        FS: np.array((m, n))
        FF, SF: np.array((n, m, neval))
    :param b: response matrix/vector
        FS, FF: np.array((m, neval))
        SF: np.array(m, )
    :param k: number of basis function
    :param wgts: individual weights for the penalty. 1 (default) or np.array with shape (n, 1)
    :param c_lam: to determine lam1 and lam2, ratio of lam_max considered
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
    :param selection_criterion: an object of class SelectionCriteria, it can be CV, GCV, EBIC.
    :param relaxed_criteria: if True a linear regression is fitted on the selected features before computing
        the selection criterion
    :param relaxed_estimates: if True a linear regression is fitted on the features to produce the final estimates
    :param sgm: starting value of the augmented lagrangian parameter sigma
    :param sgm_increase: increasing factor of sigma
    :param sgm_change: we increase sgm -- sgm *= sgm_increase -- every sgm_change iterations
    :param tol_nwt: tolerance for the nwt algorithm
    :param tol_dal: global tolerance of the dal algorithm
    :param maxiter_nwt: maximum number of iterations for nwt
    :param maxiter_dal: maximum number of global iterations
    :param use_cg: True/False. If true, the conjugate gradient method is used to find the direction of the nwt
    :param r_exact: number of features such that we start using the exact method
    :param print_lev: different level of printing (0, 1, 2)
    --------------------------------------------------------------------------------------------------------------------

    OUTPUT: OutputSolver object with the following attributes
    --------------------------------------------------------------------------------------------------------------------
    :return x_curves: curves computed just for the not 0 estimated coefficients
        FF: x_basis1 @ x_scores @ x_basis2.T, np.array((r, neval, neval))
        FS, SF: x_curves = x_scores @ x_basis.T,  np.array((r, neval))
        They are returned as None by this function then and computed for the best models in path_solver
    :return x_coeffs: standardized estimated coefficients. They are estimated based on: (b - b.mean) / b.std()
        FS, SF: np.array((n, k))
        FF: np.array((n, k, k))
    :return x_basis: they are returned as None by this function then inserted in path_solver
    :return b_coeffs: coefficient form of b
        FS, FF: np.array((m, k))
        SF: np.array(m, ), same as b, but standardized
    :return A_coeffs: coefficient form of A
        FS: np.array((m, n))
        FF, SF: np.array((n, m, k))
    :return y: optimal value of the first dual variable
    :return z: optimal value of the second dual variable
    :return r: number of selected features (after adaptive step if it is performed)
    :return r_no_adaptive: number of selected features before adaptive step. None if adaptive is not performed
    :return indx: position of the selected features
    :return selection_criterion_value: value of the chosen selected criterion
    :return sgm: final value of the augmented lagrangian parameter sigma
    :return c_lam: same as input
    :return alpha: same as input
    :return lam1_max: same as input
    :return lam1: lasso penalization
    :return lam2: ridge penalization
    :return time: total time of dal
    :return iters: total dal's iteration
    :return Aty: np.dot(A.T(), y) computed at the optimal y. Useful to implement warmstart
    :return convergence: True/False. If false the algorithm has not converged
    --------------------------------------------------------------------------------------------------------------------

"""

import time
import numpy as np
from numpy import linalg as LA
import scipy.sparse.linalg as ss_LA
from scipy.linalg import block_diag
from scipy.special import logit
from fungcn.ffs.enum_classes import SelectionCriteria
from fungcn.ffs.output_classes import OutputSolver, OutputSolverCore
from fungcn.ffs.auxiliary_functions_logit import AuxiliaryFunctionsLogit

class SolverLogit:

    def solver_core(self, A, b, k, m, n, wgts,
                    lam1, lam2,
                    x0=None, y0=None, z0=None, Aty0=None,
                    sgm=5e-3, sgm_increase=5, sgm_change=3,
                    tol_nwt=1e-6, tol_dal=1e-6,
                    maxiter_nwt=50, maxiter_dal=100,
                    use_cg=False, r_exact=2e4,
                    print_lev=1):

        # -------------------------- #
        #    initialize variables    #
        # -------------------------- #

        af = AuxiliaryFunctionsLogit()

        x = x0
        y = y0
        z = z0
        Aty = Aty0

        if x is None:
            x = np.zeros((n, k))
        if y is None:
            y = np.zeros(m)
        if z is None:
            z = np.zeros((n, k))
        if Aty0 is None:
            Aty = (A.T @ y).reshape(n, k)

        convergence_dal = False

        # -------------------------- #
        #    start dal iterations    #
        # -------------------------- #

        start_dal = time.time()

        for it_dal in range(maxiter_dal):

            if print_lev > 2:
                print('')
                print('   dal iteration = %.f  |  sgm = %.2e' % (it_dal + 1, sgm))
                print('   -------------------------------------------------------------------')

            # --------------- #
            #    start nwt    #
            # --------------- #

            start_nwt = time.time()

            convergence_nwt = False
            t = x - sgm * Aty

            for it_nwt in range(maxiter_nwt):

                # ---------------------------- #
                #    select active features    #
                # ---------------------------- #

                normst = LA.norm(t, axis=1).reshape(n, 1)
                indx = (normst > wgts * sgm * lam1).reshape(n)
                tj = t[indx, :]
                xj = x[indx, :]
                AJ = A[:, np.repeat(indx, k)]
                r = tj.shape[0]
                normsj = normst[indx]
                wgtsj = wgts
                if isinstance(wgtsj, np.ndarray):
                    wgtsj = wgts[indx, :]

                # ------------------------- #
                #    compute direction d    #
                # ------------------------- #

                if r == 0:
                    method = 'E '

                    # set norm grad = = 0
                    norm_grad = 0

                    # gradient when we do not select any columns
                    d = b * 0.5

                    # update y
                    y = y + d
                    Aty = (A.T @ y).reshape(n, k)

                else:

                    rhs = - af.grad_phi(A, y, x, b, Aty, sgm, lam1, lam2, wgts)

                    # compute norm grad to check convergence
                    norm_grad = LA.norm(rhs)

                    # ------------------------ #
                    #    compute delta prox    #
                    # ------------------------ #

                    const = 1 / (1 + wgtsj * sgm * lam2)
                    delta_prox = ((const * (1 - wgtsj * sgm * lam1 / normsj)).reshape(r, 1, 1) * np.eye(k) +
                                  (const * wgtsj * sgm * lam1 / normsj ** 3).reshape(r, 1, 1) *
                                  np.einsum('...i,...j', tj, tj))

                    # manual hessian
                    by = np.abs(b * y)
                    I = np.where((by > 0) & (by < 1))
                    byI = by[np.where((by > 0) & (by < 1))]
                    hloss = np.zeros(m)
                    hloss[I] = 1 / (byI * (1 - byI))

                    # ------------------- #
                    #    standard case    #
                    # ------------------- #

                    if m <= r:
                        H = np.diag(hloss) + sgm * AJ @ block_diag(*delta_prox) @ AJ.T

                        # conjugate method
                        if r * k > r_exact and use_cg:
                            method = 'CG'
                            d = (ss_LA.cg(H, rhs, tol=1e-04, maxiter=1000)[0])

                        # exact method:
                        else:
                            method = 'E '
                            d = LA.solve(H, rhs)

                    # ---------------------- #
                    #    Woodbury formula    #
                    # ---------------------- #

                    else:
                        # precompute inverse matrices
                        inv_delta_prox = block_diag(*LA.inv(delta_prox))
                        inv_hloss = np.diag(1 / hloss)
                        AJT_inv_hloss = AJ.T @ inv_hloss

                        # conjugate method
                        if r * k > r_exact and use_cg:
                            method = 'CG'
                            d_temp = (ss_LA.cg(inv_delta_prox / sgm + AJT_inv_hloss @ AJ, AJT_inv_hloss @ rhs,
                                               tol=1e-04, maxiter=1000)[0])
                            d = inv_hloss @ (rhs - AJ @ d_temp)

                        # exact method:
                        else:
                            method = 'E '
                            d_temp = LA.solve(inv_delta_prox / sgm + AJT_inv_hloss @ AJ, AJT_inv_hloss @ rhs)
                            d = inv_hloss @ (rhs - AJ @ d_temp)

                    # ------------------------------- #
                    #    line-search for step size    #
                    # ------------------------------- #

                    if isinstance(wgtsj, np.ndarray):
                        y = y + 0.3 * d
                        Aty = (A.T @ y).reshape(n, k)

                    else:

                        # TODO: lineaserch is not working!
                        y = y + 1 * d
                        Aty = (A.T @ y).reshape(n, k)

                        # ------------------- #
                        #    line-search 1    #
                        # ------------------- #

                        # step_size = 1
                        # mu = 0.2
                        # step_reduce = 0.8
                        #
                        # rhs_term_1 = af.phi_y(y, xj, b, Aty[indx, :], sgm, lam1, lam2, wgtsj)
                        # rhs_term_2 = mu * np.sum(rhs * d)
                        #
                        # while True:
                        #     y = y + step_size * d
                        #     Aty = (A.T @ y).reshape(n, k)
                        #
                        #     if af.phi_y(y, x, b, Aty, sgm, lam1, lam2, wgts) <= rhs_term_1 + step_size * rhs_term_2:
                        #         break
                        #
                        #     step_size *= step_reduce

                        # ------------------- #
                        #    line-search 2    #
                        # ------------------- #

                        # step_size = 1
                        # step_reduce = 0.8
                        #
                        # fval = af.dual_obj(y, z, b, lam1, lam2, wgts)
                        #
                        # while True:
                        #     y = y + step_size * d
                        #     f_val_new = af.dual_obj(y, z, b, lam1, lam2, wgts)
                        #
                        #     if f_val_new <= fval:
                        #         Aty = (A.T @ y).reshape(n, k)
                        #         break
                        #     step_size *= step_reduce

                # ---------------------- #
                #    update variables    #
                # ---------------------- #

                z = af.prox_star(x / sgm - Aty, wgts * sgm * lam1, wgts * sgm * lam2, sgm)
                t = x - sgm * Aty
                x_temp = t - sgm * z
                # x_temp = af.prox(x - sgm * Aty, wgts * lam1 * sgm, wgts * lam2 * sgm)

                # --------------------------- #
                #    nwt convergence check    #
                # --------------------------- #

                # compute kkt1
                by = np.abs(b * y)
                I = np.where((by > 0) & (by < 1))
                gloss = np.zeros(y.shape[0])
                gloss[I] = b[I] * logit(by[I])
                kkt1 = LA.norm(AJ @ x_temp[indx, :].reshape(r * k) - gloss) / (1 + LA.norm(b) + LA.norm(y))

                # # compute kk2
                # par1 = wgts * lam1
                # par2 = wgts * lam2
                # z_norm = LA.norm(z, axis=1).reshape(n, 1)
                # g_p_star = np.maximum(0, z_norm - par1) / par2 * z / z_norm
                # kkt2 = (np.sum(LA.norm(g_p_star - x_temp, axis=1))
                #         / (1 + np.sum(LA.norm(g_p_star, axis=1) + LA.norm(x_temp, axis=1))))

                # compute paper inner condition
                inn_condition = 2 / np.sqrt(sgm) * np.sum(LA.norm(x - x_temp, axis=1))

                if print_lev > 2:
                    if it_nwt + 1 > 9:
                        space = ''
                    else:
                        space = ' '
                    print(space, '   %.f| ' % (it_nwt + 1), method, ' kkt1 = %.2e - r = %.f' % (kkt1, r), sep='')

                if kkt1 < tol_nwt or r == 0 or norm_grad < inn_condition:
                    convergence_nwt = True
                    break

                # if r == 0 or norm_grad < inn_condition:
                #     convergence_nwt = True
                #     break

            # ------------- #
            #    end nwt    #
            # ------------- #

            time_nwt = time.time() - start_nwt

            if print_lev > 2:
                print('   -------------------------------------------------------------------')
                print('   nwt time = %.4f' % time_nwt)
                print('   -------------------------------------------------------------------')

            # if not convergence_nwt and kkt1 > 10 * tol_nwt:
            if not convergence_nwt:
                print('\n')
                print('  * NEWTON DOES NOT CONVERGE -- try to: ' '\n'
                      '    - increase Newton tolerance' '\n'
                      '         newton tolerance = ', tol_nwt, '\n'
                      '                     kkt1 = ', kkt1, '\n'
                      '    - start from smaller sgm0' '\n'
                      '    - increase the value of alpha' '\n'
                      '    - use print_lev = 7 to see all details')
                print('\n')
                break

            # ---------------------- #
            #    update variables    #
            # ---------------------- #

            if r > 0:
                indx_new = (LA.norm(t, axis=1).reshape(n, 1) > wgts * sgm * lam1).reshape(n)
                if np.sum(1 * indx_new - 1 * indx) != 0:
                    indx = indx_new
                    AJ = A[:, np.repeat(indx, k)]
                    r = int(AJ.shape[1] / k)

            x = x_temp
            xj = x[indx, :]
            if isinstance(wgtsj, np.ndarray):
                wgtsj = wgts[indx, :]

            # --------------------------- #
            #    dal convergence check    #
            # --------------------------- #

            # compute kkt3
            kkt3 = np.sum(LA.norm(z + Aty, axis=1)) / (1 + np.sum(LA.norm(z, axis=1)) + LA.norm(y))

            # compute dual gap
            prim = af.prim_obj(AJ, xj, b, lam1, lam2, wgtsj)
            dual = af.dual_obj(y, z[indx, :], b, lam1, lam2, wgtsj)
            dual_gap = np.abs((prim - dual) / prim)
            # dual_gap = np.abs(prim - dual) / (prim + dual)

            if print_lev > 2:
                print('   kkt3 = %.5e  -  dual gap = %.5e' % (kkt3, dual_gap))

            if r == 0 and it_dal == 0 and kkt3 < 1e-10:
                convergence_dal = True
                it_dal += 1
                break

            if kkt3 < tol_dal:  # and dual_gap < 1e-3:
                convergence_dal = True
                it_dal += 1
                break

            # if dual_gap < 1e-3:
            #     convergence_dal = True
            #     it_dal += 1
            #     break

            if np.mod(it_dal + 1, sgm_change) == 0:
                sgm *= sgm_increase

        # ------------- #
        #    end dal    #
        # ------------- #

        dal_time = time.time() - start_dal

        # ------------------- #
        #    create output    #
        # ------------------- #

        return OutputSolverCore(x, xj, AJ, y, z, r, sgm, indx, dal_time, it_dal, Aty, prim, dual, kkt3,
                                convergence_dal)

    def solver(self, A, b, k, wgts=1,
               c_lam=None, alpha=None, lam1_max=None,
               x0=None, y0=None, z0=None, Aty0=None,
               selection_criterion=SelectionCriteria.GCV,
               sgm=5e-3, sgm_increase=5, sgm_change=3,
               tol_nwt=1e-6, tol_dal=1e-6,
               maxiter_nwt=50, maxiter_dal=100,
               use_cg=False, r_exact=2e4,
               print_lev=1):

        # ---------------------------- #
        #   dimension of the problem   #
        # ---------------------------- #

        m, nk = A.shape
        n = int(nk / k)

        # ------------------------------------- #
        #    compute lam1 max, lam1 and lam2    #
        # ------------------------------------- #

        lam1 = alpha * c_lam * lam1_max
        lam2 = (1 - alpha) * c_lam * lam1_max

        if print_lev > 1:
            print('  -------------------------------------------------------------------')
            print('  lam1_max = %.3f  |  lam1 = %.3f   |   lam2 = %.4f   ' % (lam1_max, lam1, lam2))
            print('  -------------------------------------------------------------------')

        # ---------------------- #
        #    call solver_core    #
        # ---------------------- #

        out_core = self.solver_core(
            A=A, b=b, k=k, m=m, n=n, wgts=wgts,
            lam1=lam1, lam2=lam2,
            x0=x0, y0=y0, z0=z0, Aty0=Aty0,
            sgm=sgm, sgm_increase=sgm_increase, sgm_change=sgm_change,
            tol_nwt=tol_nwt, tol_dal=tol_dal,
            maxiter_nwt=maxiter_nwt, maxiter_dal=maxiter_dal,
            use_cg=use_cg, r_exact=r_exact,
            print_lev=print_lev)

        x, xj, AJ = out_core.x, out_core.xj, out_core.AJ
        indx, r = out_core.indx, out_core.r

        # ------------------------------ #
        #    model selection criteria    #
        # ------------------------------ #

        selection_criterion_value = None

        # # compute dof
        # df_core = LA.inv(AJ.T @ AJ + lam2 * np.eye(r * k))
        # df = np.trace(AJ @ df_core @ AJ.T)

        # compute log-likelihood
        # rss = LA.norm(b - AJ @ xj_criteria.ravel()) ** 2
        ll = np.sum(np.log(1 + np.exp(b * (AJ @ xj.ravel()))))

        if selection_criterion == SelectionCriteria.GCV:
            selection_criterion_value = r * ll / (m - k) ** 2
            # selection_criterion_value = r * ll / (m - k * df) ** 2

        elif selection_criterion == SelectionCriteria.EBIC:
            selection_criterion_value = (ll + r * np.log(m) + np.log(r + 1e-32)) / m
            # selection_criterion_value = np.log(rss / m) + 0.1 * df * np.log(m) / m + 0.1 * df * np.log(r + 1e-32) / m

        if print_lev > 0:

            print('')
            print('  ==================================================')
            print('   * core iters ......... %.f' % out_core.iters)
            print('   * core time .......... %.4f' % out_core.time)
            print('   * prim object ........ %.4e' % out_core.prim)
            print('   * dual object ........ %.4e' % out_core.dual)
            print('   * kkt3 ............... %.4e' % out_core.kkt3)
            print('   * not0 ............... %.f' % r)
            if r > 0:
                if selection_criterion == SelectionCriteria.EBIC:
                    print('   * ebic ............... %.4f' % selection_criterion_value)
                if selection_criterion == SelectionCriteria.GCV:
                    print('   * gcv ................ %.4f' % selection_criterion_value)
            print('  ==================================================')
            print(' ')

        if not out_core.convergence:
            print('\n')
            print('   * THE SOLVER HAS NOT CONVERGED:')
            print('\n')

        # ------------------- #
        #    create output    #
        # ------------------- #

        return OutputSolver(None, -x, None, b, A, out_core.y, out_core.z,
                            r, None, indx, selection_criterion_value, out_core.sgm,
                            c_lam, alpha, lam1_max, lam1, lam2,
                            out_core.time, out_core.iters, out_core.Aty, out_core.convergence)

