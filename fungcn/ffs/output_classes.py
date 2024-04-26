"""

Definition of classes needed to specify different output types

    class OutputSolverCore: definition of the output for the core part of each fungcn (FS, FF, and SF)

    class OutputSolver: definition of the output for each fungcn (FS, FF, and SF)

    class OutputPathCore: definition of the output for the core part of path fungcn

    class OutputPath: definition of the output for the path fungcn


"""


class OutputSolverCore:

    """
    Definition of the output class for solver_core

    """

    def __init__(self, x, xj, AJ, y, z, r, sgm, indx, time, iters, Aty, prim, dual, kkt3, convergence):
        self.x = x
        self.xj = xj
        self.AJ = AJ
        self.y = y
        self.z = z
        self.r = r
        self.sgm = sgm
        self.indx = indx
        self.time = time
        self.iters = iters
        self.Aty = Aty
        self.prim = prim
        self.dual = dual
        self.kkt3 = kkt3
        self.convergence = convergence


class OutputSolver:

    """
    Definition of the output class for single (best) model FASTEN

    """

    def __init__(self, x_curves, x_coeffs, x_basis, b_coeffs, A_coeffs, y, z, r, r_no_adaptive, indx,
                 selection_criterion_value, sgm, c_lam, alpha, lam1_max, lam1, lam2, time_tot, iters, Aty, convergence):
        self.x_curves = x_curves
        self.x_coeffs = x_coeffs
        self.x_basis = x_basis
        self.b_coeffs = b_coeffs
        self.A_coeffs = A_coeffs
        self.y = y
        self.z = z
        self.r = r
        self.r_no_adaptive = r_no_adaptive
        self.indx = indx
        self.selection_criterion_value = selection_criterion_value
        self.sgm = sgm
        self.c_lam = c_lam
        self.alpha = alpha
        self.lam1_max = lam1_max
        self.lam1 = lam1
        self.lam2 = lam2
        self.time = time_tot
        self.iters = iters
        self.Aty = Aty
        self.convergence = convergence


class OutputPathCore:

    """
    Definition of the output class for solver_path_core

    """

    def __init__(self, best_model, time_path, time_cv, r_vec, c_lam_entry_value, times_vec, iters_vec,
                 selection_criterion_vec, convergence):
        self.best_model = best_model
        self.time_path = time_path
        self.time_cv = time_cv
        self.r_vec = r_vec
        self.c_lam_entry_value = c_lam_entry_value
        self.times_vec = times_vec
        self.iters_vec = iters_vec
        self.selection_criterion_vec = selection_criterion_vec
        self.convergence = convergence


class OutputPath:

    """
    Definition of the output class for FASTEN solver

    """

    def __init__(self, best_model, k_selection, k_estimation,
                 r_vec, selection_criterion_vec, c_lam_entry_value, c_lam_vec, alpha, lam1_vec, lam2_vec, lam1_max,
                 time_total, time_path, time_cv, time_adaptive, time_curves, iters_vec, times_vec):
        self.best_model = best_model
        self.k_selection = k_selection
        self.k_estimation = k_estimation
        self.r_vec = r_vec
        self.selection_criterion_vec = selection_criterion_vec
        self.c_lam_entry_value = c_lam_entry_value
        self.c_lam_vec = c_lam_vec
        self.alpha = alpha
        self.lam1_vec = lam1_vec
        self.lam2_vec = lam2_vec
        self.lam1_max = lam1_max
        self.time_total = time_total
        self.time_path = time_path
        self.time_cv = time_cv
        self.time_adaptive = time_adaptive
        self.time_curves = time_curves
        self.iters_vec = iters_vec
        self.times_vec = times_vec


