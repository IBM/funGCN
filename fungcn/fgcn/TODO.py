
"""
TO DO LIST

"""

# DONE: 1) pass directly coefficient form == True (does not have to compute FPC every time and standardize just once)

# DONE: 1a) use pc specific for each variable to represent them
#   first: standardize A
#   second: compute Ac from A
#   third: for FF: take b from Ac, for SF and LOGIT take b from std(A) --
#       check taking b from std(A) is the same that taking b from A and then std(b): done it is the same

# DONE: use solver path logit for categorical variables (now you use SF)

# DONE: tune parameters: max selected, best model, pruning value, k, adaptive scheme and funGCN ones
#       The pruning parameter is very important! For example, 0.7 vs 0.8 for education_mod makes big difference
#  UPDATE: some first findings
#   - lr: larger (1e-4) for categorical target and horizon prediction / smaller (1e-5) for longitudinal targets
#   - k_gcn:  larger (>10) for or longitudinal targets  / smaller (~10) for categorical prediction
#   - optmizer: AdamW(weight_decay=1e-5) RMSprop(alpha=0.9) seem to achieve better performances

# DONE: save b_std for each variables in compute coefficients and then use it to reconstruct the target predicted curves

# DONE: try to represent the categorical variable in k dimension as constant functions but also with aggregation layers
#       (you could use these representations also for the feature selection phase). Add a fake categorical variables
#       with more than k values (just add all the categorical var together), aggregate it and try to use it also to
#       perform classification

# DONE: extend to classification (not just regression).
#       Questions:
#         1) do I do the embeddings before or inside the GCN?
#            If I do it before, I can use the embeddings for constructing the graph, but I do not train it
#            (and maybe improve it) during phase 2 and I need to find a way to train
#            reconstruct the original classification from the embedding
#            If I do it in the GCN, I cannot use the embedding for constructing the class.
#            Can I do it before and then train it during the GCN?
#   IDEA!!! two different Ac, one for graph, one for GCN, they can have different dimension. In Ac_graph you can or
#           cannot use embeddings. In Ac_GCN, you start from [n, m, k] and try two approaches:
#               1) categorical variables are expressed as fpc --> once you estimate the curve/value, you need
#                  to round it
#               2) categorical variables are not transformed --> you take the first value and then embed it in the GCN
#           Test with the accuracy score (or entropy) which approach works better

# DONE: use k different from graph construction and gcn. You can compute basis for k_max and then two different Ac
#       Careful: different options, for creating graph you can: not transform scalar categorical variables and
#       transform them only at a grap stage

# DONE: try to add Conv2d layer

# DONE: add prediction in the future: split curves in two, compute FPC for first and second part, use first part
#       to predict the second

# DONE: extend to predict more than one target

# DONE: extend to predict using not all the nodes --> implement check criterion, how many edges does the target node
#       has? If not connected, impossible to use graph information

# DONE: reconstruct the curves. You: compute coefficients, standardize coefficients. To go back, you have to:
#       inverse standardize coefficients, multiply for basis

# TODO: see if you can adaptevely change some parameters, i.e. learning rate

# DONE: if predict_future equal true, for functional variables:
#           - divide each variable in history and horizon, compute coeff form with k1 and k2 fpc and add them
#             together. Ac will have shape (n, m, k1 + k2)
#           - for all the other modalities k2 are all 0, since we will never use them as targets for future
#             predictions, it does not make sense to fill their values.
#       QUESTION: Which one of these options we use to construct the graph?
#           1) use all the curves with k, and then divide them in a second moment if you want to predict future
#           2) use only history (k1)
#           3) use history + horizon, in this case other modalities will have 0 for all horizon
#       I would say 1), I want to find the connection between all modalities and all the curves (also the horizon),
#       since I am going to use all the modalities to predict the future.

# TODO: raise exception if self are not defined: you have to launch functions before

# DONE: implement a check if predict_future == True, y_ind can contain just 'f' var_modality

# TODO: remember you are using just the train sample to build the graph, not all of them!

# DONE: take into consideration derivative when computing distance. OR try to compute the loss on the whole curve
#  for 'f' variables: to do that you have to retrieve original form and compute the loss based on that.
#  For forecast, you could also force the first prediction point to be the same as the last history point
#  UPDATE: you tried to compute loss on the whole curves, it does not work. We still have flat estimation.
#   Also for the train, flat estimation: try to add derivative.
#  UPDATE: also with derivative did not work, you moved to bsplines

# TODO: tune the patience criteria: the validation sometimes stops too early

# DONE: check standardization for forecast task. It should be correct: you use std and mean of [y_ind] which
#   are not 0, and they are computed independently per k, so we do not mix hist and future: just remember to
#   save only the last part of the vector.
#   Standardizing the original data before passing them as input does not improve estimation, we still have flat curves

# TODO: check standardization in general: you use std and mean computed on everything to reconstruct the test.
#   UPDATE: you tried it, in old_versions_gcn solver_fmmgcn, it does not change the results, choose the version!!

# DONE: with bsplines scalar non constant: you can impute them constant value during compute_embedding.
#   DONE it. Even with constant, the estimation is not constant, I just take the first value
#   YOU TRIED ALL THE COMBINATION
#   bsplines embedding + bsplines[0] (not constant function) (not constant function) --> it seems to work better
#   const embedding + const[0] (not constant function) estimation
# DONE: in forecast settings, for scalar you can use: fpc, bsplines or constant

# DONE: compare fpc and bsplines
#   for prediction not forecast first comparisons: bspline seems to work a bit better

# TODO: comments modules and rewrite readme