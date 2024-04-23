## funGCN

### Functional Multi-modal Multi-task Graph Convolutional Network


--- 

### FILES DESCRIPTION

The folder contains the files to implement the funGCN model. 
    
    
    early_stopper.py: class to implement early stopper criterion
    
    gcn_model.py: class to implement the gcn model 
    
    generate_sim_fgcn: class to generate a simulation for the FunGCN model

    output_classes.py: classes to define the model outputs 

    solver_fgcn.py: class to implement the FunGCN model

---
        
### FunGCN CLASS DESCRIPTION: 

#### METHODS 
    
    standardize_X: 

        Standardize the design matrix.
        This function subtracts the mean and divides by the standard deviation for each column,
        resulting in columns with a mean of 0 and a standard deviation of 1.


    compute_categories_embeddings: 

        Computes embeddings for categorical variables.
        This function creates an embedding layer for a given set of categorical variable values, `x`,
        using a specified embedding dimension, `k`. Each category is represented as a dense vector
        of dimension `k`. The function initializes an embedding layer, applies it to the input data,
        and returns the resulting embeddings as a numpy array.


    compute_graph_embeddings: 

        Computes embeddings for graph data based on variable modality.
        This function processes a dataset to compute embeddings for each feature. 
        For categorical features, it computes category embeddings. For other types of variables, 
        it computes embeddings based on the eigenfunctions of the data matrix. This method allows 
        the handling of mixed data types within a single framework.


    find_forecast_domain: divides the time and the basis coefficients into past and horizon

    compute_gcn_embeddings: computes embeddigns for the GCN model

    update_embeddings: compute and update (if necessary) all the embeddings, calling compute_graph_embeddings 
        and compute_gcn_embeddings

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

#### ATTRIBUTES 

    X_gcn: (nobs, nvar, k_gcn) embedded array for GCN
    
    X_gcn_test: (nobs_test, nvar, k_gcn) embedded array for the GCN testing

    X_gcn_test: (nobs_train, nvar, k_gcn) embedded array for the GCN training

    X_gcn_test:(nobs_val, nvar, k_gcn) embedded array for the GCN validation

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

