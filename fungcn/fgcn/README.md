## funGCN

### Functional Multi-modal Multi-task Graph Convolutional Network


--- 

### FILES DESCRIPTION

The folder contains the files to implement the funGCN model. 
    
    
    early_stopper.py: 

        class to implement early stopper criterion
    
    
    gcn_model.py: 

        class to implement the gcn model 
    
    
    generate_sim_fgcn: 

        class to generate a simulation for the funGCN model

    
    output_classes.py: 

        classes to define the model outputs 

    
    solver_fgcn.py: 

        class to implement the funGCN model

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


    find_forecast_domain:
        
        Find the forecast domain for a given set of parameters.
        The function calculates the starting index for the forecast based on the forecast ratio,
        defines the full grid of points, and calculates the number of interior knots.
        It then generates knots evenly spaced within the data range and separates them into
        historical and forecast knots. The function also prepares the historical and forecast
        knots for basis functions and ensures there are enough knots for the spline basis.


    compute_gcn_embeddings:

        Compute GCN embeddings for a given dataset.
        The function first checks if data and var_modality are provided, and if not, it uses the default values.
        It then extracts the shape of the data and sets the degree of the spline basis to 3.
        If forecast_ratio is greater than 0, it prepares the data for the forecast task by dividing the domain,
        initializing variables, and computing the GCN embeddings for each feature.
        If forecast_ratio is 0, it prepares the data for the regression/classification task by initializing variables
        and computing the GCN embeddings for each feature.


    update_embeddings:
    
        Update the embeddings for the graph and GCN models.
        It updates the embeddings for the graph and GCN models if the input values are different from the current
        values. It also prints some information about the updated embeddings if verbose is True and forecast_ratio
        is greater than 0.


    split_data_train_validation_test:

        Split the data into training, validation, and testing sets.
        This function takes in three inputs: test_size, val_size, and random_state.
        It splits the data into training, validation, and testing sets based on the input sizes.
        It updates the training, validation, and testing indices and saves the corresponding data for the graph
        and GCN models.


    standardize_data:
        
        Standardize the data and create torch tensors for the graph and GCN models.
        It standardizes the data for the graph and GCN models using the standardize_X function.
        It creates torch tensors for the standardized data and stores them in the object's attributes.


    isolate_target_from_data:
        
        Isolate the target variable from the data for the GCN model.
        It separates the target variable from the input data for the GCN model, depending on whether it's a forecast
        task or a regression/classification task. It updates the X and y attributes for the training, validation,
        and testing data.


    preprocess_data:
        
        Preprocess the data for the GCN model.
        It preprocesses the data by updating the embeddings, splitting the data into training, validation,
        and testing sets, standardizing the data, and isolating the target variable.
        It also checks for consistency in the input parameters and raises an exception if necessary.


    graph_estimation: 
    
        Estimate the graph structure using the FASTEN algorithm.
        It estimates the graph structure by solving the FASTEN feature selection problem for each node variable.
        It initializes the solver and parameters, and then iterates over each node variable to estimate the adjacency
        matrix. It makes the adjacency matrix symmetric and upper triangular, and saves it to a file if specified.


    preprocess_adjacency_matrix: 

        Preprocess the adjacency matrix of a graph.
        It takes the adjacency matrix of a graph, prunes it by removing edges with weights below a certain
        threshold, adds self-loops to the matrix, normalizes the matrix, and extracts the edge weights and indices.


    initialize_gcn_module:
        
        Initialize the GCN model.
        It checks the number of edges for target nodes, and initializes the GCN model.


    train_model:

        Train the GCN model.
        It trains the GCN model using the Adam optimizer and early stopping.


    retrieve_original_form:

        Retrieve the original form of the data.
        This function takes the output of the GCN model and transforms it back to the original form by reversing
        the standardization and normalization steps. It also handles different types of variables, such as categorical,
        scalar, and longitudinal variables, and applies specific transformations to each type.


    compute_evaluation_metrics: 
    
        Compute evaluation metrics for the model.
        This function computes the Root Mean Squared Error (RMSE), standardized RMSE, and accuracy
        for each output variable.
        

    predict:

        Predict targets given new observations.
        This function predicts targets given new observations. If new data is None, it predicts  the test observations.
        If y_true is given, it also computes the evaluation metrics.


    print_prediction_metrics:

        Print prediction metrics.
        This function prints the prediction metrics, including RMSE, standardized RMSE, and accuracy, 
        for each output variable.
    

    plot_true_vs_predicted:

        Plot the predicted curves vs the true ones.
        This function plots the predicted curves vs the true ones for a specific target variable and a set of curves.
        It can plot either the test or train data.

#### ATTRIBUTES 

    X_gcn: 
    
        (nobs, nvar, k_gcn) embedded array for GCN
    

    X_gcn_test: 

        (nobs_test, nvar, k_gcn) embedded array for the GCN testing

    
    X_gcn_test: 

        (nobs_train, nvar, k_gcn) embedded array for the GCN training

    
    X_gcn_test: 

        (nobs_val, nvar, k_gcn) embedded array for the GCN validation
    

    X_graph: 
        
        (nvar, nobs, k_graph) embedded array for graph estimation

    
    X_graph_train: 

        (nvar_train, nobs, k_graph) embedded array with observations used for graph_estimation

    
    accuracy_test (dict): 

        keys are categorical targets and values are accuracy on the test sets

    
    adjacency (nvar, nvar): 

        adjancency matrix

    
    adjusted_start_forecast: 

        position of the first basis in the forecasting domain

    
    basis_gcn: 
        
        basis used to represents the longitudinal features for gcn

    
    basis_graph: 

        basis used to represents the longitudinal features for graph estimation

    
    data: 

        input data, array of dimensions (nvar, nobs, ntimes)

    
    edge_index: 

        list of the graph edges

    
    edge_weigths: 

        list of the graph weights

    
    forecast_ratio: 

        portion of prediction with respect to the total domain if the task is forecasting

    
    k_gcn_in: 

        number of coefficients of the GCN input

    
    k_gcn_out: 

        number of coefficients of the GCN output

    
    k_graph:

        number of coefficients for graph estimation

    
    loss_train: 

        loss on the training set

    
    loss_val: 
        
        loss on the validation set

    
    mean_vec_gcn: 

        vector with the original features means, used to reconstruct the orginal features

        
    model: 

        the gcn model

    
    nedges:

        number of edges for each node of the graph

    
    rmse_test (dict): 

        keys are longitudinal/scalar targets and values are the rmse on the test sets

    
    start_forecsat: 

        time at which the forecasting starts

    
    std_rmse_test (dict): 

        keys are longitudinal/scalar targets and values are the standardized rmse on the test sets

    
    std_vec_gcn: 

        vector with the original features standard deviations, used to reconstruct the orginal features

    
    test_ind: 

        index of the test set

    
    train_ind: 

        index of the training set

    
    val_ind: 

        index of the validation set

    
    var_modality: 

        list of modalities for each variables; possible values: 'f', 'c', 's'

    
    verbose: 
    
        if true, the output is printed

    
    y_gcn_train: 

        targets for the gcn restricted on the train set

    
    y_gcn_val: 

        targets for the gcn restricted on the validation set

    
    y_ind: 

        list containing the indeces of the target variables

    
    y_test_hat_embedded: 

        estimated features on the test set and in the embedded space

    
    y_test_hat_os: 

        estimated features on the test set and retransformed in the original space

    
    y_train_hat_embedded: 

        estimated features on the trian set and in the embedded space

    
    y_val_hat_embedded: 

        estimated features on the validation set and in the embedded space

