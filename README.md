# funGCN

### Functional Graph Convolutional Networks and Feature Selection

The repo contains the code to perform two models: 

1) **Functional adaptive Feature Selection with elastic-net penalty (ffs)** using a Dual Augmented Lagrangian algorithm for the following models:
    * *Function-on-Function* (FF)
    * *Function-on-Scalar* (FS)
    * *Scalar-on-Function* (SF)
    * *Logit* (with functional features and categorical response)


2) **Functional Graph Convolutional Networks (fgcn)**. 
The model estimates a knowledge graph and implements a GCN to work with multi-modal (longitudinal, scalar, and categorical) data and cuncurrently 
perform different tasks (regression, classification, forecasting).

--- 

The implemented methodology is described in the following papers:

- [A Highly-Efficient Group Elastic Net Algorithm with an Application to Function-On-Scalar Regression](https://proceedings.neurips.cc/paper/2021/hash/4d410063822cd9be28f86701c0bc3a31-Abstract.html)
- [FAStEN: an efficient adaptive method for feature selection and estimation in high-dimensional functional regressions](https://arxiv.org/abs/2303.14801)
- [A new computationally efficient algorithm to solve Feature Selection for Functional Data Classification in high-dimensional spaces](https://arxiv.org/abs/2401.05765)
- [Functional Graph Convolutional Networks: A unified multi-task and multi-modal learning framework to facilitate health and social-care insights](https://arxiv.org/abs/2403.10158)

--- 

### FILES DESCRIPTION

    expes --------------------------------------------------------------------------------------------------------------

        ffs 
        
            expes/ffs/sim_FF.py:
              file to run feature selection on synthetic data for the Function-On-Function model 

            expes/ffs/sim_FS.py:
              file to run feature selection on synthetic data for the Function-On-Scalar model
            
            expes/ffs/sim_logit.py:
              file to run feature selection on synthetic data for the Logit model

            expes/ffs/sim_SF.py:
              file to run feature selection on synthetic data for the Scalar-On-Function model
        
        fgcn 
            
            expes/fgcn/sim_fgcn.py:
              file to run fungcn on synthetic data


    fungcn -------------------------------------------------------------------------------------------------------------
        
        fungcn/ffs:
          folder containing the files to implement ffs. More info is in the folder readme
    
        fasten/fgcn:
          folder containing the files to implement fmmgcn. More info is in the folder readme
    
### REQUIREMENTS

1) Create a python3.11 environment: `conda create -n my_env python=3.12`
2) clone funGCN (main branch)
3) install funGCN and required packages by running `pip install -e .` 
at the root of the repository, i.e. where the setup.py file is.
4) Lunch the desired experiments, e.g., `python expes/ffs/sim_FF.py`

**For Apple M processors' users:**
* To install `numpy` with the apple library `vecLib` (which is optimized for Apple processors) run:
    ```
    pip install cython pybind11
    pip install numpy cython
    ```
   
