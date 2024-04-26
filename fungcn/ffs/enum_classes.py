"""

Definition of classes needed to specify the problem:

    class RegressionType: definition of the regression type
        FS = Function-on-Scalar
        FF = Function-on-Function
        SF = Scalar-on-Function
        FMix = Function-on-mixed features
        SMix = Scalar-on-mixed features

    class SelectionCriteria: definition of the selection criterion to evaluate the best model
        CV = Cross Validation
        GCV = Generalized Cross Validation
        EBIC = Extended-BIC

    class AdaptiveScheme: definition of the adaptive scheme
        NONE = No adaptive scheme is performed
        SOFT = The adaptive step is performed just on the optimal value of lambda
        FULL = a new path is investigated starting from the weights obtained at the previous path\

    class FPCFeatures:
        response = FPC of b are used for the features
        features = each feature uses itw own FPC
"""

import enum


class RegressionType(enum.Enum):

    """
    Definition of the class to select regression type

    """

    FS = 1
    FF = 2
    SF = 3
    FMix = 4
    SMix = 5
    Logit = 6


class SelectionCriteria(enum.Enum):

    """
    Definition of the class to select model selection criterion

    """

    GCV = 1
    EBIC = 2
    CV = 3


class AdaptiveScheme(enum.Enum):

    """
    Definition of the class to select adaptive scheme

    """

    NONE = 1
    FULL = 2
    SOFT = 3


class FPCFeatures(enum.Enum):

    """
    Definition of the class to select FPC for the features

    """

    features = 1
    response = 2
