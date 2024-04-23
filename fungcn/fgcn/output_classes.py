"""
Classes needed to define outputs

"""


class OutputPrediction:

    """
    Definition of the output class for the prediction method

    """

    def __init__(self, y_hat_os, y_hat_embedded, rmse, std_rmse, accuracy):
        self.y_hat_os = y_hat_os
        self.y_hat_embedded = y_hat_embedded
        self.rmse = rmse
        self.std_rmse = std_rmse
        self.accuracy = accuracy
