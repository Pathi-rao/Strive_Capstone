from typing import Tuple, Union, List
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


# The get_model_parameters function returns the model parameters. These are found in the coef_ and intercept_ attributes for LogisticRegression .
def get_model_parameters(model):
    """Returns the paramters of a sklearn LogisticRegression model"""
    if model.fit_intercept:
        params = (model.coef_, model.intercept_) 
    else:
        params = (model.coef_,)
    return params


# The set_model_params function sets/updates the model's parameters. Here care needs to be taken to set the parameters using the same order/index in 
# which they were returned by get_model_parameters.
def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model"""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """
    Sets initial parameters as zeros
    """
    n_classes = 11 # threat types
    n_features = 33 # Number of features in dataset
    model.classes_ = np.array([i for i in range(11)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList: # returns list of Xy (read more about function annotations)
    """Split X and y into a number of partitions."""
    sss = StratifiedShuffleSplit(n_splits=num_partitions, test_size=0.001, random_state=0)

    for train_index, test_index in sss.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # print(f'Unique classes are before zip.. ', len(np.unique(y_train)))
        # print(np.array_split(y_train, num_partitions))
        
    return list(
        zip(np.array_split(X_train, num_partitions), 
        np.array_split(y_train, num_partitions))
    )
