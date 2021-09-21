# do the necessary imports
import warnings
from joblib import dump, load
import flwr as fl
import numpy as np
import datahandler as dh

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn import metrics

import matplotlib.pyplot as plt
# from sklearn.metrics import plot_confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import utils


X_train, X_test, y_train, y_test = dh.data_processor()

class_labels = ['smurf','neptune','normal','satan','ipsweep','portsweep','nmap','back','warezclient','teardrop','pod'] 

# print(f'Shape of X_trian and y_train before partition.... ', X_train.shape, y_train.shape)
# print(f'Unique classes are.. ', len(np.unique(y_train)))
print(' ')
print('Setting up the client.... ')
# Split train set into 10 partitions and randomly use one for training.
print(' ')
partition_id = np.random.choice(10)
(X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

# print(f'Shape of X_trian and y_train after partition.... ', X_train.shape, y_train.shape)
# print(f'Unique classes are.. ', len(np.unique(y_train)))

# Using class weights for imbalanced data
# classes = np.unique(y_train)
# cw = class_weight.compute_class_weight('balanced',
#                                                 classes,
#                                                 y_train)
# weights = dict(zip(classes,cw))


# using SMOTE
# smote = SMOTE('minority')
# X_sm, y_sm = smote.fit_resample(X_train, y_train)


# Create LogisticRegression Model
model = LogisticRegression(
penalty="l2",
max_iter=1, # local epoch
warm_start=True, # prevent refreshing weights when fitting
)
model = LogisticRegression() #class_weight = weights
# Setting initial parameters, akin to model.compile for keras models
utils.set_initial_params(model)


class KDDClient(fl.client.NumPyClient):
    def get_parameters(self): # type: ignore
        return utils.get_model_parameters(model)

    def fit(self, parameters, config): # type: ignore
        utils.set_model_params(model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # print(f'Shape of X_trian and y_train before model fitting.... ', y_train.shape)
            # print(f'Unique classes are.. ', len(np.unique(y_train)))
            # print(y_train)
            
            model.fit(X_train, y_train)
            
            # print(model.classes)
            print(f"Training finished for round {config['rnd']}")

        # dump(model, 'Federated_model.joblib')
        # print("Model has been locally saved.")

        return utils.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config): # type: ignore
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        # accuracy = model.score(X_test, y_test)

        y_test_pred = model.predict(X_test)
        
        conf_matrix = metrics.confusion_matrix(y_test, y_test_pred)  
        fig, ax = plot_confusion_matrix(conf_mat=conf_matrix,
                                colorbar=True,
                                show_absolute=False,
                                show_normed=True,
                                class_names=class_labels)
        plt.show()

        accuracy = metrics.accuracy_score(y_test, y_test_pred)
        return loss, len(X_test), {"accuracyyy": accuracy}

    # def disconnect (self):
    #     fl.common.typing.Disconnect('disconnecting...')

fl.client.start_numpy_client("localhost:8080", client=KDDClient())
