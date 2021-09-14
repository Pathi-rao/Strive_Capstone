import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 1, 1, 1, 0, 1, 2, 2, 2])
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.5, random_state=13)
# sss.get_n_splits(X, y)

# print(sss)

for train_index, test_index in sss.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print(X_train, y_train)
# print(X_test, y_test)

a = (np.array_split(X_train, 5), 
        np.array_split(y_train, 5))

print(a)