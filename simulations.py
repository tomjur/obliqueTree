from TreeClassifier import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
import numpy as np

depth = 10


def get_simulation_data_1(d=100,n=1000):
    X = np.random.normal(size=[n, d])
    y = np.greater_equal(np.sum(X, axis=1), 0.0)
    return X, y


def get_simulation_data_2(d=100,n=1000):
    X = np.random.normal(size=[n, d])
    w = np.random.normal(size=[d, 1])
    y = np.greater_equal(np.dot(X,w).reshape(-1), 0.0)
    print 'positive class weight {}'.format(float(y.sum()) / len(y))
    return X, y


def get_simulation_data_3(d=100,n=1000):
    X = np.random.normal(size=[n, d])
    w1 = np.random.normal(size=[d, 1])
    w2 = np.random.normal(size=[d, 1])
    y = np.greater_equal(np.multiply(np.dot(X,w1),np.dot(X,w2)).reshape(-1), 0.0)
    print 'positive class weight {}'.format(float(y.sum()) / len(y))
    return X, y

X, y = get_simulation_data_1()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


tree_classifier = TreeClassifier(0.001, depth, normalizer_mode="norm", feature_drop_probability=0.0)
scores = cross_val_score(tree_classifier, X, y, cv=5)
print "stochastic"
print scores.mean()

scores = cross_val_score(tree.DecisionTreeClassifier(max_depth=depth), X, y, cv=5)
print "regular"
print scores.mean()

print 'positive class weight {}'.format(float(y.sum()) / len(y))