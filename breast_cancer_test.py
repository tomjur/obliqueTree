from TreeClassifier import *
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_breast_cancer_data():
    data = load_breast_cancer()
    return data['data'], data['target']

X, y = get_breast_cancer_data()
treeClassifier = TreeClassifier(0.001, 15, normalizer_mode="norm", feature_drop_probability=0.0)
treeClassifier.fit(X, y)
print treeClassifier.print_tree()

y_hat = treeClassifier.predict_deterministic(X)
print accuracy_score(y, y_hat)

y_hat = treeClassifier.predict_stochastic(X)
print accuracy_score(y, y_hat)

