from TreeClassifier import *
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score


def get_breast_cancer_data():
    data = load_breast_cancer()
    return data['data'], data['target']

X, y = get_breast_cancer_data()
treeClassifier = TreeClassifier(0.05, 3, normalizer_mode="norm", feature_drop_probability=0.0)
treeClassifier.create_tree(X, y)
y_hat = treeClassifier.predict_deterministic(X)

print treeClassifier.print_tree()
print accuracy_score(y, y_hat)

