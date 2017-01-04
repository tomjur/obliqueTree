from TreeClassifier import *
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def get_breast_cancer_data():
    data = load_breast_cancer()
    return data['data'], data['target']

X, y = get_breast_cancer_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


tree_classifier = TreeClassifier(0.001, 5, normalizer_mode="norm", feature_drop_probability=0.0)
tree_classifier.fit(X_train, y_train)
print tree_classifier.print_tree()

acc = tree_classifier.score(X_test, y_test)
print acc

scores = cross_val_score(tree_classifier, X, y, cv=5)
print scores

