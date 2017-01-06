from TreeClassifier import *
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# gets teh mnist data
mnist = fetch_mldata('MNIST original', data_home=r'c:\temp\mnist')


def create_data_matrix(label0, label1):

    def get_data_for_label(label):
        return mnist.data[mnist.target == label]

    d1 = get_data_for_label(label0)
    d2 = get_data_for_label(label1)
    d = np.concatenate((d1, d2))
    l = np.concatenate((np.zeros(d1.shape[0]), np.ones(d2.shape[0])))

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    return unison_shuffled_copies(d, l)


def accuracy_for_pair(label0, label1, epsilon=0.001, depth=5):
    X, y = create_data_matrix(label0, label1)
    tree_classifier = TreeClassifier(epsilon, depth, normalizer_mode="norm", feature_drop_probability=0.0)
    scores = cross_val_score(tree_classifier, X, y, cv=5)
    return scores

print accuracy_for_pair(3, 5)

# f = open(r"C:\temp\mnist_results.txt", "w")
# for i in range(10):
#     for j in range(i+1, 10):
#         scores = accuracy_for_pair(i, j)
#         f.write(str((i, j)))
#         f.write(str(scores))
# f.close()

