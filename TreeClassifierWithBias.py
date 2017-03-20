from BaseClassifier import *
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np


class TreeNodeWithBias:

    def __init__(self, bias_to_data, data_dist, labels, epsilon):
        # data
        self.bias_to_data = bias_to_data
        self.data_dist = data_dist
        self.labels = labels
        self.epsilon = epsilon

        # internal data structures
        self.left = None
        self.right = None
        self.is_leaf = True
        self.classifier = None
        self.bias = None
        self.gain = 0.0
        self.weight = data_dist.sum()

        # set majority vote
        self.m = np.multiply(data_dist, labels).sum() / self.weight
        self.label = np.rint(self.m).astype(int)

    def set_classifier(self):
        if self.classifier is not None:
            return
        normalized_data_dist = np.divide(self.data_dist, self.weight)
        normalized_epsilon = self.epsilon / self.weight
        normalized_epsilon = np.min([0.5, normalized_epsilon, 10*self.epsilon])
        bias_unset=True
        gains = []
        for bias in self.bias_to_data:
            classifier = BaseClassifier(normalized_epsilon)
            classifier.approximate_solver(self.bias_to_data[bias], self.labels, normalized_data_dist)
            gain = self.weight*(classifier.get_ginni(self.m) - classifier.g)
            gains.append(gain)
            if np.greater(gain, self.gain):
                # set classifier if better
                self.classifier = classifier
                self.gain = gain
                self.bias = bias
                bias_unset = False
        if bias_unset:
            print gains

    def set_as_internal(self):
        pl1 = self.classifier.get_probabilities(self.bias_to_data[self.bias])
        pl0 = np.subtract(1.0, pl1)
        self.is_leaf = False
        self.left = TreeNodeWithBias(self.bias_to_data, np.multiply(self.data_dist, pl1), self.labels, self.epsilon)
        self.right = TreeNodeWithBias(self.bias_to_data, np.multiply(self.data_dist, pl0), self.labels, self.epsilon)
        return self.left, self.right

    def get_gain(self):
        if np.equal(self.m, 0.0) or np.equal(self.m, 1.0):
            return -1.0
        self.set_classifier()
        return self.gain

    def get_probabilities(self, bias_to_data, depth=-1):
        if self.is_leaf or depth == 0:
            return np.multiply(np.ones(bias_to_data[bias_to_data.keys()[0]].shape[0]), self.m)

        data = bias_to_data[self.bias]
        current_probabilities = self.classifier.get_probabilities(data)
        left = np.multiply(current_probabilities, self.left.get_probabilities(bias_to_data, depth=depth - 1))
        right = np.multiply(np.subtract(1.0, current_probabilities),
                            self.right.get_probabilities(bias_to_data, depth=depth - 1))
        return np.add(left, right)

    def predict_deterministic(self, bias_to_data, depth=-1):
        return np.greater_equal(self.get_probabilities(bias_to_data, depth), 0.5)

    def predict_stochastic(self, bias_to_data, depth=-1):
        current_probabilities = self.get_probabilities(bias_to_data, depth)
        random_coin = np.random.uniform(0.0, 1.0, current_probabilities.shape)
        return np.greater_equal(current_probabilities, random_coin)


class TreeClassifierWithBias(BaseEstimator, ClassifierMixin):

    def __init__(self, epsilon=0.001, number_of_iterations=5, fit_full_tree=False, print_debug=True, normalize_to_one=True):
        # hyper parameters
        self.epsilon = epsilon
        self.number_of_iterations = number_of_iterations
        self.normalize_to_one = normalize_to_one
        self.initial_normalizer = None
        self.normalizers = None

        self.fit_full_tree = fit_full_tree
        self.print_debug = print_debug

        # model
        self.root = None

    def set_normalizers(self,data):
        if self.normalizers is not None:
            return
        self.initial_normalizer = GenericNormalizer(reduce_mean=True, divide_by_std=True,
                                                    bias_coordinate=0.0, normalize_to_one=self.normalize_to_one)
        norm_data = self.initial_normalizer.normalize_train(data)
        data_min = np.min(np.abs(norm_data))
        data_max = np.max(np.abs(norm_data))
        self.normalizers = {}
        while np.less(data_min, 10*data_max):
            self.normalizers[data_min] = GenericNormalizer(reduce_mean=False, divide_by_std=False,
                                                           bias_coordinate=data_min, normalize_to_one=self.normalize_to_one)
            data_min *= 10.0

    def fit(self, data, labels):
        self.set_normalizers(data)
        data = self.initial_normalizer.normalize_test(data)
        bias_to_data = {b:self.normalizers[b].normalize_train(data) for b in self.normalizers}
        if self.fit_full_tree:
            return self.fit2(bias_to_data , labels)
        return self.fit1(bias_to_data , labels)

    def fit1(self, bias_to_data, labels):
        # initial data distribution
        data_dist = np.ones(labels.shape) / labels.shape[0]
        root = TreeNodeWithBias(bias_to_data, data_dist, labels, self.epsilon)
        leaves = [root]
        for t in range(0, self.number_of_iterations):
            if self.print_debug:
                print "iteration {} from {}".format(t, self.number_of_iterations)
            # get maximal gain (weighted)
            gains = [l.get_gain() for l in leaves]
            if np.less_equal(np.max(gains), 0.0):
                if self.print_debug:
                    print 'negative gain aborting, gains:'
                if self.print_debug:
                    print gains
                break
            leaf_index = np.argmax(gains)
            if self.print_debug:
                print "splitting leaf with weight {} and purity {}".format(leaves[leaf_index].weight, leaves[leaf_index].m)
            # set that leaf as internal and get the split
            l, r = leaves[leaf_index].set_as_internal()
            if self.print_debug:
                print "l_child weight {} and purity {}".format(l.weight, l.m)
            if self.print_debug:
                print "r_child weight {} and purity {}".format(r.weight, r.m)
            # remove now-internal-node from list of leaves
            leaves.pop(leaf_index)
            # add the newly discovered leaves as potential splits
            leaves.extend([l, r])
        self.root = root

    def fit2(self, bias_to_data, labels):
        data_dist = np.ones(labels.shape) / labels.shape[0]
        root = TreeNodeWithBias(bias_to_data, data_dist, labels, self.epsilon)
        self.recursive_fit(self.number_of_iterations, root)
        self.root = root

    def recursive_fit(self, levels, tree_node):
        if tree_node.weight < (1.0 / tree_node.labels.shape[0]) or levels < 2:
            if self.print_debug:
                print "stopping level {}".format(levels)
            return
        if self.print_debug:
            print "splitting leaf with weight {} and purity {}".format(tree_node.weight, tree_node.m)
        tree_node.get_gain()
        l, r = tree_node.set_as_internal()
        if self.print_debug:
            print "l_child weight {} and purity {}".format(l.weight, l.m)
        if self.print_debug:
            print "r_child weight {} and purity {}".format(r.weight, r.m)
        self.recursive_fit(levels-1, l)
        self.recursive_fit(levels-1, r)

    def predict_deterministic(self, data, depth=-1):
        return self.root.predict_deterministic({b:self.normalizers[b].normalize_test(data) for b in self.normalizers}, depth)

    def predict_stochastic(self, data, depth=-1):
        return self.root.predict_stochastic({b:self.normalizers[b].normalize_test(data) for b in self.normalizers}, depth)

    def predict(self, data, y=None):
        return self.predict_deterministic(data)

    def score(self, data, y=None):
        return accuracy_score(y, self.predict(data))

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        res = "\t"*level + str(node.m) + '/' + str(node.weight) + '\n'
        if not node.is_leaf:
            res += self.print_tree(node.left, level+1)
            res += self.print_tree(node.right, level+1)
        return res


def run_main():
    import matplotlib.pyplot as plt

    def create_sized_data():
        X = np.random.uniform(-2, 2, [300, 2])
        sizes = np.reshape(np.sqrt(np.square(X).sum(axis=1)), (-1, 1)).reshape([-1])
        keep = np.logical_or(np.less_equal(sizes, 0.75), np.greater_equal(sizes, 1.25))
        # X = np.delete(X, to_remove,axis=0)
        X = X[keep]
        sizes = np.reshape(np.sqrt(np.square(X).sum(axis=1)), (-1, 1)).reshape([-1])
        y = np.greater_equal(sizes, 1.0)
        return X, y


    def create_data_simulation_middle():
        X1 = np.random.multivariate_normal([1.0,1.0], [[0.05,0.0],[0.0,0.05]], size = 50)
        y1 = np.ones([50,])
        X2 = np.random.multivariate_normal([-1.0,-1.0], [[0.05,0.0],[0.0,0.05]], size = 50)
        y2 = np.ones([50,])
        X3 = np.random.multivariate_normal([0.0,0.0], [[0.05,0.0],[0.0,0.05]], size = 50)
        # X3 = np.random.multivariate_normal([-3.0,3.0], [[0.05,0.0],[0.0,0.05]], size = 50)
        y3 = np.zeros([50,])
        return np.concatenate([X1,X2,X3], axis =0), np.concatenate([y1,y2,y3], axis =0)

    def create_data_simulation_xor():
        X1 = np.random.multivariate_normal([1.0,1.0], [[0.05,0.0],[0.0,0.05]], size = 50)
        y1 = np.ones([50,])
        X2 = np.random.multivariate_normal([-1.0,-1.0], [[0.05,0.0],[0.0,0.05]], size = 50)
        y2 = np.ones([50,])
        X3 = np.random.multivariate_normal([-1.0,1.0], [[0.05,0.0],[0.0,0.05]], size = 50)
        y3 = np.zeros([50,])
        X4 = np.random.multivariate_normal([1.0,-1.0], [[0.05,0.0],[0.0,0.05]], size = 50)
        y4 = np.zeros([50,])
        return np.concatenate([X1,X2,X3,X4], axis =0), np.concatenate([y1,y2,y3,y4], axis =0)


    # X, y = create_data_simulation_middle()
    X, y = create_data_simulation_xor()
    # X, y = create_sized_data()
    treeClassifier = TreeClassifierWithBias(0.05, 3, print_debug=False, normalize_to_one=True)
    treeClassifier.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    y_hat1 = treeClassifier.predict_deterministic(X)
    z1 = treeClassifier.predict_deterministic(np.c_[xx.ravel(), yy.ravel()])
    z1 = z1.reshape(xx.shape)
    # y_hat2 = treeClassifier.root.classifier.predict_deterministic(X)
    # z2 = treeClassifier.root.classifier.predict_deterministic(np.c_[xx.ravel(), yy.ravel()])
    y_hat2 = treeClassifier.predict_deterministic(X, 1)
    z2 = treeClassifier.predict_deterministic(np.c_[xx.ravel(), yy.ravel()], 1)
    z2 = z2.reshape(xx.shape)

    def fit_x2(w):
        return lambda x:  np.divide(w[0], -w[1])*x

    def all_boundaries(xs, depth=-1):
        def get_ys_for_tree_node(node, depth):
            if node.is_leaf or depth == 0:
                return []
            res = [fit_x2(node.classifier.w)(xs)]
            res.extend(get_ys_for_tree_node(node.left, depth-1))
            res.extend(get_ys_for_tree_node(node.right, depth-1))
            return res
        return get_ys_for_tree_node(treeClassifier.root, depth)

    plt.figure()
    # plt.subplot(131)
    # plt.title("True", fontsize='small')
    # plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    # plt.subplot(132)
    plt.subplot(111)
    plt.contourf(xx, yy, z1, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.title("Predicted", fontsize='small')
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_hat1)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # plt.subplot(133)
    # plt.contourf(xx, yy, z2, cmap=plt.cm.coolwarm, alpha=0.8)
    # plt.title("Predicted just root", fontsize='small')
    # plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_hat2)
    # plt.xlim(xx.min(), xx.max())
    # plt.ylim(yy.min(), yy.max())
    plt.show()
    print 'done'

if __name__ == "__main__":
    import cProfile
    cProfile.run('run_main()', sort=1)
