from TreeClassifier import *
from TreeClassifierWithBias import *
import matplotlib.pyplot as plt


def get_data_current_node(tree_node, title_so_far, my_id, data, grid_data):
    my_title = title_so_far + '->' + str(my_id)

    return my_id, my_title, tree_node.data_dist, tree_node.classifier.get_probabilities(data), \
           tree_node.classifier.get_probabilities(grid_data)


def visualize_tree(data, labels, grid_data, grid_labels, classifier):
    data = classifier.normalizer.normalize_test(data)
    grid_data = classifier.normalizer.normalize_test(grid_data)

    queue = [(classifier.root, '')]

    res = []

    last_id = -1
    while len(queue) > 0:
        last_id += 1
        node, title_so_far = queue.pop(0)
        my_res = get_data_current_node(node,title_so_far,last_id, data, grid_data)
        res.append(my_res)
        if node.left is not None and node.left.classifier is not None:
            queue.append((node.left, my_res[1]))
        if node.right is not None and node.right.classifier is not None:
            queue.append((node.right, my_res[1]))

    return res

def visualize_bias_tree(data, labels, grid_data, grid_labels, classifier):
    data = classifier.initial_normalizer.normalize_test(data)
    grid_data = classifier.initial_normalizer.normalize_test(grid_data)
    bias_to_data = {b: classifier.normalizers[b].normalize_test(data) for b in classifier.normalizers}
    bias_to_grid_data = {b: classifier.normalizers[b].normalize_test(grid_data) for b in classifier.normalizers}

    queue = [(classifier.root, '')]

    res = []

    last_id = -1
    while len(queue) > 0:
        last_id += 1
        node, title_so_far = queue.pop(0)
        my_res = get_data_current_node(node, title_so_far,last_id, bias_to_data[node.bias], bias_to_grid_data[node.bias])
        res.append(my_res)
        if node.left is not None and node.left.classifier is not None:
            queue.append((node.left, my_res[1]))
        if node.right is not None and node.right.classifier is not None:
            queue.append((node.right, my_res[1]))

    return res

def pred_hyperplane(X):
    w = np.ones((2, 1), dtype=np.float32)
    y = np.dot(X, w)
    y = np.greater(y, 0.0).reshape(-1)
    return y

def pred_XOR(X):
    w = np.ones((2, 2), dtype=np.float32)
    w[0, 0] = -1.0
    y = np.dot(X, w)
    y = np.prod(y, axis=1)
    y = np.greater(y, 0.0).reshape(-1)
    return y


def pred(X):
    return pred_hyperplane(X)

X = np.random.uniform(low=-1, high=1, size=(1500, 2))
y = pred(X)
margin = 0.2
x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
h = .02  # step size in the mesh
x1, x2 = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_data = np.c_[x1.ravel(), x2.ravel()]
grid_labels = pred(grid_data)
grid_labels = grid_labels.reshape(x1.shape)

# plt.figure()
# plt.subplot(1,1,1)
# plt.contourf(x1, x2, grid_labels, cmap=plt.cm.coolwarm, alpha=0.8)
# plt.title("Predicted", fontsize='small')
# plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
# plt.xlim(x1.min(), x1.max())
# plt.ylim(x2.min(), x2.max())
# plt.show()

depth = 3
epsilon = 0.001
fit_full_tree = False
use_bias = False

if use_bias:
    treeClassifier = TreeClassifierWithBias(epsilon, depth, normalize_to_one="False", print_debug=False, fit_full_tree=fit_full_tree)
    treeClassifier.fit(X, y)

    res = visualize_bias_tree(X, y, grid_data, grid_labels, treeClassifier)
else:
    treeClassifier = TreeClassifier(epsilon, depth, normalizer_mode="no", print_debug=False, fit_full_tree=fit_full_tree)
    treeClassifier.fit(X, y)

    res = visualize_tree(X, y, grid_data, grid_labels, treeClassifier)

size_factor = 4000

f, axarr = plt.subplots(len(res), 2)
f.tight_layout()
for i,(my_id, my_title,data_dist, pred, grid_pred) in enumerate(res):
    # draw data dist
    axarr[i, 0].set_title(my_title)
    axarr[i, 0].contourf(x1, x2, grid_labels, cmap=plt.cm.coolwarm, alpha=0.8)
    axarr[i, 0].scatter(X[:, 0], X[:, 1], marker='o', c=y, s=data_dist*size_factor)
    # axarr[i, 0].title(my_title, fontsize='small')

    # draw prediction
    axarr[i, 1].set_title(my_title)
    axarr[i, 1].contourf(x1, x2, np.greater(grid_pred, 0.5).reshape(x1.shape), cmap=plt.cm.coolwarm, alpha=0.8)
    axarr[i, 1].scatter(X[:, 0], X[:, 1], marker='o', c=np.greater(pred, 0.5), s=data_dist*size_factor)
    # axarr[i, 1].title(my_title, fontsize='small')

plt.show()
