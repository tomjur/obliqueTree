from TreeClassifier import *
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np

datasets = ['mnist', 'iris', 'breast_cancer', 'polish_companies_y1', 'polish_companies_y2', #5
            'polish_companies_y3', 'polish_companies_y4', 'polish_companies_y5', 'diabetic_retinopathy_debrecen', #9
            'adult', 'dorothea', 'cover'] # adult + cover are bad CART gets ~1.0
current_dataset_index = 8
# depths_of_tree = [1, 3, 5, 10]
depths_of_tree = [3,5,7]
# depth_of_tree = 1

dataset = datasets[current_dataset_index]


def load_arff(filepath):
    from scipy.io.arff import loadarff
    dataAndLabels = loadarff(filepath)[0]
    data = np.array([list(row)[:-1] for row in dataAndLabels])
    target = np.array([int(list(row)[-1]) for row in dataAndLabels])
    keep_index = ~np.isnan(data).any(axis=1)
    data = data[keep_index]
    target = target[keep_index]
    return data, target


def load_dorothea():
    with open(r"C:\temp\thesis_data\dorothea\dorothea_train.data") as f:
        content = f.readlines()
    res = []
    ps = []
    for i in range(len(content)):
        parts = content[i].split()
        for p in parts:
            if len(p)>0:
                ps.append(int(p))
                res.append((i-1, int(p)-1))
    data = np.zeros([len(content),max(ps)])
    for r in res:
        data[r]=1.0
    with open(r"C:\temp\thesis_data\dorothea\dorothea_train.labels") as f:
        content = f.readlines()
    target = np.array([1 if int(l) == 1 else 0 for l in content])
    return data,target


def load_cover():
    import pandas as pd
    df = pd.read_csv(r"C:\Temp\thesis_data\cover-type\covtype.data", header=None)
    return df.values[:, :-1].astype(float), df.values[:, -1]-1


def load_polish(year):
    return load_arff(r"C:\temp\thesis_data\Dane\{}year.arff".format(year))


def load_adult():
    import pandas as pd
    from sklearn.feature_extraction import DictVectorizer

    def encode_onehot(df, cols):
        vec = DictVectorizer()

        vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
        vec_data.columns = vec.get_feature_names()
        vec_data.index = df.index

        df = df.drop(cols, axis=1)
        df = df.join(vec_data)
        return df
    data = pd.read_csv(r"C:\temp\thesis_data\adult\adult.data",header=None)
    data = data.dropna()
    data = encode_onehot(data,[1,3,5,6,7,8,9,13,14])
    data = data.values.astype(float)
    
    return data[:,:-1], data[:,-1].astype(int)


def get_data(dataset):
    if dataset == datasets[0]:
        # gets the mnist data
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original', data_home=r'c:\temp\mnist')
        data = mnist.data
        target = mnist.target
    elif dataset == datasets[1]:
        # gets the iris data
        from sklearn.datasets import load_iris
        data, target = load_iris(True)
    elif dataset == datasets[2]:
        # gets the breast cancer data
        from sklearn.datasets import load_breast_cancer
        breast = load_breast_cancer()
        data = breast['data']
        target = breast['target']
    elif 2 < current_dataset_index < 8:
        # polish companies
        # https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
        data, target = load_polish(current_dataset_index-2)
    elif current_dataset_index == 8:
        # load diabetic dataset
        # https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
        data,target = load_arff(r"C:\temp\thesis_data\messidor_features.arff")
    elif current_dataset_index == 9:
        # http://archive.ics.uci.edu/ml/datasets/Adult
        data,target = load_adult()
    elif current_dataset_index == 10:
        # http://archive.ics.uci.edu/ml/datasets/Dorothea
        data,target = load_dorothea()
    elif current_dataset_index == 11:
        # https://archive.ics.uci.edu/ml/datasets/Covertype
        data,target = load_cover()
    return data, target

data, target = get_data(dataset)


def create_data_matrix(label0, label1):

    def get_data_for_label(label):
        return data[target == label]

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
    res = []
    split = 3
    for i in range(split):
        slice_size = (i+1)*len(y)/float(split)
        indices = np.random.permutation(len(y))[:int(slice_size)]
        sliced_data = X[indices, :]
        sliced_labels = y[indices]
        X_train, X_test, y_train, y_test = train_test_split(sliced_data, sliced_labels, test_size=0.2, random_state=0)
        if is_tree:
            tree_classifier = tree.DecisionTreeClassifier(max_depth=depth)
        else:
            # tree_classifier = TreeClassifier(epsilon, depth, normalizer_mode="dropSize", print_debug=False, fit_full_tree=True)
            tree_classifier = TreeClassifier(epsilon, depth, normalizer_mode="evalMode", print_debug=False, fit_full_tree=True)
        tree_classifier.fit(X_train, y_train)
        res.append((i, accuracy_score(y_test, tree_classifier.predict(X_test))))
    return res
    # return cross_val_score(tree_classifier, X, y, cv=5)

for depth_of_tree in depths_of_tree:
    for is_tree in [True, False]:
    # for is_tree in [True]:
        f = open(r"C:\temp\small\{}_{}_{}.txt".format(dataset, depth_of_tree, is_tree), "w")
        number_of_classes = np.max(target).astype(int) + 1
        print 'number of classes {}'.format(number_of_classes)
        means = []
        for i in range(number_of_classes):
            for j in range(i+1, number_of_classes):
                print (i,j)
                scores = accuracy_for_pair(i, j, depth=depth_of_tree)
                f.write(str((i, j)))
                f.write(' ')
                f.write(str(scores))
                f.write(' ')
                # f.write(str(np.mean(scores)))
                f.write('\n')
                # means += [np.mean(scores)]
        # f.write(str(np.mean(means)))
        f.close()
