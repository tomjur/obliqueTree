from TreeClassifier import *
from sklearn.model_selection import cross_val_score
from sklearn import tree
import numpy as np

datasets = ['mnist', 'iris', 'breast_cancer', 'polish_companies_y1', 'polish_companies_y2', #5
            'polish_companies_y3', 'polish_companies_y4', 'polish_companies_y5', 'diabetic_retinopathy_debrecen', #9
            'adult']
current_dataset_index = 9
depth_of_tree = 5

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
    if dataset == datasets[1]:
        # gets the iris data
        from sklearn.datasets import load_iris
        data, target = load_iris(True)
    if dataset == datasets[2]:
        # gets the breast cancer data
        from sklearn.datasets import load_breast_cancer
        breast = load_breast_cancer()
        data = breast['data']
        target = breast['target']
    if 2 < current_dataset_index < 8:
        # polish companies
        # https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data
        data, target = load_polish(current_dataset_index-2)
    if current_dataset_index == 8:
        # load diabetic dataset
        # https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
        data,target = load_arff(r"C:\temp\thesis_data\messidor_features.arff")
    if current_dataset_index == 9:
        # http://archive.ics.uci.edu/ml/datasets/Adult
        data,target = load_adult()
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
    if is_tree:
        tree_classifier = tree.DecisionTreeClassifier(max_depth=depth)
    else:
        tree_classifier = TreeClassifier(epsilon, depth, normalizer_mode="norm", feature_drop_probability=0.0)
    return cross_val_score(tree_classifier, X, y, cv=5)

# print accuracy_for_pair(3, 5, depth=45)

for is_tree in [True, False]:
    f = open(r"C:\temp\{}_{}_{}.txt".format(dataset, depth_of_tree, is_tree), "w")
    number_of_classes = np.max(target).astype(int) + 1
    print 'number of classes {}'.format(number_of_classes)
    means = []
    for i in range(number_of_classes):
        for j in range(i+1, number_of_classes):
            print (i,j)
            scores = accuracy_for_pair(i, j, depth=5)
            f.write(str((i, j)))
            f.write(' ')
            f.write(str(scores))
            f.write(' ')
            f.write(str(np.mean(scores)))
            f.write('\n')
            means += [np.mean(scores)]
    f.write(str(np.mean(means)))
    f.close()

