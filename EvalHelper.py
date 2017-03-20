from TreeClassifier import *
from TreeClassifierWithBias import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.svm import LinearSVC
import numpy as np
import datetime


def eval_single_classifier(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    return accuracy_score(y_test, classifier.predict(X_test))


def file_print(res_tree, res_svm, purity, print_prefix):
    test_time = str(datetime.datetime.now()).replace(':','-').replace(' ','-').replace('.','-')
    with open(r"C:\temp\formatted-results\{}-{}.csv".format(print_prefix, test_time), "w") as f:
        for d in res_tree:
            f.write('depth: {}\n'.format(d))
            f.write(',,CART:,Approx:,Linear SVM:,purities:\n')
            for r in res_tree[d]['CART']:
                i = r[0]
                samples = r[1]
                cart = r[2]
                approx = [a for j, s, a in res_tree[d]['approx'] if j == i and s == samples][0]
                svm = [a for j, s, a in res_svm if j == i and s == samples][0]
                p = [a for j, s, a in purity if j == i and s == samples][0]
                f.write('{},{},{},{},{},{}\n'.format(i, samples, cart, approx, svm, p))
            f.write('\n')


def console_print(res_tree, res_svm, purity):
    print 'purities:'
    for r in purity:
        print '{}: {} samples, p: {}'.format(r[0], r[1], r[2])
    print ''
    for d in res_tree:
        print 'depth: {}'.format(d)
        print 'CART:'
        for r in res_tree[d]['CART']:
            print '{}: {} samples, acc: {}'.format(r[0], r[1], r[2])
        print 'Approx:'
        for r in res_tree[d]['approx']:
            print '{}: {} samples, acc: {}'.format(r[0], r[1], r[2])
        print ''
        print 'Approx with bias:'
        for r in res_tree[d]['approxWithBias']:
            print '{}: {} samples, acc: {}'.format(r[0], r[1], r[2])
        print ''
    print 'Linear SVM:'
    for r in res_svm:
        print '{}: {} samples, acc: {}'.format(r[0], r[1], r[2])


def trim_non_informative_features(data):
    print data.shape
    reduced_data = np.subtract(data, data.mean(axis=0))
    not_zeros = np.not_equal(reduced_data, 0.0)
    occ = np.sum(not_zeros, axis=0)
    data = data[:, np.greater(occ, 0.0)]
    print data.shape
    # print data
    return data


def upsample(data, number):
    added_index = np.random.choice(data.shape[0], number, replace=True)
    data = np.concatenate((data, data[added_index,:]), axis =0)
    return data


def set_same_size(data, target):
    positive_index = np.equal(1.0, target)
    positive_data = data[positive_index]
    negative_data = data[np.logical_not(positive_index)]
    positive_size = positive_data.shape[0]
    negative_size = negative_data.shape[0]
    if positive_size == negative_size:
        return data, target
    if positive_size > negative_size:
        negative_data = upsample(negative_data, positive_size - negative_size)
    else:
        positive_data = upsample(positive_data, negative_size - positive_size)
    data = np.concatenate((positive_data, negative_data), axis=0)
    class_size = np.max([negative_size,positive_size])
    target = np.concatenate((np.ones(class_size), np.zeros(class_size)))
    new_index = np.random.permutation(np.arange(2*class_size))
    data = data[new_index, :]
    target = target[new_index]
    return data, target


def eval_classifiers(X, y, depths=[5], split_data=3, print_prefix=None, normalizer_mode="evalMode"):
    res_tree = {d: {'CART': [], 'approx': [], 'approxWithBias': []} for d in depths}
    res_svm = []
    purity = []
    for i in range(split_data):
        try:
            slice_size = int((i+1)*len(y)/float(split_data))
            indices = np.random.permutation(len(y))[:slice_size]
            # print len(indices)
            sliced_data = X[indices, :]
            sliced_data = trim_non_informative_features(sliced_data)
            # print sliced_data.shape
            sliced_labels = y[indices]
            # print sliced_labels.shape
            # print sliced_labels
            X_train, X_test, y_train, y_test = train_test_split(sliced_data, sliced_labels, test_size=0.2, random_state=0)
            # X_train, X_test = set_same_size(X_train, y_train)
            # print X_train.shape, X_test.shape, y_train.shape, y_test.shape
            # print np.isnan(X_train).any(), np.isnan(X_test).any(), np.isnan(y_train).any(), np.isnan(y_test).any()
            p = float(y_test.sum()) / len(y_test)
            p = np.max([p, 1.0-p])
            purity.append((i, slice_size, p))
            for d in depths:
                res_tree[d]['approx'].append((i, slice_size,
                                  eval_single_classifier(TreeClassifier(0.001, d, normalizer_mode=normalizer_mode,
                                                                        print_debug=False, fit_full_tree=True),
                                                         X_train, X_test, y_train, y_test)))
                res_tree[d]['approxWithBias'].append((i, slice_size,
                                  eval_single_classifier(TreeClassifierWithBias(0.001, d,
                                                                                print_debug=False, fit_full_tree=True),
                                                         X_train, X_test, y_train, y_test)))
                res_tree[d]['CART'].append((i, slice_size,
                                  eval_single_classifier(tree.DecisionTreeClassifier(max_depth=d),
                                                         X_train, X_test, y_train, y_test)))
            res_svm.append((i, slice_size,
                              eval_single_classifier(LinearSVC(),
                                                         X_train, X_test, y_train, y_test)))
        except Exception, e:
            print 'error at {}'.format(i)
            print str(e)
            for d in res_tree:
                res_tree[d]['approxWithBias'] = [x for x in res_tree[d]['approx'] if x[0] < i]
                res_tree[d]['approx'] = [x for x in res_tree[d]['approx'] if x[0] < i]
                res_tree[d]['CART'] = [x for x in res_tree[d]['CART'] if x[0] < i]
            res_svm = [x for x in res_svm if x[0] < i]
            purity = [x for x in purity if x[0] < i]
            break
    console_print(res_tree, res_svm, purity)
    if print_prefix is not None:
        file_print(res_tree, res_svm, purity, print_prefix)
    return res_tree, res_svm, purity
