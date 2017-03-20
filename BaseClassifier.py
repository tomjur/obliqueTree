
# coding: utf-8

import numpy as np
from operator import itemgetter
from abc import ABCMeta, abstractmethod


class DataNormalizer:
    def __init__(self):
        pass

    __metaclass__ = ABCMeta

    @abstractmethod
    def normalize_train(self, data): pass

    @abstractmethod
    def normalize_test(self, data): pass


class NormOneNormalizer(DataNormalizer):

    def __init__(self):
        super(NormOneNormalizer, self).__init__()
        self.data_mean = None

    def normalize_data(self, data):
        # subtract mean
        data = np.subtract(data, self.data_mean)
        # add coordinate to match scale of data
        res = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)
        # normalize
        row_squared_sums = np.reshape(np.sqrt(np.square(res).sum(axis=1)), (-1, 1))
        res = np.divide(res, row_squared_sums)
        # add bias coordinate
        res = np.concatenate((res, np.ones((data.shape[0], 1))), axis=1)
        # normalize
        res = np.divide(res, np.sqrt(2.0))
        return res

    def normalize_test(self, data):
        return self.normalize_data(data)

    def normalize_train(self, data):
        self.data_mean = data.mean(axis=0)
        return self.normalize_data(data)


class DropSizeNormalizer(DataNormalizer):

    def __init__(self):
        super(DropSizeNormalizer, self).__init__()
        self.data_mean = None

    def normalize_data(self, data):
        # subtract mean
        data = np.subtract(data, self.data_mean)
        # add coordinate to match scale of data
        # res = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)
        # normalize
        row_squared_sums = np.reshape(np.sqrt(np.square(data).sum(axis=1)), (-1, 1))
        res = np.divide(data, row_squared_sums)
        # add bias coordinate
        res = np.concatenate((res, np.ones((data.shape[0], 1))), axis=1)
        # normalize
        res = np.divide(res, np.sqrt(2.0))
        return res

    def normalize_test(self, data):
        return self.normalize_data(data)

    def normalize_train(self, data):
        self.data_mean = data.mean(axis=0)
        return self.normalize_data(data)


class JustBiasNormalizer(DataNormalizer):

    def __init__(self):
        super(JustBiasNormalizer, self).__init__()
        self.data_mean = None

    def normalize_data(self, data):
        # subtract mean
        res = np.subtract(data, self.data_mean)
        # add coordinate to match scale of data
        # res = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)
        # normalize
        # row_squared_sums = np.reshape(np.sqrt(np.square(res).sum(axis=1)), (-1, 1))
        # res = np.divide(res, row_squared_sums)
        # add bias coordinate
        res = np.concatenate((res, np.ones((data.shape[0], 1))), axis=1)
        # normalize
        res = np.divide(res, np.sqrt(2.0))
        return res

    def normalize_test(self, data):
        return self.normalize_data(data)

    def normalize_train(self, data):
        self.data_mean = data.mean(axis=0)
        return self.normalize_data(data)


class RangeNormalizer(DataNormalizer):

    def __init__(self):
        super(RangeNormalizer, self).__init__()
        self.x_min = None
        self.x_max = None
        self.range = None

    def normalize_test(self, data):
        return np.divide(np.subtract(np.multiply(2.0, data), np.add(self.x_max, self.x_min)), self.range)

    def normalize_train(self, data):
        self.x_min = data.min(axis=0)
        self.x_max = data.max(axis=0)
        self.range = np.subtract(self.x_max, self.x_min)
        self.range[np.less_equal(self.range, 0.0)] = 1e-06
        return self.normalize_test(data)


class GenericNormalizer(DataNormalizer):

    def __init__(self, reduce_mean=True, divide_by_std=True, normalize_to_one=True, size_coordinate=False, bias_coordinate=1.0):
        super(GenericNormalizer, self).__init__()
        self.reduce_mean = reduce_mean
        self.divide_by_std = divide_by_std
        self.normalize_to_one = normalize_to_one
        self.size_coordinate = size_coordinate
        self.bias_coordinate = bias_coordinate

        self.data_mean = None
        self.data_std = None

    def normalize_test(self, data):
        res = data
        if self.reduce_mean:
            res = np.subtract(res, self.data_mean)
        if self.divide_by_std:
            res = np.divide(res, self.data_std)
        if self.normalize_to_one:
            if self.size_coordinate:
                # add coordinate to match scale of data
                res = np.concatenate((res, np.ones((res.shape[0], 1))), axis=1)
            # normalize
            row_squared_sums = np.reshape(np.sqrt(np.square(res).sum(axis=1)), (-1, 1))
            res = np.divide(res, row_squared_sums)
        if self.bias_coordinate:
            # add bias coordinate
            res = np.concatenate((res, self.bias_coordinate*np.ones((res.shape[0], 1))), axis=1)
            # normalize
            if self.normalize_to_one:
                row_squared_sums = np.reshape(np.sqrt(np.square(res).sum(axis=1)), (-1, 1))
                res = np.divide(res, row_squared_sums)
        return res

    def normalize_train(self, data):
        if self.reduce_mean:
            self.data_mean = data.mean(axis=0)
        if self.divide_by_std:
            self.data_std = data.std(axis=0)
        return self.normalize_test(data)


class BaseClassifier:

    def __init__(self, epsilon, inter_cell_bound=100, algorithm_mode=None):
        # hyper parameters
        self.epsilon = epsilon
        self.interCellBound = inter_cell_bound
        self.cell_step = epsilon / inter_cell_bound
        self.algorithm_mode = algorithm_mode  # if None, to be inferred when the data is available

        # internal values for classifiers
        self.a = None
        self.b = None
        self.m = -1.0
        self.lambda_ = -1.0

        # outputs
        self.w = None
        self.g = None

    def set_globals(self, data, labels, data_dist):
        weighted_label = np.multiply(data_dist, labels)
        self.m = weighted_label.sum()
        self.a = np.squeeze(np.dot(data_dist.transpose(), data))
        if len(self.a.shape) == 0:
            self.a = np.reshape(self.a, (-1))
        self.b = np.squeeze(np.dot(weighted_label.transpose(), data))
        if len(self.b.shape) == 0:
            self.b = np.reshape(self.b, (-1))

    def set_dependency_coefficient(self):
        lambda_ = np.true_divide(self.b, self.a)
        first = lambda_.flatten()[0]
        if np.all(np.equal(lambda_, first)):
            self.lambda_ = first
        else:
            self.lambda_ = 0.0

    def generate_p_candidates(self, lower, upper, must_occur=[]):
        p = np.arange(lower, upper, self.cell_step)
        p = np.append(p, [upper])
        p = np.append(p, [x for x in must_occur if np.less_equal(x, upper) and np.less(lower, x)])
        p = np.unique(p)
        return p

    def generate_q_candidates(self, p, lower_q=0.0, upper_q=1.0):
        # these are the 3 lines that define the Q_w ranges
        lower_q = np.max([0.0, np.sum([self.m-1.0, p])])
        middle_q = np.multiply(self.m, p)
        upper_q = np.min([self.m, p])

        def generate_range_helper(lower, upper):
            lower = np.max([lower_q, lower])
            upper = np.min([upper_q, upper])
            if np.less(upper, lower):
                return []
            if np.equal(lower, upper):
                return [upper]
            total_range = np.subtract(upper, lower)
            scaled_step = np.multiply(self.cell_step, total_range)  # step is relative to range
            q = np.arange(lower, upper, scaled_step)
            q = np.append(q, [upper])
            return q

        # calculate the two ranges and combine
        qs = np.append(generate_range_helper(lower_q, middle_q), generate_range_helper(middle_q, upper_q))
        qs = np.unique(qs)
        return qs

    @staticmethod
    def get_ginni(x):
        return np.multiply(4.0, np.multiply(x, np.subtract(1.0, x)))

    def get_weighted_ginni(self, p, q):
        if np.any(np.isnan(p)) or np.any(np.isnan(q)) or np.any(np.isnan(self.m)):
            raise Exception('Invalid ginni')
        term1 = np.multiply(p, self.get_ginni(np.divide(q, p)))
        q_tag = np.subtract(self.m, q)
        p_tag = np.subtract(1.0, p)
        term2 = np.multiply(p_tag, self.get_ginni(np.divide(q_tag, p_tag)))
        res = np.add(term1, term2)
        # make sure no result is nan
        t1 = np.multiply(self.get_ginni(self.m), np.isnan(res))
        res = np.nan_to_num(res)
        return np.add(t1, res)

    def approximate_dependent(self):
        a_norm = np.linalg.norm(self.a)
        self.w = np.divide(self.a, a_norm)
        self.g = 4*(self.m - (self.m + self.lambda_ * a_norm)**2 /(0.5+0.5*a_norm) -
                    (self.m - self.lambda_ * a_norm)**2 /(0.5-0.5*a_norm))

        # must_occur = [0.5, self.m, 1.0 - self.m]
        #
        # def get_intersection_points():
        #     alphas = np.arange(0.0, 1.0, self.cell_step)
        #     alphas = np.append(alphas, [1.0])
        #
        #     lambda_m = np.multiply(self.lambda_, self.m)
        #     p1 = [(lambda_m-1.0) /
        #           (2.0*(alpha*lambda_m - 1.0)) for alpha in alphas]
        #     p1 = [p for p in p1 if np.less_equal(0.0, p) and np.less_equal(p, 1.0-self.m)]
        #
        #     p2and3 = [(2.0*alpha*lambda_m+4.0*self.lambda_-lambda_m-1.0) /
        #               (2.0*(alpha*lambda_m+self.lambda_-alpha*self.lambda_-1.0)) for alpha in alphas]
        #     p2and3 = [p for p in p2and3 if np.less_equal(1.0-self.m, p) and np.less_equal(p, 1.0)]
        #
        #     p4and5 = [(lambda_m-1.0) /
        #               (2.0*(alpha*self.lambda_+lambda_m-alpha*lambda_m-1.0)) for alpha in alphas]
        #     p4and5 = [p for p in p4and5 if np.less_equal(0.0, p) and np.less_equal(p, self.m)]
        #
        #     p6 = [(lambda_m-1.0-2.0*alpha*lambda_m) /
        #           (2.0*(lambda_m-alpha*lambda_m-1.0)) for alpha in alphas]
        #     p6 = [p for p in p6 if np.less_equal(self.m, p) and np.less_equal(p, 1.0)]
        #
        #     return np.append(np.append(p1, p2and3), np.append(p4and5, p6))
        #
        # p_candidates = self.generate_p_candidates(0.0, 1.0, must_occur)
        # p_candidates = np.unique(np.append(p_candidates, get_intersection_points()))
        #
        # b_norm = np.linalg.norm(self.b)
        # const_term = 0.5*(self.m - 1.0 / self.lambda_)
        # pair_candidates = [(p, p/self.lambda_ + const_term) for p in p_candidates]
        # pair_candidates = [(p, q) for p, q in pair_candidates if
        #                    np.less_equal(np.max([0.0, self.m+p-1.0]), q) and np.greater_equal(np.min([self.m, p]), q)]
        # pair_candidates = [(self.get_weighted_ginni(p, q), p, q) for p, q in pair_candidates]
        #
        # pair_candidates.sort()
        #
        # for g, p, q in pair_candidates:
        #     beta = 2.0*q-self.m
        #     if np.less_equal(np.square(beta), 1.0):
        #         self.w = np.multiply(beta/np.square(b_norm), self.b)
        #         self.g = g
        #         return
        # raise Exception('Invalid situation')

    def approximate_2_ranges(self):
        must_occur = [0.5, self.m, 1.0-self.m]

        a_norm = np.linalg.norm(self.a)
        b_norm = np.linalg.norm(self.b)

        lower_p = np.max([0.5*(1-a_norm), 0.0])
        upper_p = np.min([0.5*(1+a_norm), 1.0])
        lower_q = np.max([0.5*(self.m-b_norm), 0.0])
        upper_q = np.min([0.5*(self.m+b_norm), 1.0])

        p_candidates = self.generate_p_candidates(lower_p, upper_p, must_occur)

        pair_candidates = [(p, q) for p in p_candidates for q in self.generate_q_candidates(p, lower_q, upper_q)]
        ps, qs = zip(*pair_candidates)
        gs = self.get_weighted_ginni(ps, qs)
        pair_candidates = zip(gs, ps, qs)

        pair_candidates = sorted(pair_candidates, key=itemgetter(0))
        a_norm_sqr = np.square(a_norm)
        b_norm_sqr = np.square(b_norm)
        a_dot_b = np.dot(self.a, self.b)
        denominator = np.subtract(np.multiply(a_norm_sqr, b_norm_sqr), np.square(a_dot_b))

        def get_components(p, q):
            c = np.subtract(np.multiply(2.0, p), 1.0)
            d = np.subtract(np.multiply(2.0, q), self.m)
            alpha = np.divide(np.subtract(np.multiply(d, a_dot_b), np.multiply(c, b_norm_sqr)), denominator)
            beta = np.divide(np.subtract(np.multiply(c, a_dot_b), np.multiply(d, a_norm_sqr)), denominator)
            return alpha, beta

        for g, p, q in pair_candidates:
            alpha, beta = get_components(p, q)
            w_norm = (alpha ** 2) * a_norm_sqr + 2.0 * alpha * beta * a_dot_b + (beta ** 2) * b_norm_sqr
            if np.less_equal(w_norm, 1.0):
                self.w = np.add(np.multiply(alpha, self.a), np.multiply(beta, self.b))
                self.g = g
                return
        raise Exception('Invalid situation')

    def approximate_1_range(self):
        a_norm = np.linalg.norm(self.a)
        a_norm_sqr = np.square(a_norm)

        a_dot_b = np.dot(self.a, self.b)
        b_hat = np.subtract(self.b, np.multiply(a_dot_b / a_norm_sqr, self.a))
        b_hat_norm = np.linalg.norm(b_hat)
        w_term1 = np.divide(self.a, np.square(a_norm))
        w_term2 = np.divide(b_hat, np.multiply(a_norm, b_hat_norm))

        must_occur = [0.5, self.m, 1.0-self.m]
        lower_p = np.max([0.5*(1-a_norm), 0.0])
        upper_p = np.min([0.5*(1+a_norm), 1.0])

        p_candidates = self.generate_p_candidates(lower_p, upper_p, must_occur)

        p = np.array(p_candidates)
        c = 2.0*p - 1.0
        root = np.sqrt(np.max(np.concatenate([np.zeros([len(c),1]), (a_norm_sqr - np.square(c)).reshape(-1,1)],axis=1),axis=1))
        # root = np.sqrt(np.max([0, a_norm_sqr - np.square(c)]))
        current_w_term1 = np.dot(c.reshape((-1,1)), w_term1.reshape((1,-1)))
        current_w_term2 = np.dot(root.reshape((-1,1)), w_term2.reshape((1,-1)))
        # current_w_term1 = np.outer(c.reshape((-1,1)), w_term1.reshape((-1,1)))
        # current_w_term2 = np.outer(root.reshape((-1,1)), w_term2.reshape((-1,1)))
        w1 = np.add(current_w_term1, current_w_term2)
        w2 = np.subtract(current_w_term1, current_w_term2)
        w = np.concatenate([w1,w2])
        p = np.concatenate([p,p])
        q = np.divide(np.add(np.dot(w, self.b), self.m), 2.0)
        g = self.get_weighted_ginni(p, q)
        best_index = np.argmin(g)
        self.w = w[best_index, :].reshape(-1)
        self.g = g[best_index]

        # pair_candidates = []
        # print 'slow mode'
        # for p in p_candidates:
        #     c = 2.0*p - 1.0
        #     root = np.sqrt(np.max([0, a_norm_sqr - np.square(c)]))
        #     current_w_term1 = np.multiply(c, w_term1)
        #     current_w_term2 = np.multiply(root, w_term2)
        #     pair_candidates += [(p,np.add(current_w_term1, current_w_term2))]
        #     pair_candidates += [(p,np.subtract(current_w_term1, current_w_term2))]
        #
        # def extend_with_q_and_ginni(p, w):
        #     q = np.divide(np.add(np.dot(self.b, w), self.m), 2.0)
        #     g = self.get_weighted_ginni(p, q)
        #     return g, p, q, w.ravel()
        # pair_candidates = [extend_with_q_and_ginni(p, w) for p, w in pair_candidates]
        #
        # best_index = np.argmin([x[0] for x in pair_candidates])
        # best = pair_candidates[best_index]
        # self.w = best[3]
        # self.g = best[0]

    def approximate_solver(self, data, labels, data_dist):
        self.set_globals(data, labels, data_dist)
        self.set_dependency_coefficient()
        if np.equal(0.0, self.lambda_):
            if self.algorithm_mode is None:
                # method is inferred according to relation between epsilon and dimensionality
                e = np.power(self.epsilon, -1)
                if 100*e*np.log(e) < data.shape[1]:
                    self.approximate_2_ranges()
                else:
                    self.approximate_1_range()
            elif self.algorithm_mode == '1':
                self.approximate_1_range()
            else:
                self.approximate_2_ranges()
        else:
            self.approximate_dependent()
        # self.set_w_to_positive_side()

    # def set_w_to_positive_side(self):
    #     probabilities_positive = self.get_probabilities(self.data)
    #     probabilities_negative = np.subtract(1.0, probabilities_positive)
    #
    #     positive_labels = np.dot(probabilities_positive, self.labels)
    #     negative_labels = np.dot(probabilities_negative, self.labels)
    #
    #     if np.greater_equal(positive_labels, negative_labels):
    #         return
    #
    #     self.w = np.multiply(-1.0, self.w)
    #     probabilities_positive = self.get_probabilities(self.data)
    #     probabilities_negative = np.subtract(1.0, probabilities_positive)
    #
    #     positive_labels = np.dot(probabilities_positive, self.labels)
    #     negative_labels = np.dot(probabilities_negative, self.labels)
    #     if np.less(positive_labels, negative_labels):
    #         raise Exception('Invalid situation')

    def get_probabilities(self, data):
        res = np.multiply(np.add(np.dot(data, self.w), 1.0), 0.5)
        # make sure probabilities are in range
        res[np.less(res, 0.0)] = 0.0
        res[np.greater(res, 1.0)] = 1.0
        return res

    def predict_deterministic(self, data):
        prediction_probabilities = self.get_probabilities(data)
        return np.greater_equal(prediction_probabilities, 0.5)

    def predict_stochastic(self, data):
        prediction_probabilities = self.get_probabilities(data)
        random_coins = np.random.uniform(0.0, 1.0, data.shape[0])
        return np.greater_equal(prediction_probabilities, random_coins)


if __name__ == "__main__":

    # def validate_utility_functions():
    #     p_x = np.array([0.3, 0.5, 0.2])
    #     print p_x
    #
    #     X = np.arange(15).reshape(3, 5)
    #     print X
    #
    #     y = np.ones(3, dtype=np.int32)
    #     y[1] = 0
    #     print y
    #
    #     a, b, N, m = set_globals(X, y, p_x)
    #     print a, b, N, m
    #
    #     print set_dependency_coefficient(a, b)
    #     print set_dependency_coefficient(b, b)
    #     print set_dependency_coefficient(b * 2, b)
    #
    #     print generate_p_candidates(0.01, 0.99, 0.3)
    #     print generate_p_candidates(0.01, 0.99, 0.3, [0.5])
    #     print generate_p_candidates(0.0, 1.0, 0.3, [0.5])
    #
    #     print generate_q_candidates(0.1, 0.7, 0.01)
    #
    #     print get_ginni([0.0, 0.3, 0.5, 0.7, 1.0])
    #
    #     print get_weighted_ginni(0.7, [0.5, 0.5], [0.3, 0.5])
    #
    # validate_utility_functions()

    def validate_approximation():
        from sklearn.datasets import make_blobs
        import matplotlib.pyplot as plt

        # create fake data
        # X, y = make_blobs(n_features=2, centers=2)
        X, y = make_blobs(n_features=2, centers=3)
        y[y>0]=1

        p_x = np.ones(y.shape) / X.shape[0]

        classifier = BaseClassifier(0.1, algorithm_mode="2", normalizer="norm")
        classifier.approximate_solver(X, y, p_x)

        print classifier.w
        print classifier.g

        y_hat = classifier.predict_deterministic(X)
        print classifier.predict_stochastic(X)
        print classifier.get_probabilities(X)

        def save_to_file(X, y, y_hat):
            import os
            import time
            time.ctime()
            time_to_use = time.strftime('%X %x').replace(':', '-').replace('/', '-').replace(' ', '-')
            dir_path = os.path.join('c:\\', 'temp', 'approximation-solver', time_to_use)
            os.makedirs(dir_path)
            np.save(os.path.join(dir_path, 'X'), X)
            np.save(os.path.join(dir_path, 'y'), y)
            np.save(os.path.join(dir_path, 'y_hat'), y_hat)

        save_to_file(X, y, y_hat)

        plt.figure()
        plt.subplot(121)
        plt.title("True", fontsize='small')
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
        plt.subplot(122)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        h = .02  # step size in the mesh
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = classifier.predict_deterministic(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        plt.title("Predicted", fontsize='small')
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_hat)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.show()
    import cProfile

    print cProfile.run('validate_approximation()')

    # validate_approximation()
    print 'done'
