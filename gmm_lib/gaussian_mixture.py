import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixture:

    def __init__(self, k, max_iter=5):
        self.k = k # number of distributions
        self.max_iter = int(max_iter)

    def initialize(self, X):
        self.shape = X.shape # shape of the input (m, n)
        self.m, self.n = self.shape

        # Initializing the mean and covariance:
        # generate row index using m.
        random_row = np.random.randint(low=0, high=self.m, size=self.k)
        # use the above random rows to initialize the mean and covariance.
        # mean: shape = (m, 1)
        self.means = [X[row_index, :] for row_index in random_row]
        # covariance: shape = (m, m)
        self.covariances = [np.cov(X, rowvar=False) for _ in range(self.k)]

        # mixing coefficient: shape = (1, k)
        self.mixture_weights = np.full((self.k, 1), 1/self.k).T
        # responsibility matrix: shape = (m, k)
        self.responsibility_k = np.full((self.m, self.k),\
                                             fill_value=1/self.k)


    def e_step(self, X):
        self.responsibility_k = self.responsibility_matrix(X)

    def m_step(self, X):
        self.mixture_weights = self.responsibility_k.mean(axis=0)
        for i in range(self.k):
            responsibility_k = self.responsibility_k[:, [i]] #(m, 1)
            sum_responsibility_k = responsibility_k.sum()

            self.means[i] = ((responsibility_k * X).sum(axis=0) /\
                          sum_responsibility_k) # mean

            self.covariances[i] = np.cov(X, rowvar=False,\
                                   aweights=(responsibility_k/\
                                             sum_responsibility_k).flatten(),\
                                   bias=True) # covariance

    def fit(self, X):
        self.initialize(X)
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)

    def responsibility_matrix(self, X): # shape = (m, k)
        likelihood = np.zeros((self.m, self.k))
        for i in range(self.k):
            likelihood[:, i] = multivariate_normal.pdf(X, mean=self.means[i],\
                                               cov=self.covariances[i])
            numerator = self.mixture_weights * likelihood # column-wise product
            denominator = numerator.sum(axis=1) # row-wise sum
            denominator = np.expand_dims(denominator, axis=1) # reshape to (m, 1)
            responsibility_matrix = numerator / denominator
        return responsibility_matrix

    def predict(self, X):
        responsibility_k = self.responsibility_matrix(X)
        return np.argmax(responsibility_k, axis=1)