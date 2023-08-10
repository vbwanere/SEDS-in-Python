import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixture:

    def __init__(self, k, max_iter=5):
        self.k = k # number of distributions
        self.max_iter = int(max_iter)

    def initialize(self, X):
        """
        The function initializes the parameters for a Gaussian Mixture Model (GMM) using the input data.
        
        :param X: X is the input data, which is a numpy array with shape (m, n).
        """
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
        """
        The function "e_step" calculates the responsibility matrix for a given input X.
        
        :param X: The parameter X is a matrix that represents the input data.
        """
        self.responsibility_k = self.responsibility_matrix(X)

    def m_step(self, X):
        """
        The function calculates the updated mixture weights, means, and covariances for each component
        in a Gaussian Mixture Model based on the responsibilities of each data point.
        
        :param X: X is a numpy array representing the input data. It has shape (m, n), where m is the
        number of samples and n is the number of features.
        """
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
        """
        The function performs the expectation-maximization algorithm to fit a model to the given data.
        
        :param X: The input data for the model. It could be a matrix or an array-like object containing
        the features or variables used for training the model.
        """
        self.initialize(X)
        for iteration in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)

    def responsibility_matrix(self, X): # shape = (m, k)
        """
        The function calculates the responsibility matrix for a given data matrix using a Gaussian
        mixture model.
        
        :param X: X is a numpy array with shape (m, n), where m is the number of features and n is
        the number of data points.
        :return: the responsibility matrix, which is a matrix of shape (m, k) where m is the number of
        data points and k is the number of clusters.
        """
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
        """
        The function predicts the class labels for a given set of input data.
        
        :param X: The parameter X represents the input data for which you want to make predictions. It
        is a matrix or array-like object with shape (n_samples, n_features), where n_samples is the
        number of samples or instances in the dataset, and n_features is the number of features or
        attributes for each sample
        :return: the index of the maximum probability value in the responsibility_k matrix for each data point.
        """
        responsibility_k = self.responsibility_matrix(X)
        return np.argmax(responsibility_k, axis=1)