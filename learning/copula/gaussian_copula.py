import numpy as np
import scipy.stats as stats
from sklearn.covariance import EmpiricalCovariance

class GaussianCopula:
    def __init__(self):
        self.cov = None
        self.dim = None

    def fit(self, X):
        """
        X: shape (n_samples, n_assets) — assumed to be returns
        """
        self.dim = X.shape[1]

        # Step 1: Convert to uniform marginals using ranks
        ranks = stats.rankdata(X, axis=0) / (X.shape[0] + 1)
        norm_marginals = stats.norm.ppf(ranks)

        # Step 2: Estimate correlation of normal scores
        self.cov = EmpiricalCovariance().fit(norm_marginals).covariance_

    def sample(self, n_samples):
        """
        Sample from the Gaussian copula
        Returns samples with uniform marginals
        """
        z = np.random.multivariate_normal(mean=np.zeros(self.dim), cov=self.cov, size=n_samples)
        u = stats.norm.cdf(z)
        return u

    def transform(self, X):
        """
        Convert original data X into Gaussian copula space
        (i.e., normal scores)
        """
        ranks = stats.rankdata(X, axis=0) / (X.shape[0] + 1)
        return stats.norm.ppf(ranks)

    def inverse_transform(self, U):
        """
        Map uniform marginals back to standard normal marginals
        """
        return stats.norm.ppf(U)
