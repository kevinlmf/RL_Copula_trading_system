import numpy as np
import scipy.stats as stats
from sklearn.covariance import EmpiricalCovariance

class TCopula:
    def __init__(self, df=4):
        self.df = df
        self.cov = None
        self.dim = None

    def fit(self, X):
        self.dim = X.shape[1]
        ranks = stats.rankdata(X, axis=0) / (X.shape[0] + 1)
        t_marginals = stats.t.ppf(ranks, df=self.df)
        self.cov = EmpiricalCovariance().fit(t_marginals).covariance_

    def sample(self, n_samples):
        z = np.random.multivariate_normal(mean=np.zeros(self.dim), cov=self.cov, size=n_samples)
        chi2 = np.random.chisquare(df=self.df, size=(n_samples, 1))
        t_samples = z / np.sqrt(chi2 / self.df)
        u = stats.t.cdf(t_samples, df=self.df)
        return u

    def transform(self, X):
        ranks = stats.rankdata(X, axis=0) / (X.shape[0] + 1)
        return stats.t.ppf(ranks, df=self.df)

    def inverse_transform(self, U):
        return stats.t.ppf(U, df=self.df)
