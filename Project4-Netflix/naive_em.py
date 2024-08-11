from typing import Tuple
import numpy as np
from common import GaussianMixture

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component"""
    n, d = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))
    log_likelihood = 0.0
    
    for i in range(n):
        for j in range(K):
            norm_factor = (2 * np.pi * mixture.var[j])**(d / 2)
            diff = X[i] - mixture.mu[j]
            exponent = -0.5 * np.sum(diff**2) / mixture.var[j]
            prob = np.exp(exponent) / norm_factor
            post[i, j] = mixture.p[j] * prob
        
        total_prob = np.sum(post[i])
        if total_prob > 0:
            post[i] /= total_prob
        log_likelihood += np.log(total_prob) if total_prob > 0 else 0.0
    
    return post, log_likelihood

def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood"""
    n, d = X.shape
    K = post.shape[1]
    n_k = np.sum(post, axis=0)
    p = n_k / n
    mu = np.dot(post.T, X) / n_k[:, np.newaxis]
    var = np.zeros(K)
    
    for j in range(K):
        diff = X - mu[j]
        var[j] = np.sum(post[:, j] * np.sum(diff**2, axis=1)) / (d * n_k[j])
    
    return GaussianMixture(mu=mu, var=var, p=p)

def run(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model"""
    prev_log_likelihood = None
    max_iter = 1000

    for _ in range(max_iter):
        post, log_likelihood = estep(X, mixture)
        mixture = mstep(X, post)
        
        if prev_log_likelihood is not None:
            improvement = log_likelihood - prev_log_likelihood
            if improvement <= 1e-6 * np.abs(log_likelihood):
                break
        
        prev_log_likelihood = log_likelihood
    
    return mixture, post, log_likelihood
