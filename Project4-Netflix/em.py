"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    
    # Initialize the responsibilities (soft counts)
    post = np.zeros((n, K))
    
    # Initialize the log-likelihood
    log_likelihood = 0.0
    
    for i in range(n):
        # Find the indices of the observed (non-missing) entries for the i-th data point
        observed = X[i] != 0
        d_obs = np.sum(observed)  # Number of observed dimensions

        for j in range(K):
            # Compute the Gaussian probability density for X[i] under mixture j, considering only observed dimensions
            norm_factor = (2 * np.pi * mixture.var[j])**(d_obs / 2)
            diff = X[i, observed] - mixture.mu[j, observed]
            exponent = -0.5 * np.sum(diff**2) / mixture.var[j]
            prob = np.exp(exponent) / norm_factor
            
            # Weight by the mixing proportion
            post[i, j] = mixture.p[j] * prob
        
        # Normalize the responsibilities
        total_prob = np.sum(post[i])
        post[i] /= total_prob
        
        # Add to the log-likelihood
        log_likelihood += np.log(total_prob)
    
    return post, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    # Effective number of points assigned to each cluster
    n_k = np.sum(post, axis=0)

    # Update the mixing proportions
    p = n_k / n

    # Update the means
    mu = np.zeros((K, d))
    for j in range(K):
        for i in range(n):
            observed = X[i] != 0
            mu[j, observed] += post[i, j] * X[i, observed]
        mu[j] /= np.sum(post[:, j][:, None] * (X != 0), axis=0)

    # Update the variances
    var = np.zeros(K)
    for j in range(K):
        total_variance = 0
        for i in range(n):
            observed = X[i] != 0
            diff = X[i, observed] - mu[j, observed]
            total_variance += post[i, j] * np.sum(diff**2)
        var[j] = max(total_variance / np.sum(post[:, j] * np.sum(X != 0, axis=1)), min_variance)

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_log_likelihood = -np.inf  # Start with negative infinity
    tolerance = 1e-6  # Slightly larger tolerance to encourage convergence

    while True:
        # E-step: Calculate the soft counts given the current mixture model
        post, log_likelihood = estep(X, mixture)
        
        # Check convergence
        if prev_log_likelihood != -np.inf and np.abs(log_likelihood - prev_log_likelihood) < tolerance:
            break
        
        prev_log_likelihood = log_likelihood
        
        # M-step: Update the mixture model given the soft counts
        mixture = mstep(X, post, mixture)
        
    return mixture, post, log_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
