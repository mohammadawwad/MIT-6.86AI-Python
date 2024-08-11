import numpy as np
import common
import kmeans
import naive_em

# Load the dataset
X = np.loadtxt("toy_data.txt")

# Number of clusters to try
K_values = [1, 2, 3, 4]

# Seeds to try
seeds = [0, 1, 2, 3, 4]

# Dictionaries to store the best results for K-means and EM
best_kmeans_mixtures = {}
best_kmeans_costs = {}
best_kmeans_posts = {}

best_em_mixtures = {}
best_em_log_likelihoods = {}
best_em_posts = {}
bic_scores = {}

for K in K_values:
    # Initialize the best values for K-means and EM
    best_kmeans_cost = float('inf')
    best_kmeans_mixture = None
    best_kmeans_post = None
    
    best_log_likelihood = float('-inf')
    best_em_mixture = None
    best_em_post = None
    
    for seed in seeds:
        # Initialize the mixture model
        mixture, post = common.init(X, K, seed)
        
        # Run K-means
        kmeans_mixture, kmeans_post, kmeans_cost = kmeans.run(X, mixture, post)
        
        # Update the best K-means solution if this one is better
        if kmeans_cost < best_kmeans_cost:
            best_kmeans_cost = kmeans_cost
            best_kmeans_mixture = kmeans_mixture
            best_kmeans_post = kmeans_post
        
        # Run EM
        em_mixture, em_post, em_log_likelihood = naive_em.run(X, mixture, post)
        
        # Update the best EM solution if this one is better
        if em_log_likelihood > best_log_likelihood:
            best_log_likelihood = em_log_likelihood
            best_em_mixture = em_mixture
            best_em_post = em_post
    
    # Store the best K-means results for this K
    best_kmeans_mixtures[K] = best_kmeans_mixture
    best_kmeans_costs[K] = best_kmeans_cost
    best_kmeans_posts[K] = best_kmeans_post

    # Store the best EM results for this K
    best_em_mixtures[K] = best_em_mixture
    best_em_log_likelihoods[K] = best_log_likelihood
    best_em_posts[K] = best_em_post
    
    # Calculate the BIC score for the best EM model
    bic_scores[K] = common.bic(X, best_em_mixture, best_log_likelihood)
    
    # Plot the best K-means solution
    kmeans_title = f"K-means: K = {K}, Cost = {best_kmeans_cost:.2f}"
    common.plot(X, best_kmeans_mixture, best_kmeans_post, kmeans_title)
    
    # Plot the best EM solution
    em_title = f"EM: K = {K}, Log-likelihood = {best_log_likelihood:.2f}"
    common.plot(X, best_em_mixture, best_em_post, em_title)

# Find the best K based on the best (lowest) BIC score
best_K = min(bic_scores, key=bic_scores.get)
best_BIC = bic_scores[best_K]

# Plot the final best EM solution based on BIC
final_em_mixture = best_em_mixtures[best_K]
final_em_post = best_em_posts[best_K]
final_em_title = f"Best EM by BIC: K = {best_K}, BIC = {best_BIC:.2f}"
common.plot(X, final_em_mixture, final_em_post, final_em_title)

# Report the best K and the corresponding BIC score
print(f"Best K: {best_K}")
print(f"Best BIC score: {best_BIC:.2f}")
