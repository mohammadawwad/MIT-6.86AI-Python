import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    # Compute the dot product between X and Y
    dot_product = np.dot(X, Y.T)
    
    # Add the coefficient c to the dot product
    kernel_matrix = (dot_product + c) ** p
    
    return kernel_matrix


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    n, d = X.shape
    m, _ = Y.shape
    
    # Expand the dimensions of X and Y for broadcasting
    X = X[:, np.newaxis, :]
    Y = Y[np.newaxis, :, :]
    
    # Compute the squared Euclidean distance between each pair of points
    squared_dist = np.sum((X - Y) ** 2, axis=2)
    
    # Compute the RBF kernel
    kernel_matrix = np.exp(-gamma * squared_dist)
    
    return kernel_matrix
