import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

"""
DATA GENERATING FUNCTIONS

The following functions simulate data in various ways for mixed regression
"""
def create_trip_data(n_trips, error_size = 0):
    distances = np.random.normal(40, 10, size = n_trips).reshape(n_trips, -1)
    trip_type = np.random.randint(0, 1+1, size = n_trips).reshape(n_trips, -1)
    beta_0 = 52 * trip_type
    beta_1 = 2.29 * (1 - trip_type)
    cost = beta_0 + distances * beta_1 + np.random.normal(0, error_size, n_trips).reshape(-1,1)
    cost = cost.reshape(-1)
    return(distances, trip_type, cost)

def create_trip_data_3d(n_trips, error_size = 0):
    distances = np.random.normal(40, 10, size = n_trips).reshape(n_trips, -1)
    tip_amount = np.random.normal(10, 2, size = n_trips).reshape(n_trips, -1)
    trip_type = np.random.randint(0, 1+1, size = n_trips).reshape(n_trips, -1)
    beta_0 = 52 * trip_type 
    beta_1 = 2.29 * (1 - trip_type)
    cost = beta_0 + distances * beta_1 + np.random.normal(0, error_size, n_trips).reshape(-1,1) + tip_amount
    cost = cost.reshape(-1)
    features = np.concatenate((distances, tip_amount), axis = 1)
    return(features, trip_type, cost)

def create_nd_data(data_size, error_size = 0, n_dim=5, orthonorm = True, mean=None, cov=None):
    beta_0 = np.random.randn(n_dim+1)
    beta_1 = np.random.randn(n_dim+1)
    
    if orthonorm:
        beta_0 = beta_0 / np.linalg.norm(beta_0)
        beta_1 = beta_1 - np.dot(beta_0, beta_1) * beta_1
        beta_1 = beta_1 / np.linalg.norm(beta_1)
    
    if mean is None:
        mean = np.random.randint(50, size=n_dim-1)
        cov = np.diag(np.random.randint(4, size=n_dim-1) + np.ones(n_dim-1) * 0.001)
    
    data = np.concatenate((np.ones(data_size).reshape(-1,1), \
                           np.random.multivariate_normal(mean, cov, data_size)), axis=1)
    
    trip_type = np.random.randint(0, 1+1, size = data_size).reshape(data_size, -1)

    cost = data.dot(beta_0) * trip_type.reshape(-1) + data.dot(beta_1) * (1-trip_type.reshape(-1)) + np.random.normal(0, error_size, data_size)
    cost = cost.reshape(-1)
    features = data[:,1:]
    
    return(features, trip_type, cost)


"""
FUNCTIONS TO RUN THE EM ALGORITHM FOR MIXED REGRESSION

This next set of functions fit linear mixed regression on a given data set
fit_mixed_regression() calls either initialize_classic() or initialize_optimal
Both initialize_*() functions call run_grid_search, unless search_grid = False and algo_type = 'classic'
fit_mixed_regression() then calls run_EM_iteration repeatedly (n_iter times)
"""
def run_grid_search(v1, v2, X, y, delta = 0.3):
    t_vals = np.arange(int(np.ceil(2 * np.pi / delta)) + 1).reshape(1, -1, 1)
    G = v1 * np.cos(delta * t_vals) + v2 * np.sin(delta * t_vals)
    G = G.reshape(-1, X.shape[0])
    min_loss = np.inf
    for u1 in G:
        l1 = y - u1.dot(X)
        for u2 in G:
            l2 = y - u2.dot(X)
            optim_mask = np.abs(l1) < np.abs(l2)
            l1_masked = l1[optim_mask]
            l2_masked = l2[~optim_mask]
            loss = np.sum(l1_masked**2) + np.sum(l2_masked**2)
            if loss < min_loss:
                min_loss = loss
                betas = {0:u1, 1:u2}
    return(betas)  

def initialize_classic(X, y, mean = 0.0, sd = 1.0, search_grid = False, delta = 0.3):
    n_features = X.shape[0]
    betas = {}
    for i in range(2):
        betas_cur = np.random.normal(mean, sd, n_features)
        betas_cur = betas_cur / np.linalg.norm(betas_cur)
        betas[i] = betas_cur
    if search_grid:
        #single iteration gram-schmidt to get orthogonal vectors for grid search
        v1 = betas[0]
        v2 = betas[1] - np.dot(v1, betas[1]) * v1
        betas = run_grid_search(v1, v2, X, y, delta)
    return(betas)

def initialize_optimal(X, y, delta=0.3):
    n_features = X.shape[0]
    M = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[1]):
        M += y[i]**2 * np.outer(X[:,i], X[:,i])
    M /= X.shape[1]
    
    eigenvalues, eigenvectors = np.linalg.eig(M)
    sorted_eigs = sorted(zip(eigenvalues, eigenvectors), key=lambda x: x[0], reverse=True)
    v1 = sorted_eigs[0][1]
    v2 = sorted_eigs[1][1]
    
    betas = run_grid_search(v1, v2, X, y, delta)
    return(betas)  

def run_EM_iteration(X, y, betas, true_betas):
    n_features = X.shape[0]
    pred0 = X.transpose().dot(betas[0])
    pred1 = X.transpose().dot(betas[1])
    err0 = np.abs(pred0 - y)
    err1 = np.abs(pred1 - y)
    lower_error = (err0 < err1)
    j0_ind = np.where(lower_error)
    j1_ind = np.where(~lower_error)
    X_0 = X[:, j0_ind].reshape(n_features,-1)
    X_1 = X[:,j1_ind].reshape(n_features,-1)
    
    ##Solve least squares 
    try:
        beta_0 = np.linalg.inv(X_0.dot(X_0.transpose())).dot(X_0).dot(y[j0_ind])
    except:
        beta_0 = betas[0]
    
    try:
        beta_1 = np.linalg.inv(X_1.dot(X_1.transpose())).dot(X_1).dot(y[j1_ind])
    except:
        beta_1 = betas[1]
    
    betas = {0: beta_0,
             1: beta_1}
    
    iter_err_0 = np.sum((beta_0.dot(X_0) - y[j0_ind])**2)
    iter_err_1 = np.sum((beta_1.dot(X_1) - y[j1_ind])**2)
    iter_err = np.log(np.sqrt((iter_err_0 + iter_err_1) / X.shape[1])) # log RMSE
    
    iter_beta_err_v1 = max(np.linalg.norm(betas[0] - true_betas[0]), np.linalg.norm(betas[1] - true_betas[1]))
    iter_beta_err_v2 = max(np.linalg.norm(betas[1] - true_betas[0]), np.linalg.norm(betas[0] - true_betas[1]))
    iter_beta_err = np.log(max(1e-15,min(iter_beta_err_v1, iter_beta_err_v2)))

    
    return(betas, iter_err, iter_beta_err)

def fit_mixed_regression(x, y, z, 
                         do_normalize = True, n_iter = 10, algo_type = 'classic', 
                         search_grid = True, return_true_betas = False, delta = 0.3):
    n_samples = x.shape[0]
    ones = np.ones(n_samples).reshape(-1, 1)
    if do_normalize:
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
    X = np.concatenate((ones, x), axis = 1).transpose()
    
    #get true betas
    X0 = X[:, (z==0).reshape(-1)]
    X1 = X[:, (z==1).reshape(-1)]
    y0 = y[(z==0).reshape(-1)]
    y1 = y[(z==1).reshape(-1)]
    try:
        true_beta_0 = np.linalg.inv(X0.dot(X0.transpose())).dot(X0).dot(y0)
        true_beta_1 = np.linalg.inv(X1.dot(X1.transpose())).dot(X1).dot(y1)
    except:
        print('X0 or X1 is not invertible')
        return (None, None, None)
    
    true_betas = {0: true_beta_0, 1: true_beta_1}
    #initialize parameters
    if algo_type == 'classic':
        betas = initialize_classic(X, y, search_grid = search_grid)
    else:
        betas = initialize_optimal(X, y)
    iter_errs = []
    iter_beta_errs = []

    for j in range(n_iter):
        betas, iter_err, iter_beta_err = run_EM_iteration(X, y, betas, true_betas)
        iter_errs.append(iter_err)
        iter_beta_errs.append(iter_beta_err)
    if return_true_betas:
        return(betas, iter_errs, iter_beta_errs, true_betas)
    else:        
        return(betas, iter_errs, iter_beta_errs)

"""
SIMULATING MANY FITS

This function takes a data generating function and fitting options, and repeatedly generates
a dataset and runs mixed regression in accordance with those options.
"""
    
def simulate_fits(n_trips, error_size, creation_func, n_sims, do_normalize,
                  n_dim = None, mean = None, cov = None, show=True):

    all_iter_errs_classic = []
    all_iter_errs_classic_grid = []
    all_iter_errs_opt = []
    beta_errs_classic = []
    beta_errs_classic_grid = []
    beta_errs_opt = []

    convergence_record_opt = []
    for run_num in range(n_sims):
        x, z, y = creation_func(n_trips, error_size, n_dim = n_dim, mean = mean, cov = cov)
        
        np.random.seed(run_num)
        
        betas_classic, iter_errs_classic, iter_beta_errs_classic = \
            fit_mixed_regression(x, y, z, do_normalize = do_normalize, algo_type = 'classic', search_grid = False)
        
        if betas_classic == None:
            continue
        
        betas_classic_grid, iter_errs_classic_grid, iter_beta_errs_classic_grid = \
            fit_mixed_regression(x, y, z, do_normalize = do_normalize, algo_type = 'classic', search_grid = True)
        
        betas_opt, iter_errs_opt, iter_beta_errs_opt = \
            fit_mixed_regression(x, y, z, do_normalize = do_normalize, algo_type = 'optimal')
        
        all_iter_errs_classic.append(iter_errs_classic)
        all_iter_errs_classic_grid.append(iter_errs_classic_grid)
        all_iter_errs_opt.append(iter_errs_opt)
        beta_errs_classic.append(iter_beta_errs_classic)
        beta_errs_classic_grid.append(iter_beta_errs_classic_grid)
        beta_errs_opt.append(iter_beta_errs_opt)

    beta_errs_classic = np.array(beta_errs_classic)
    beta_errs_classic_grid = np.array(beta_errs_classic_grid)
    beta_errs_opt = np.array(beta_errs_opt)

    avg_beta_errs_classic = np.mean(beta_errs_classic, axis = 0)
    avg_beta_errs_classic_grid = np.mean(beta_errs_classic_grid, axis = 0)
    avg_beta_errs_opt = np.mean(beta_errs_opt, axis = 0)

    
    plt.plot(avg_beta_errs_classic, label = 'Classic')
    plt.plot(avg_beta_errs_classic_grid, label = 'Classic + Grid Search')
    plt.plot(avg_beta_errs_opt, label = 'Optimal')
    plt.title('Average log parameter error')
    plt.legend()
    if show:
        plt.show()

