# import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
from tqdm import tqdm

#############################################################
# Utility methods
#############################################################
def res(x, y, beta):
    return abs(np.dot(x, beta) - y)


def find_core(X, Y, frac = 0.1, stop_criterion = 1e-3, max_iter = 10):
    n = len(Y)
    n_core = int(frac * n)

    beta     = np.linalg.solve(X.T @ X, X.T @ Y)
    resid    = np.abs(X @ beta - Y)
    core_ind = np.argsort(resid)[:int(n / 10)]

    X_core = X[core_ind]
    Y_core = Y[core_ind]

    for i in range(max_iter):
        new_beta = np.linalg.solve(X_core.T @ X_core, X_core.T @ Y_core)
        if np.linalg.norm(new_beta - beta) < stop_criterion:
            print(f'Found stable core ({i} iterations)')
            break

        else:
            res = np.abs(X @ new_beta - Y)
            new_core_ind = np.argsort(res)[:int(n / 10)]
            X_core = X[core_ind]
            Y_core = Y[core_ind]
            beta = new_beta

    return X_core.copy(), Y_core.copy()


def neighbor_core(X, Y, k):
    n, d = X.shape
    reg = 1e-10
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    _, indices = nbrs.kneighbors(X)
    assert len(indices) == n

    best_MSE = np.inf
    core_ind = []
    for i in range(n):
        nbhd = indices[i]
        X_nbhd = X[nbhd]
        Y_nbhd = Y[nbhd]
        beta   = np.linalg.solve(X_nbhd.T @ X_nbhd + reg * np.eye(d), X_nbhd.T @ Y_nbhd)
        MSE    = np.mean((X_nbhd @ beta - Y_nbhd) ** 2)
        if MSE < best_MSE:
            core_ind = nbhd
            best_MSE = MSE

    return X[core_ind].copy(), Y[core_ind].copy()


def different_core(X, Y, k):
    n, d = X.shape
    reg = 1e-10
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    _, indices = nbrs.kneighbors(X)
    assert len(indices) == n

    base_beta = np.linalg.solve(X.T @ X + reg * np.eye(d), X.T @ Y)

    best_diff = 0
    core_ind = []
    for i in range(n):
        nbhd = indices[i]
        X_nbhd = X[nbhd]
        Y_nbhd = Y[nbhd]
        beta   = np.linalg.solve(X_nbhd.T @ X_nbhd + reg * np.eye(d), X_nbhd.T @ Y_nbhd)
        diff   = np.linalg.norm(beta - base_beta)
        if diff > best_diff:
            core_ind = nbhd
            best_diff = diff

    return X[core_ind].copy(), Y[core_ind].copy()


def core_fit(X_core, Y_core):
    n_core, d = X_core.shape
    reg = 1e-10
    XTX = X_core.T @ X_core + reg * np.eye(d)
    beta_hat = np.linalg.solve(XTX, X_core.T @ Y_core)
    min_eig = min(np.linalg.eigvals(XTX)) / n_core
    s_hat = np.sqrt(np.sum((X_core @ beta_hat - Y_core) ** 2) / (len(Y_core) - len(X_core[0]) - 1))
    return beta_hat, min_eig, s_hat


def proj(X, R):
    return np.clip(X, R[:, 0], R[:, 1])


def in_box(X, R):
    # Returns a list of length len(X) whose i-th entry is True if X[i] is in R.
    return np.equal(proj(X, R), X).all(axis = len(X.shape) - 1)


def test_MAE(X_test, Y_test, beta, R):
    incl_points = in_box(X_test[:, :-1], R)
    if np.sum(incl_points) == 0:
        return float('nan')
    abs_resid = np.abs(X_test @ beta - Y_test) * incl_points
    return np.sum(abs_resid) / np.sum(incl_points)


def box_intersection(B1, B2):
    assert len(B1) == len(B2)

    vol = 1.
    for i in range(len(B1)):
        lower = max(B1[i, 0], B2[i, 0])
        upper = min(B1[i, 1], B2[i, 1])
        if lower >= upper:
            return 0
        else:
            vol *= upper - lower

    return vol


#############################################################
# Method 1 (Hard grow): Hard thresholding + growing box
#############################################################
def hard_grow_cutoff(std, min_eig, n_core, n_full, alpha, x):
    d = len(x)
    # return std * (np.linalg.norm(x) * np.sqrt(d * np.log(4 * d / alpha) / n_core) / min_eig + np.sqrt(2 * np.log(4 * n_full / alpha)))
    return alpha * std * np.linalg.norm(x)

def hard_grow_labels(X, Y, alpha, std, min_eig, n_core, beta):
    n = len(Y)
    labels = np.zeros(n)
    for k in range(n):
        x = X[k]
        y = Y[k]
        if res(x, y, beta) > hard_grow_cutoff(std, min_eig, n_core, n, alpha, x):
            labels[k] = 1
    return labels


def directed_infty_norm(x, S):
    best = 0
    for j in range(len(x)):
        if S[j] != set():
            best = max(best, max([x[j] * s for s in S[j]]))
    return best


def hard_grow_region(X, labels, B, center = None, speeds = None, tol = 1e-5, shrinkage=0):
    # B is a d x 2 array containing the maximum allowed box.
    # B must contain the origin and the origin must be contained in the final selected region.
    # Any valid point x must have B[i,0] <= x[i] <= B[i,1].
    # All of the points in X should be within this box.
    X2 = X[labels == 1].copy()
    n, d = X2.shape
    R = B.copy()

    if center is not None:
        X2 -= center
        R  -= center.reshape((d, 1))
    if speeds is not None:
        assert len(speeds) == d
        for j in range(d):
            X2[:, j] /= speeds[j]
            R[:, j]  /= speeds[j]
    
    
    S = [set([-1,1]) for j in range(d)]
    # print(S)

    while X2.any() and S != [set() for _ in range(d)]: # Terminate when there are no labeled points left or all sides are supported
        directed_infty_norms = [directed_infty_norm(x, S) for x in X2]
        i = np.argmin(directed_infty_norms) # i = point which supports the new side
        j = list(np.abs(X2[i]) == directed_infty_norms[i]).index(True) # j = dimension which is being supported
        sign = int(np.sign(X2[i, j]))

        S[j].remove(sign)
        R[j, int((sign + 1)/2)] = X2[i, j]
        # print(X2[i,j])
        # print(B2[j, int((sign + 1)/2)])

        X2 = X2[[k for k in range(len(X2)) if sign * X2[k, j] < directed_infty_norms[i] + shrinkage]] # Allows us to still consider points which are within shrinkage of the side supported by the last selected point

    if speeds is not None:
        for j in range(d):
            R[:, j] *= speeds[j]
    if center is not None:
        R += center.reshape((d, 1))
    
    if np.linalg.norm(R - B) <= tol:
        return B
        
    return R


#############################################################
# Method 2 (Hard opt): Hard thresholding + optimization based
#############################################################
def hard_opt_cutoff(std, min_eig, n_core, n_full, alpha, x):
    d = len(x)
    return std * (np.linalg.norm(x) * np.sqrt(d * np.log(2 * d / alpha) / n_core) / min_eig + np.sqrt(2 * np.log(2 / alpha)))


def hard_opt_labels(X, Y, alpha, std, min_eig, n_core, beta):
    n = len(Y)
    labels = np.zeros(n)
    for k in range(n):
        x = X[k]
        y = Y[k]
        if res(x, y, beta) > hard_opt_cutoff(std, min_eig, n_core, n, alpha, x):
            labels[k] = 1
    return labels


# def vol(R):
#     v = 1.
#     for i in range(len(R)):
#         v *= R[i, 1] - R[i, 0]  # Note: We might run into problems if lb > ub, watch for this.
#     return v

def vol(R):
    total = 0
    for i in range(len(R)):
        total += R[i, 1] - R[i, 0]
    return total

def dist(X, R):
    return torch.linalg.norm(X - torch.clamp(X, R[:, 0], R[:, 1]), dim = len(X.shape) - 1)


def hard_obj(X, R, labels, reg, c1 = 1., c2 = 1.):
    return vol(R) - reg * torch.sum(labels * torch.exp(-c1 * dist(X, R)) - (1 - labels) * torch.exp(-c2 * dist(X, R)))


def hard_opt_region(X, labels, B, args):
    init_R = args[0]
    reg    = args[1]
    iters  = args[2]
    lr     = args[3]
    c1     = args[4]
    c2     = args[5]

    R = torch.tensor(init_R, requires_grad = True)
    for i in range(iters):
        if i % 10 == 0:
            print(R)

        if R.grad is not None:
            R.grad.zero_()
        
        obj = hard_obj(X, R, labels, reg, c1, c2)
        obj.backward()
        with torch.no_grad():
            R += lr * R.grad.data
            R[:, 0] = torch.clamp(R[:, 0], min = B[:, 0])
            R[:, 1] = torch.clamp(R[:, 1], max = B[:, 1])
        
    return R.detach().numpy()


#############################################################
# Method 2' (Hard opt done right): Hard thresholding + optimization based
#############################################################
def incl_grad(R, x, c):
    grad = np.zeros(R.shape)
    base = np.prod([norm.cdf(c * (R[j, 1] - x[j])) - norm.cdf(c * (R[j, 0] - x[j])) for j in range(len(R))])
    for j in range(len(R)):
        grad[j, 0] = -c * norm.pdf(c * (R[j, 0] - x[j])) * base / (norm.cdf(c * (R[j, 1] - x[j])) - norm.cdf(c * (R[j, 0] - x[j]))) 
        grad[j, 1] =  c * norm.pdf(c * (R[j, 1] - x[j])) * base / (norm.cdf(c * (R[j, 1] - x[j])) - norm.cdf(c * (R[j, 0] - x[j])))
    return grad


def size_grad(R):
    grad = np.ones(R.shape)
    grad[:, 0] *= -1
    return grad


def obj_grad(R, X, labels, alpha, reg, c):
    full_incl_grad = np.zeros(R.shape)
    for i in range(len(X)):
        full_incl_grad += (alpha - labels[i]) * incl_grad(R, X[i], c)
    print(full_incl_grad)
    return size_grad(R) + reg * full_incl_grad


def objective(R, X, labels, alpha, reg, c):
    obj = np.sum([R[j, 1] - R[j, 0] for j in range(len(R))])
    # obj = np.prod([R[j, 1] - R[j, 0] for j in range(len(R))])
    for i in range(len(X)):
        obj += reg * (alpha - labels[i]) * np.prod([norm.cdf(c * (R[j, 1] - X[i, j])) - norm.cdf(c * (R[j, 0] - X[i, j])) for j in range(len(R))])
    return obj


def hard_opt_region2(X, labels, B, args):
    init_R = args[0]
    lr     = args[1]
    iters  = args[2]
    alpha  = args[3]
    reg    = args[4]
    c      = args[5]

    R = init_R.copy()
    for t in tqdm(range(iters)):
        grad = obj_grad(R, X, labels, alpha, reg, c)
        R += lr * grad
        for j in range(len(R)):
            if R[j, 1] < R[j, 0]:
                print('Sides crossed!')
                temp = R[j, 1]
                R[j, 0] = R[j, 1]
                R[j, 1] = temp
        R[:, 0] = np.clip(R[:, 0], a_min = B[:, 0], a_max = None)
        R[:, 1] = np.clip(R[:, 1], a_max = B[:, 1], a_min = None)
        if t % 5 == 0:
            print(np.linalg.norm(grad))
            print(R)
    return R