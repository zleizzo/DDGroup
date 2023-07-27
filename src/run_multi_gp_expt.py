import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from utils import *

seed = 104
dir_data = '../data/regr'

datasets = ['Korea_grip', 'Dutch_drinking_inh', 'Dutch_drinking_wm', 'Dutch_drinking_sha', 
            'Brazil_health_heart', 'Brazil_health_stroke', 'China_glucose_women2', 'China_glucose_men2', 
            'Spain_Hair', 'China_HIV']
p_cores  = [0.01, 0.05, 0.1, 0.15, 0.2]
# alphas   = [0.01, 0.05, 0.1, 0.2, 0.4]
alphas = [2 ** (i - 4) for i in range(10)]
refits   = [True]
speed_types  = ['same', 'scaled']
num_groups = 3

job = int(sys.argv[1])
# job = 0

dataset = datasets[job % len(datasets)]
job = int(job / len(datasets))

p_core = p_cores[job % len(p_cores)]
job = int(job / len(p_cores))

alpha = alphas[job % len(alphas)]
job = int(job / len(alphas))

refit = refits[job % len(refits)]
job = int(job / len(refits))

speed_type = speed_types[job % len(speed_types)]

# dataset = sys.argv[1]
# n_core  = int(sys.argv[2])
# alpha   = float(sys.argv[3])
# speeds  = [float(s) for s in sys.argv[4].split(',')]
# speeds  = None
# refit   = bool(sys.argv[5])

# Load dataset & split into train/test/val
X, Y, names_covariates = load_regr_data(dataset)
n, d = X.shape
B = np.column_stack([np.min(X, axis = 0) - 0.1, np.max(X, axis = 0) + 0.1]) # Bounding box

if speed_type == 'same':
    speeds = None
else:
    speeds = [1. / (B[i, 1] - B[i, 0]) for i in range(len(B))]

X = np.concatenate([X, np.ones((len(X), 1))], axis = 1) # Add bias

Y = StandardScaler().fit_transform(Y.reshape(-1, 1)).reshape(-1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=seed)
X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.4, random_state=seed)

used_points = np.zeros(len(Y_train))

val_results = {}
test_results = {}

for r in range(num_groups):
    print(f'Finding group number {r+1}.')

    if sum(used_points) == len(used_points):
        val_results[r] = {
            'beta' : None,
            'R'    : None,
            'mse'  : -1,
            'vol'  : -1,
            'inc'  : -1,
            'msevvol' : -1,
            'msevinc' : -1
        }
        test_results[r] = {
            'beta' : None,
            'R'    : None,
            'mse'  : -1,
            'vol'  : -1,
            'inc'  : -1,
            'msevvol' : -1,
            'msevinc' : -1
        }
        continue

    # Find the core group & resulting core model
    print('Starting core set search.')
    n_core = int(p_core * len(Y_train))
    X_core, Y_core = neighbor_core(X_train, Y_train, n_core)
    print('Finished core set search.')

    beta, min_eig, s_hat = core_fit(X_core, Y_core)
    center = np.mean(X_core[:, :-1], axis = 0)
    assert n_core == len(Y_core)


    # Compute the inclusion labels
    print('Compute inclusion labels.')
    new_labels = hard_grow_labels(X_train, Y_train, alpha, s_hat, min_eig, n_core, beta)
    labels = np.maximum(used_points, new_labels) # Labels keeps track of rejected points for this core fit AND points belonging to existing groups


    # Compute the region & refit the model if necessary
    print('Compute the region.')
    R = hard_grow_region(X_train[:, :-1], labels, B, center, speeds)
    ind = in_box(X_train[:, :-1], R)
    used_points = np.maximum(used_points, ind)

    if refit:
        X_fit = X_train[ind]
        Y_fit = Y_train[ind]
        beta = np.linalg.solve(X_fit.T @ X_fit, X_fit.T @ Y_fit)


    # Compute the validation & test statistics
    print('Compute validation and test stats.')
    # Validation
    ind_val = in_box(X_val[:, :-1], R)
    X_val_incl = X_val[ind_val]
    Y_val_incl = Y_val[ind_val]

    mse_val = np.mean((X_val_incl @ beta - Y_val_incl) ** 2)
    vol_val = box_intersection(R, R)
    inc_val = len(Y_val_incl)

    msevvol_val = mse_val / vol_val
    msevinc_val = mse_val / inc_val

    # Test
    ind_test = in_box(X_test[:, :-1], R)
    X_test_incl = X_test[ind_test]
    Y_test_incl = Y_test[ind_test]

    mse_test = np.mean((X_test_incl @ beta - Y_test_incl) ** 2)
    vol_test = box_intersection(R, R)
    inc_test = len(Y_test_incl)

    msevvol_test = mse_test / vol_test
    msevinc_test = mse_test / inc_test

    val_results[r] = {
        'beta' : beta.tolist(),
        'R'    : R.tolist(),
        'mse'  : mse_val,
        'vol'  : vol_val,
        'inc'  : int(inc_val),
        'msevvol' : msevvol_val,
        'msevinc' : msevinc_val
    }
    assert val_results[r]['inc'] <= len(Y_val), 'Error with validation set!'

    test_results[r] = {
        'beta' : beta.tolist(),
        'R'    : R.tolist(),
        'mse'  : mse_test,
        'vol'  : vol_test,
        'inc'  : int(inc_test),
        'msevvol' : msevvol_test,
        'msevinc' : msevinc_test
    }
    assert test_results[r]['inc'] <= len(Y_test), 'Error with test set!'

# Save the results
print('Saving results.')
val_res_filename = f'../data/regr/results/{seed}/{dataset},{p_core},{alpha},{speed_type},{refit},val.json'
with open(val_res_filename, 'w') as outfile:
    json.dump(val_results, outfile)

test_res_filename = f'../data/regr/results/{seed}/{dataset},{p_core},{alpha},{speed_type},{refit},test.json'
with open(test_res_filename, 'w') as outfile:
    json.dump(test_results, outfile)