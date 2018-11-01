# Import Statements
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from scipy import optimize
import time
from datetime import timedelta, datetime
import random
import csv
from math import floor
from methods_helper import *
from preprocess import *


PREDICTED = 'label'
CV_DATES = [('2010-12-31', '2011-12-31'),
            ('2011-12-31', '2012-12-31'),
            ('2012-12-31', '2013-12-31'),
            ('2013-12-31', '2014-12-31')]


def model_ready(clean_train, clean_test, features):
    '''
    subsets model based on features selected
    generates training and test sets for model fitting
    '''
    features = list(features)
    x_train = clean_train[features]
    y_train = clean_train[PREDICTED]
    x_test = clean_test[features]
    y_test = clean_test[PREDICTED]
    return x_train, y_train, x_test, y_test


def temporal_validation_loop(cv_pairs, grid_size, to_run, basic, filename):
    '''
    loops over different models and params for different time splits
    '''

    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters',
                                        'cutoff_date', 'validation_date',
                                        'train_set_size', 'validation_set_size', 'baseline',
                                        'precision_at_5','precision_at_10','precision_at_20', 'precision_at_30', 'precision_at_50',
                                        'recall_at_5','recall_at_10','recall_at_20', 'recall_at_30', 'recall_at_50',
                                        'auc-roc',
                                        'max_risk_score', 'min_risk_score', 'mean_risk_score', 'median_risk_score',
                                        'time_elapsed'))

    classifiers, grid = define_clfs_params(grid_size)

    for c, v in cv_pairs:

        print('CUTOFF: {} VALIDATION: {}'.format(c, v))
        train = pd.read_pickle('data/c{}_v{}_train.pkl'.format(c[:4], v[:4]))
        test = pd.read_pickle('data/c{}_v{}_test.pkl'.format(c[:4], v[:4]))

        # preprocess and split data
        train, test, features = pre_process(train, test)
        if basic:
            features = basic

        X_train, y_train, X_test, y_test = model_ready(train, test, features)

        for i, clf in enumerate([classifiers[x] for x in to_run]):
            model_name = to_run[i]
            print(model_name)
            params = grid[to_run[i]]
            for p in ParameterGrid(params):
                try:
                    print(p)
                    clf.set_params(**p)
                    row = get_row(model_name, clf, p, c, v, X_train, y_train, X_test, y_test)
                    results_df.loc[len(results_df)] = row

                except IndexError:
                    print('Error')
                    continue

        results_df.to_pickle(filename)

    return results_df


def get_row(model_name, clf, params, cutoff_date, validation_date,
            X_train, y_train, X_test, y_test):
    '''
    gets row of results for given model with given parameters
    '''

    start_time = time.time()
    y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
    end_time = time.time()
    tot_time = end_time - start_time  # time to train and test model

    # distribution of risk score
    max_score, min_score, mean_score = y_pred_probs.max(), y_pred_probs.min(), y_pred_probs.mean()
    med_idx = floor(len(y_pred_probs_sorted)/2)
    median_score = y_pred_probs_sorted[med_idx]

    # metrics
    precision_5, accuracy_5, recall_5 = scores_at_k(y_test_sorted,y_pred_probs_sorted,5.0)
    precision_10, accuracy_10, recall_10 = scores_at_k(y_test_sorted,y_pred_probs_sorted,10.0)
    precision_20, accuracy_20, recall_20 = scores_at_k(y_test_sorted,y_pred_probs_sorted,20.0)
    precision_30, accuracy_30, recall_30 = scores_at_k(y_test_sorted,y_pred_probs_sorted,30.0)
    precision_50, accuracy_50, recall_50 = scores_at_k(y_test_sorted,y_pred_probs_sorted,50.0)

    print(precision_5)
    plot_precision_recall_n(y_test, y_pred_probs, clf, False)

    row = [model_name, clf, params, cutoff_date, validation_date,
            y_train.shape[0], y_test.shape[0],
            scores_at_k(y_test_sorted, y_pred_probs_sorted,100.0)[0],
            precision_5, precision_10, precision_20, precision_30, precision_50,
            recall_5, recall_10, recall_20, recall_30, recall_50,
            roc_auc_score(y_test, y_pred_probs),
            max_score, min_score, mean_score, median_score,
            tot_time]

    return row