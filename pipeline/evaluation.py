import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.tree import export_graphviz
from methods_loop import *
from methods_helper import *
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness


def get_all_results(files):
    '''
    combines individual pickle files
    returns results df
    '''
    result_list = []
    for f in files:
        rdf = pd.read_pickle(f)
        result_list.append(rdf)

    results = pd.concat(result_list, axis=0)

    return results


def get_best_model(results_df, metric, threshold=-1):
    '''
    discard some garbage models
    can subset top k% performing model based on desired metrics
    default is the models with highest metric score
    '''


    # discard with fluke-y precision
    results_df = results_df[results_df[metric] < results_df[metric].mean() + results_df[metric].std() * 2]

    # discard models that assign same pred score to all
    results_df = results_df[results_df['max_risk_score'] != results_df['min_risk_score']]

    # filter models with auc-roc over .5
    results_df = results_df[results_df['auc-roc'] > 0.5]

    if threshold == -1:
        best_models = results_df[(results_df[metric] == results_df[metric].max())]

    else:
        best_models = results_df[(results_df[metric] > results_df[metric].quantile(threshold))]

    return best_models


def evaluate_best_models(best_results_df):
    '''
    get feature importance and bias info for best models
    '''

    train = pd.read_pickle('data/c2014_v2015_train.pkl')
    test = pd.read_pickle('data/c2014_v2015_test.pkl')
    train, test, ft = pre_process(train, test)
    X_train, y_train, X_test, y_test = model_ready(train, test, ft)


    for i, row in best_results_df.iterrows():
        model, clf, params = row.model_type, row.clf, row.parameters
        print(row)
        print(i, model, params, 'precision at 5: {}'.format(row.precision_at_5), 'precision at 10: {}'.format(row.precision_at_10))

        clf.set_params(**params)
        clf.fit(X_train, y_train)
        y_preds = clf.predict_proba(X_test)[:,1]

        plot_precision_recall_n(y_test, y_preds, clf, False)

        f = 'results/ft_{}_{}.png'.format(model, i)

        if model == 'LR':
            important_df = get_importance(clf, X_train, y_train, top_k=10, logit_=True, fname=f)
            corr_feature_matrix(train, important_df, 'label', save=False, filename=None)
        elif model == 'KNN':
            continue
        else:
            important_df = get_importance(clf, X_train, y_train, top_k=10, logit_=False, fname=f)
            corr_feature_matrix(train, important_df, 'label', save=False, filename=None)

        aequitas = get_bias_df(clf, X_train, y_train, X_test, y_test, test)
        bdf = measure_bias(aequitas, {'rank_abs': [200]}, {'race': 'WHITE', 'sex': 'MALE'}, ['race', 'sex'])
        print(bdf)


def print_tree(dtree, col_names, dot_name):
    '''
    '''
    viz = export_graphviz(dtree, feature_names = col_names,
                    class_names=['Will not seek MH', 'Will seek MH'],
                    rounded=False, filled=True, out_file=dot_name)


def time_split_trees(p={'criterion': 'entropy', 'max_depth': 2},
                     cv_pairs=[('2010-12-31', '2011-12-31'),
                               ('2011-12-31', '2012-12-31'),
                               ('2012-12-31', '2013-12-31'),
                               ('2013-12-31', '2014-12-31')], basic=None):
    '''
    print(tree)
    '''
    for c, v in cv_pairs:

        print('CUTOFF: {} VALIDATION: {}'.format(c, v))
        train = pd.read_pickle('data/c{}_v{}_train.pkl'.format(c[:4], v[:4]))
        test = pd.read_pickle('data/c{}_v{}_test.pkl'.format(c[:4], v[:4]))

        train, test, features = pre_process(train, test)
        if basic:
            features = basic

        X_train, y_train, X_test, y_test = model_ready(train, test, features)

        dtree = DecisionTreeClassifier()
        dtree.set_params(**p)
        dtree.fit(X_train, y_train)
        y_pred_probs = dtree.predict_proba(X_test)[:,1]
        y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
        precision_5, accuracy_5, recall_5 = scores_at_k(y_test_sorted,y_pred_probs_sorted,5.0)
        print(precision_5)

        dot_name = 'results/c{}_v{}_tree.dot'.format(c[:4], v[:4])
        png_name = dot_name.split('.')[0]

        viz = export_graphviz(dtree, feature_names=X_train.columns,
                        class_names=['Will not seek MH', 'Will seek MH'],
                        rounded=False, filled=True, out_file=dot_name)
        #print_tree(dtree, X_train.columns, dot_name)

        os.system("dot -Tpng {} -o {}.png ".format(dot_name, png_name))


def get_importance(clf, X, y, top_k=-1, logit_=False, fname=None):
    '''
    generalized ft importance fcn
    if logistic regression: use coefficients
    else use feature_importances_

    # add SIGN of importance
    '''

    clf.fit(X, y)

    if logit_:
        importances = clf.coef_[0]
    else:
        importances = clf.feature_importances_

    indices = np.argsort(importances)[::-1]
    xrange = range(X.shape[1])

    if top_k != -1:
        xrange = range(top_k)
        indices = np.argsort(importances)[::-1][:top_k]


    #Plot the feature importances of the classfier
    plt.figure(figsize=(15,12))
    plt.title("Feature Importances", fontsize=30)
    plt.bar(xrange, importances[indices], color=sns.color_palette("Set2", 10), align="center")
    plt.xticks(xrange, X.columns[indices], rotation=90, fontsize=18)
    plt.ylabel('Importance', fontsize=18)

    plt.show()

    if fname:
        plt.savefig(fname)

    # Feature Ranking DataFrame
    importance = pd.DataFrame(columns=['feature', 'importance'])

    for f in xrange:
        importance.loc[f+1] = [X.columns[indices[f]], importances[indices[f]]]

    return importance


def corr_feature_matrix(df, important_df, label_col, save=False, filename=None):
    '''
    Plot correlation between all important variables
    '''
    cols = list(important_df['feature']) + [label_col]
    corr = df[cols].corr()
    sns.heatmap(corr, xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.title('Correlation Matrix of Important Features')
    plt.show()
    if save:
        plt.savefig(filename)


def get_200_indiv(clf):
    '''
    get top 200 riskiest indivs based on model
    '''
    return


def get_bias_df(clf, X_train, y_train, X_test, y_test, test_raw):
    '''
    create cols race, sex, age_cat
    '''
    clf.fit(X_train, y_train)
    y_pred_probs = clf.predict_proba(X_test)[:,1]
    bias_df = test_raw[['hash_ssn', 'release_date', 'sex', 'race', 'label']]
    bias_df = pd.concat([bias_df, pd.DataFrame(y_pred_probs)], axis=1, join='inner')
    bias_df.columns = ['hash_ssn', 'release_date', 'sex', 'race', 'label_value', 'score']
    return bias_df


def measure_bias(df, score_thresholds={'rank_abs': [200]}, groups= {'race': 'WHITE','sex': 'MALE'},
                                                        attr_cols = ['race', 'sex']):
    '''
    measure bias and fairness of selected model using AEQUITAS
    Taken from demo:
    https://github.com/dssg/aequitas/blob/master/docs/source/examples/compas_demo.ipynb
    Docs:
    https://dssg.github.io/aequitas/30_seconds_aequitas.html#Input-machine-learning-predictions

    1. df must cont following columns:
        - 'score': pred_score
        - 'label_value', true_label
    2. attr_cols should be catgorical cols
    '''

    g, b, f = Group(), Bias(), Fairness()
    xtab, _ = g.get_crosstabs(df, score_thresholds=score_thresholds, attr_cols=attr_cols)

    bdf = b.get_disparity_predefined_groups(xtab, groups)
    fdf = f.get_group_value_fairness(bdf)
    cols = ['attribute_name', 'attribute_value',
            'fdr_disparity', 'fpr_disparity', 'for_disparity', 'fnr_disparity', 'precision']
            # 'Statistical Parity', 'Impact Parity',
            # 'FDR Parity', 'FPR Parity', 'FOR Parity', 'FNR Parity',
            # 'TypeI Parity', 'TypeII Parity', 'Unsupervised Fairness', 'Supervised Fairness',
    gaf = f.get_group_attribute_fairness(fdf)

    return fdf[cols]
