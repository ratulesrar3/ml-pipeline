# loading and exploring module
# ratul esrar, spring 18

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split


def read_from_csv(filename, index=None, split=False, target=None):
	'''
	Using os.path.splitext(file_name)[-1].lower(), find the extension of filename and then read into pandas dataframe
	Found here for reference:
	    http://stackoverflow.com/questions/5899497/checking-file-extension
	'''
	ext = os.path.splitext(filename)[-1].lower()

	if ext == '.csv':
		df = pd.read_csv(filename, index_col=index)
		if split:
			X_train, X_test, y_train, y_test = train_test_split(df.drop([target], axis=1), df[target], test_size=0.25, random_state=3)
			return X_train, X_test, y_train, y_test
	else:
		print('Incorrect file extension')
	return df


def count_nulls(df):
	'''
	Return number of null values for each column
	'''
	return df.isnull().sum()


def plot_correlations(df, title):
	'''
	Plot heatmap of columns in dataframe
	'''
	ax = plt.axes()
	corr = df.corr()
	sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, ax=ax)
	ax.set_title(title)


def plot_dist(df, col, title, normal=True):
	'''
	Plot distribution of a column
	'''
	ax = plt.axes()
	if normal:
		sns.distplot(df[col], fit=stats.norm, kde=False, ax=ax)
	else:
		sns.distplot(df[col], kde=False, ax=ax)
	ax.set_title(title)


def corr_matrix(df, col, save=False, filename=None):
    '''
    Plot correlation between all variables
    '''
    corr = df[col].corr()
    sns.heatmap(corr, xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.title('Correlation Matrix')
    plt.show()
    if save:
        plt.savefig(filename)


def density_plot(df, column, log_=False, ignore_null=True):
    '''
    Plot density of variable
    Refuses to plot Missing Values
    '''

    if ignore_null:
        x = df[column].dropna()
    else:
        x = df[column]

    if log_:
        sns.distplot(x.apply(logify))
        plt.title('Log {}'.format(column))
    else:
        sns.distplot(x)
        plt.title(column)

    plt.show()


def logify(x):
    if x > 0:
        return log(x)
    else:
        return 0


def plot_hist(df, col, label, top_k, sort=True):
    '''
    plots histogram of column
    '''

    if sort:
        if top_k:
            hist_idx = df[col].value_counts().head(top_k)
        else:
            hist_idx = df[col].value_counts()
    else:
        hist_idx = df[col].value_counts(sort=False)

    graph = sns.countplot(x=col, saturation=1, data=df, order=hist_idx.index)
    plt.ylabel('Number in Sample')
    plt.xlabel(label)
    plt.title('Distribution of {}'.format(label))
    plt.show()
    

def count_values(df, col, sort_by='hash_ssn', ascending=False):
    '''
    Find the values that make up a particular column of interst
    '''
    groupby = df.groupby(col, sort=False).count()
    return groupby.sort_values(by=sort_by, ascending=False)


def plot_pies(values, labels, colors = ['violet', 'yellow']):
    '''
    Plots a pie chart
    '''

    plt.pie(values, labels = labels, shadow=False, colors= colors, startangle=90, autopct='%1.1f%%')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    

def pie_vals(df, cols1, ordered='hash_ssn', labels = '', colors = ["violet", "yellow", "green", "blue", "orange"]):
    '''
    Pie chart for the top values in any given column
    '''

    df_to_plot = count_values(df, cols1)
    df_to_plot = df_to_plot[ordered]
    if labels == '':
        labels = tuple(df_to_plot.index)

    plot_pies(tuple(df_to_plot), labels, colors)
    


def graph_crosstab(df, col1, col2):
    '''
    Graph crosstab of two discrete variables

    Inputs: Dataframe, column names (strings)
    '''
    pd.crosstab(df[col1], df[col2]).plot(kind='bar', figsize=(20,20))
    plt.title(col2 + " " + "distribution by" + " " + col1)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.show()