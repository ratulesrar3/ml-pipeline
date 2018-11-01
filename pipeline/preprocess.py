import os
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler


def pre_process(train, test):
    aggregate = re.compile(".*yr$")
    fill_zero = [col for col in train.columns if aggregate.search(col)]
    for col in fill_zero:
        impute_missing(train, col, 'zero')
        impute_missing(test, col, 'zero')

    features = set(fill_zero)
    for col in train.columns:
        if col in ['lsir_cls', 'race']: # categorical all distinct values
            dummies = top_categories(train, col, top_k=-1)
            x = set(['{}_is_{}'.format(col, str(val)) for val in dummies])
            features = features.union(x)
            dummify(dummies, train, col)
            dummify(dummies, test, col)
        if col == 'booking_date':
            get_day_month_season(train, date_col='booking_date', month_col='booking_month', season='booking_season')
            get_day_month_season(test, date_col='booking_date', month_col='booking_month', season='booking_season')
            features = features.union({'booking_season_is_Summer', 'booking_season_is_Winter',
                                       'booking_season_is_Fall', 'booking_season_is_Spring',
                                       'booking_date_day_of_week_is_Monday', 'booking_date_day_of_week_is_Sunday',
                                       'booking_date_day_of_week_is_Friday', 'booking_date_day_of_week_is_Wednesday',
                                       'booking_date_day_of_week_is_Thursday', 'booking_date_day_of_week_is_Saturday',
                                       'booking_date_day_of_week_is_Tuesday'})
        if col == 'state':
            dummies = top_categories(train, col, top_k=2) # KS, MO
            x = set(['{}_is_{}'.format(col, str(val)) for val in dummies])
            features = features.union(x)
            dummify(dummies, train, col)
            dummify(dummies, test, col)
        if col == 'city':
            dummies = top_categories(train, col, top_k=5)
            x = set(['{}_is_{}'.format(col, str(val)) for val in dummies])
            features = features.union(x)
            dummify(dummies, train, col)
            dummify(dummies, test, col)
        if col == 'sex':
            convert_bad_boolean(train, col, 'FEMALE')
            convert_bad_boolean(test, col, 'FEMALE')
            features = features.add('sex')
        if col == 'bail_type':
            discretize_bail(train, 'bail_type')
            discretize_bail(test, 'bail_type')
            dummies = top_categories(train, col + '_discrete', top_k=-1)
            x = set(['{}_is_{}'.format(col, str(val)) for val in dummies])
            features = features.union(x)
            dummify(dummies, train, col + '_discrete')
            dummify(dummies, test, col + '_discrete')
        if col == 'age':
            features.union({'age'})


        '''
        currently only getting capricorns! work in progress

        train['Astro Sign'] = train.apply(lambda train: whats_yer_sign(train['bday'], train['bmonth']), axis=1)
        test['Astro Sign'] = test.apply(lambda test: whats_yer_sign(test['bday'], test['bmonth']), axis=1)
        dummies = top_categories(train, "Astro Sign", top_k=-1)
        x = set(['Astro_sign_is_{}'.format(str(val)) for val in dummies])
        features = features.union(x)
        dummify(dummies, train, 'Astro Sign')
        dummify(dummies, test, 'Astro Sign')
        '''

        return train, test, features


def make_bins(df, col, num_bins=15):
    '''
    Assigns column values into bins
    '''
    new_col = str(col)+'_bin'
    df[new_col] = pd.cut(df[col], bins=num_bins)

    return df


def categorical_dummies(df, columns):
    '''
    Pandas function wrapper to inplace combine a new set of dummy variable columns
    '''
    for column in columns:
        dummies = pd.get_dummies(df[column], prefix=column+"_is", prefix_sep='_', dummy_na=True)
        df = pd.concat([df, dummies], axis=1)

    return df


def impute_missing(df, column, fill_type):
    '''
    Fill in missing values using method specified
    '''
    if fill_type == 'median':
        df[column].fillna(df[column].median(), inplace=True)
    if fill_type == 'mean':
        df[column].fillna(df[column].mean(), inplace=True)
    if fill_type == 'zero':
        df[column].fillna(0, inplace=True)


def impute_by(df, by='mean'):
    '''
    Replace the NaNs with the column mean, median, or mode
    '''
    null_cols = df.columns[pd.isnull(df).sum() > 0].tolist()
    for column in null_cols:
        # input the mean of the data if they are numeric
        data = df[column]
        if data.dtype in [int, float]:
            if by is 'median':
                imputed_value = data.median()
            elif by is 'mode':
                imputed_value = data.mode()
            else:
                imputed_value = data.mean()
            df.loc[:,(column)] = data.fillna(imputed_value)

    return df


def scale_cols(train_df, test_df, col):
    '''
    Scale columns that do not follow a well-defined distribution
    '''
    pd.options.mode.chained_assignment = None
    robust_scaler = RobustScaler()
    scaled_col = str(col)+'_scaled'
    train_df[scaled_col] = robust_scaler.fit_transform(train_df[col].values.reshape(-1,1))
    test_df[scaled_col] = robust_scaler.transform(test_df[col].values.reshape(-1,1))

    return train_df, test_df


def to_discretize(df, var):
    '''
    This function discretizes variables into more workable ranges.

    The ranges are not automated.
    '''

    age_bins = range(0, 110, 10)

    if var == 'age':
        df['age_groups'] = pd.cut(df[var], age_bins, labels=False)

    del df[var]


def top_categories(df, col, top_k=-1):
    '''
    get top five categories + nan/others
    make top k or top percent?

    '''
    if top_k != -1:
        dummies = set(df[col].value_counts().head(top_k).index)
        dummies = dummies.union({'others'})

    else:
        dummies = set(df[col].value_counts().index)


    if df[col].isnull().any():
        dummies = dummies.union({np.nan})

    return dummies


def dummify(dummies, df, col):
    '''
    '''
    for val in dummies:
        col_name = '{}_is_{}'.format(col, str(val))
        if val != 'others':
            df.loc[:, col_name] = df[col].apply(lambda x: 1 if x == val else 0)
        else:
            df.loc[:, col_name] = df[col].apply(lambda x: 1 if x not in dummies else 0)
        #df = pd.concat([df, df[col_name]], axis=1)


def convert_bad_boolean(df, column, val='t'):
    '''
    converts boolean value in inconsistent format
    to 1 if true, 0 if false
    '''
    df.loc[:,column] = df[column].apply(lambda x: 1 if x == val else 0)


def convert_lsir_cls(x):

    lsir_cls = {'Low-Medium': 1,
                'High-Medium': 2,
                'Minimum': 0,
                'Maximum': 3}

    if x != 0:
        return lsir_cls[x]
    else:
        return x


def disaggregate_thyme(df, col, interval):

    col_name = '{}_{}'.format(col, interval)
    if interval == 'year':
        df.loc[:,col_name] = df[col].apply(lambda x: x.year)
    elif interval == 'month':
        df.loc[:,col_name] = df[col].apply(lambda x: x.month)


def cap_extreme(df, column, lb=0.001, ub=0.999):
    '''
    cap extreme outliers using quantile specified as max value
    '''
    lb, ub = df[column].quantile(lb), df[column].quantile(ub)
    print('Column was capped between {} and {}.'.format(lb, ub))
    df.loc[:, column] = df[column].apply(cap_value, args=(lb, ub))


def cap_value(x, lb, ub):
    '''
    helper function that returns cap for values exceeding it,
    0 for values
    itself otherwise
    '''
    if x > ub:
        return ub
    elif x < lb:
        return lb
    else:
        return x


# get season based on month
def get_season(month):
    '''
    Discretizes season based on month
    season = {
        'Winter':0, 'Spring':1, 'Summer':2,
        'Fall':3
    }
    '''
    if month in range(1, 4):
        return 'Winter'
    elif month in range(4, 7):
        return 'Spring'
    elif month in range(7, 10):
        return 'Summer'
    else:
        return 'Fall'


# booking/release day/month/season
# day of the week/month of year the subject was booked/released on
def get_day_month_season(df, date_col='booking_date', month_col='booking_month', season='booking_season'):
    '''
    Takes a date column and adds columns for booking day of week and month of year
    '''
    df[month_col] = pd.to_datetime(df[date_col]).dt.month
    df[season] = df[month_col].apply(get_season)

    # gets columns --> is_winter, is_fall, etc
    dummies = top_categories(df, season)
    dummify(dummies, df, season)

    day = date_col+'_day_of_week'
    df[day] = pd.to_datetime(df[date_col]).dt.weekday_name
    dummies = top_categories(df, day)
    dummify(dummies, df, day)


# bail posting
def get_bail_type(bail_type):
    '''
    Gets discretized bail types
    '''

    self_posted = set(['CA', 'CA-SU', 'GPS', 'PPS'])
    not_required = set(['PR'])
    bondsman_posted = set(['SUR'])

    if bail_type in not_required:
        return 'self_posted'
    elif bail_type in self_posted:
        return 'not_required'
    elif bail_type in bondsman_posted:
        return 'bondsman_posted'
    else:
        return np.nan


# bail posting type
def discretize_bail(df, bail_col='bail_type'):
    '''
    Applies discretized bail types to df
    '''
    new_col = bail_col+'_discrete'
    df[new_col] = df[bail_col].apply(get_bail_type)


#hi there
def whats_yer_sign(bday, bmonth):
    day = int(bday)
    month = int(bmonth)
    dob = (month*100) + day

    sign_list = [(120,"Capricorn"),
            (218,"Aquarius"),
            (320,"Pisces"),
            (420,"Aries"),
            (521,"Taurus"),
            (621,"Gemini"),
            (722,"Cancer"),
            (823,"Leo"),
            (923,"Virgo"),
            (1023,"Libra"),
            (1122,"Scorpio"),
            (1222,"Sagittarius"),
            (1231,"Capricorn")]

    for pair in sign_list:
        if dob <= pair[0]:
            sign = pair[1]
    return sign