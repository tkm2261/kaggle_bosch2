import pickle
import pandas as pd
import numpy as np
import glob
import pickle
import re
import gc
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool

from logging import getLogger
logger = getLogger(__name__)

from group import LIST_CAT_GROUP, LIST_NUM_GROUP, LIST_DATE_GROUP

TRAIN_CAT = '../input/train_categorical_2.csv.ftr'
TRAIN_NUM = '../input/train_numeric.csv.ftr'
TRAIN_DAT = '../input/train_date.csv.ftr'


TEST_CAT = '../input/test_categorical_2.csv.ftr'
TEST_NUM = '../input/test_numeric.csv.ftr'
TEST_DAT = '../input/test_date.csv.ftr'

MAP_COL = {col: cols[0] for cols in LIST_CAT_GROUP + LIST_NUM_GROUP + LIST_DATE_GROUP for col in cols}


def make_etl(df, output_path):
    cat_usecols = sorted([cols[0] for cols in LIST_CAT_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    num_usecols = sorted([cols[0] for cols in LIST_NUM_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    dat_usecols = sorted([cols[0] for cols in LIST_DATE_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    names = ['CAT', 'NUM', 'DAT']
    agg_name = ['mean', 'max', 'min', 'std']
    df_new = pd.DataFrame()
    for i, _cols in enumerate([cat_usecols, num_usecols, dat_usecols]):
        for line in tqdm(['L', 'L0', 'L1', 'L2', 'L3'], desc='line'):
            cols = [c for c in _cols if line in c]
            for j, agg in enumerate(tqdm([np.nanmean, np.nanmax, np.nanmin, np.nanstd], desc='agg')):
                tmp = agg(df[cols], axis=1)
                df_new[f'feat_{names[i]}_{line}_{agg_name[j]}'] = tmp

    df_new.to_feather(output_path)


def make_etl_section(df, output_path):
    cat_usecols = sorted([cols[0] for cols in LIST_CAT_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    num_usecols = sorted([cols[0] for cols in LIST_NUM_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    dat_usecols = sorted([cols[0] for cols in LIST_DATE_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    names = ['CAT', 'NUM', 'DAT']
    agg_name = ['mean', 'max', 'min', 'std']
    df_new = pd.DataFrame()
    for i, _cols in enumerate([cat_usecols, num_usecols, dat_usecols]):
        for line in tqdm(['S'] + [f'S{i}' for i in range(52)], desc='line'):
            cols = [c for c in _cols if line in c]
            if df[cols].shape[1] == 0:
                continue
            for j, agg in enumerate(tqdm([np.nanmean, np.nanmax, np.nanmin, np.nanstd], desc='agg')):
                tmp = agg(df[cols], axis=1)
                df_new[f'feat_{names[i]}_{line}_{agg_name[j]}'] = tmp

    df_new.to_feather(output_path)


def make_date_min(df, output_path):
    df_new = pd.DataFrame()
    dat_usecols = sorted([cols[0] for cols in LIST_DATE_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    for line in tqdm(['L', 'L0', 'L1', 'L2', 'L3'], desc='line'):
        cols = [c for c in dat_usecols if line in c]
        min_date = df[cols].min(axis=1)
        for col in cols:
            df_new[f'{col}-M-{line}'] = df[col] - min_date
    df_new.to_feather(output_path)


def make_num_pass_sec(df, output_path):
    df_new = pd.DataFrame()
    cat_usecols = sorted([cols[0] for cols in LIST_CAT_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    num_usecols = sorted([cols[0] for cols in LIST_NUM_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])

    usecols = cat_usecols + num_usecols
    for line in tqdm([f'S{i}' for i in range(52)], desc='line'):
        cols = [c for c in usecols if line in c]
        df_new[f'num_pass_{line}'] = df[cols].notnull().sum(axis=1)
    df_new.to_feather(output_path)


def make_diff(df, output_path):
    df_new = pd.DataFrame()
    dat_usecols = sorted([cols[0] for cols in LIST_DATE_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    num_usecols = sorted([cols[0] for cols in LIST_NUM_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])

    for cols in [dat_usecols, num_usecols]:
        for i in range(1, len(cols)):
            c1 = cols[i-1]
            c2 = cols[i]
            df_new[f'{c2}_diff_{c1}'] = df[c2] - df[c1]
    df_new.to_feather(output_path)


def make_hash_count(df, df_test):
    df_new = pd.DataFrame()

    train_num = df.shape[0]

    cat_usecols = sorted([cols[0] for cols in LIST_CAT_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    num_usecols = sorted([cols[0] for cols in LIST_NUM_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    dat_usecols = sorted([cols[0] for cols in LIST_DATE_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])

    usecols = cat_usecols + num_usecols + dat_usecols
    df = pd.concat([df[usecols], df_test[usecols]], axis=0, ignore_index=True)

    for line in tqdm(['L', 'L0', 'L1', 'L2', 'L3'], desc='line'):
        cols = [c for c in usecols if line in c]
        hashs = [','.join(map(str, row)) for row in tqdm(df[cols].values)]
        cnt = Counter(hashs)
        df_new[f'hash_{line}'] = [cnt[h] for h in hashs]

    df_new.iloc[:train_num].reset_index(drop=True).to_feather('train_hash_cnt.ftr')
    df_new.iloc[train_num:].reset_index(drop=True).to_feather('test_hash_cnt.ftr')


def make_hash_rank(df, df_test):
    df_new = pd.DataFrame()

    train_num = df.shape[0]

    cat_usecols = sorted([cols[0] for cols in LIST_CAT_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    num_usecols = sorted([cols[0] for cols in LIST_NUM_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    dat_usecols = sorted([cols[0] for cols in LIST_DATE_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])

    usecols = cat_usecols + num_usecols + dat_usecols
    df = pd.concat([df[usecols], df_test[usecols]], axis=0, ignore_index=True)

    for line in tqdm(['L', 'L0', 'L1', 'L2', 'L3'], desc='line'):
        cols = [c for c in usecols if line in c]
        df['hash'] = [','.join(map(str, row)) for row in tqdm(df[cols].values)]
        df_new[f'hash_rank_{line}'] = df.groupby('hash')['Id'].transform(
            lambda x: x.index.rank(method='first') if x.shape[0] > 1 else [-1]).values

    df_new.iloc[:train_num].reset_index(drop=True).to_feather('train_hash_rank.ftr')
    df_new.iloc[train_num:].reset_index(drop=True).to_feather('test_hash_rank.ftr')


def make_sr(row):
    return ','.join(map(str, row))


def make_hash_count_no_s38(df, df_test):
    df_new = pd.DataFrame()

    train_num = df.shape[0]

    cat_usecols = sorted([cols[0] for cols in LIST_CAT_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    num_usecols = sorted([cols[0] for cols in LIST_NUM_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    dat_usecols = sorted([cols[0] for cols in LIST_DATE_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])

    usecols = cat_usecols + num_usecols + dat_usecols
    df = pd.concat([df[usecols], df_test[usecols]], axis=0, ignore_index=True)

    for line in tqdm(['L', 'L0', 'L1', 'L2', 'L3'], desc='line'):
        cols = [c for c in usecols if line in c and 'S38' not in c]
        with Pool() as p:
            hashs = p.map(make_sr, tqdm(df[cols].values), chunksize=1000)
        cnt = Counter(hashs)
        df_new[f'hash_nos38_{line}'] = [cnt[h] for h in hashs]

    df_new.iloc[:train_num].reset_index(drop=True).to_feather('train_hash_cnt_nos38.ftr')
    df_new.iloc[train_num:].reset_index(drop=True).to_feather('test_hash_cnt_nos38.ftr')


def make_hash_count_sec(df, df_test):
    df_new = pd.DataFrame()

    train_num = df.shape[0]

    cat_usecols = sorted([cols[0] for cols in LIST_CAT_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    num_usecols = sorted([cols[0] for cols in LIST_NUM_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])
    dat_usecols = sorted([cols[0] for cols in LIST_DATE_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])

    usecols = cat_usecols + num_usecols + dat_usecols
    df = pd.concat([df[usecols], df_test[usecols]], axis=0, ignore_index=True)

    for line in tqdm(['S'] + [f'S{i}' for i in range(52)], desc='line'):
        cols = [c for c in usecols if line in c]
        with Pool() as p:
            hashs = p.map(make_sr, tqdm(df[cols].values), chunksize=1000)
        cnt = Counter(hashs)
        df_new[f'hash_{line}'] = [cnt[h] for h in hashs]

    df_new.iloc[:train_num].reset_index(drop=True).to_feather('train_hash_cnt_sec.ftr')
    df_new.iloc[train_num:].reset_index(drop=True).to_feather('test_hash_cnt_sec.ftr')


from numba import jit


@jit
def date_mean(_date_feat, _num_feat, i):
    if _date_feat[i] != _date_feat[i]:
        return np.nan, np.nan, np.nan, np.nan

    start = max(i - 100, 0)
    end = i + 100
    date_feat = _date_feat[start:end]
    num_feat = _num_feat[start:end]

    tmp = np.abs(date_feat - _date_feat[i])
    tmp[np.isnan(tmp)] = 100
    idx = tmp <= 0.1
    tmp = num_feat[idx]

    if tmp.shape[0] == 0:
        return np.nan, np.nan, np.nan, np.nan
    ret_mean = tmp.mean()
    ret_std = tmp.std()
    ret_max = tmp.max()
    ret_min = tmp.min()

    return ret_mean, ret_std, ret_max, ret_min


import functools


def make_time_mean(df, df_test):
    df_new = pd.DataFrame()

    train_num = df.shape[0]

    df = pd.concat([df, df_test], axis=0, ignore_index=True)
    df_new['tmp'] = np.arange(df.shape[0])

    dat_usecols = sorted([cols[0] for cols in LIST_DATE_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])

    for col in tqdm(dat_usecols, desc='dat_cols'):
        df = df.sort_values(col)

        num = int(col.split('_')[-1][1:])
        try:
            data_col = MAP_COL['_'.join(col.split('_')[:-1]) + '_F' + str(num - 1)]
        except KeyError:
            continue
        date_feat = df[col].values
        num_feat = df[data_col].values

        ret_mean = np.zeros(df.shape[0]) * np.nan
        ret_std = np.zeros(df.shape[0]) * np.nan
        ret_min = np.zeros(df.shape[0]) * np.nan
        ret_max = np.zeros(df.shape[0]) * np.nan

        date_mean_part = functools.partial(date_mean, date_feat, num_feat)
        with Pool() as p:
            ret = p.map(date_mean_part, tqdm(range(df.shape[0]), desc='col'), chunksize=30000)
        for i, (m, s, mx, mm) in enumerate(ret):
            ret_mean[i] = m
            ret_std[i] = s
            ret_max[i] = mx
            ret_min[i] = mm

        """
        for i in tqdm(range(df.shape[0]), desc='col'):
            idx = np.abs(date_feat - date_feat[i]) <= 0.1
            tmp = num_feat[idx]
            ret_mean[i] = tmp.mean()
            ret_std[i] = tmp.std()
        """
        df_new.loc[df.index.values, f'{data_col}_dur_mean'] = ret_mean
        df_new.loc[df.index.values, f'{data_col}_dur_std'] = ret_std
        df_new.loc[df.index.values, f'{data_col}_dur_max'] = ret_max
        df_new.loc[df.index.values, f'{data_col}_dur_min'] = ret_min

    df_new.drop('tmp', axis=1, inplace=True)
    df_new.sort_index(inplace=True)
    df_new.iloc[:train_num].reset_index(drop=True).to_feather('train_time_mean.ftr')
    df_new.iloc[train_num:].reset_index(drop=True).to_feather('test_time_mean.ftr')


def make_time_mean_norm(df, df_test):
    dat_usecols = sorted([cols[0] for cols in LIST_DATE_GROUP if cols[0] != 'Id' and cols[0] != 'Response'])

    df_new = pd.DataFrame()

    train_num = df.shape[0]

    df = pd.concat([df, df_test], axis=0, ignore_index=True)
    del df_test
    gc.collect()
    mm = df[dat_usecols].min(axis=1).values
    for col in tqdm(dat_usecols, desc='norm'):
        df[col] = df[col].values - mm

    df_new['tmp'] = np.arange(df.shape[0])

    for col in tqdm(dat_usecols, desc='dat_cols'):
        df = df.sort_values(col)

        num = int(col.split('_')[-1][1:])
        try:
            data_col = MAP_COL['_'.join(col.split('_')[:-1]) + '_F' + str(num - 1)]
        except KeyError:
            continue
        date_feat = df[col].values
        num_feat = df[data_col].values

        ret_mean = np.zeros(df.shape[0]) * np.nan
        ret_std = np.zeros(df.shape[0]) * np.nan
        ret_min = np.zeros(df.shape[0]) * np.nan
        ret_max = np.zeros(df.shape[0]) * np.nan

        date_mean_part = functools.partial(date_mean, date_feat, num_feat)
        with Pool() as p:
            ret = p.map(date_mean_part, tqdm(range(df.shape[0]), desc='col'), chunksize=30000)
        for i, (m, s, mx, mm) in enumerate(ret):
            ret_mean[i] = m
            ret_std[i] = s
            ret_max[i] = mx
            ret_min[i] = mm
        df_new.loc[df.index.values, f'{data_col}_dur_mean_norm'] = ret_mean
        df_new.loc[df.index.values, f'{data_col}_dur_std_norm'] = ret_std
        df_new.loc[df.index.values, f'{data_col}_dur_max_norm'] = ret_max
        df_new.loc[df.index.values, f'{data_col}_dur_min_norm'] = ret_min

    df_new.drop('tmp', axis=1, inplace=True)
    df_new.sort_index(inplace=True)
    df_new.iloc[:train_num].reset_index(drop=True).to_feather('train_time_mean_norm.ftr')
    df_new.iloc[train_num:].reset_index(drop=True).to_feather('test_time_mean_norm.ftr')


def load_train_data():
    logger.info('enter')

    usecols = sorted([cols[0] for cols in LIST_CAT_GROUP])
    df_cat = pd.read_feather(TRAIN_CAT, nthreads=8)[usecols]
    logger.info(f'cat size: {df_cat.shape}')

    usecols = sorted([cols[0] for cols in LIST_NUM_GROUP])
    df_num = pd.read_feather(TRAIN_NUM, nthreads=8)[usecols].drop('Id', axis=1)
    logger.info(f'num size: {df_num.shape}')

    usecols = sorted([cols[0] for cols in LIST_DATE_GROUP])
    df_dat = pd.read_feather(TRAIN_DAT, nthreads=8)[usecols].drop('Id', axis=1)
    logger.info(f'dat size: {df_dat.shape}')

    df = pd.concat([df_cat, df_num, df_dat], axis=1)
    logger.info(f'data size: {df.shape}')
    logger.info('exit')
    return df


def load_test_data():
    logger.info('enter')
    usecols = sorted([cols[0] for cols in LIST_CAT_GROUP])
    df_cat = pd.read_feather(TEST_CAT, nthreads=8)[usecols]
    logger.info(f'cat size: {df_cat.shape}')

    usecols = sorted([cols[0] for cols in LIST_NUM_GROUP if cols[0] != 'Response'])
    df_num = pd.read_feather(TEST_NUM, nthreads=8)[usecols].drop('Id', axis=1)
    logger.info(f'num size: {df_num.shape}')

    usecols = sorted([cols[0] for cols in LIST_DATE_GROUP])
    df_dat = pd.read_feather(TEST_DAT, nthreads=8)[usecols].drop('Id', axis=1)
    logger.info(f'dat size: {df_dat.shape}')

    df = pd.concat([df_cat, df_num, df_dat], axis=1)
    logger.info(f'data size: {df.shape}')
    logger.info('exit')
    return df


if __name__ == '__main__':
    from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger, NullHandler
    logger = getLogger()

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    handler = FileHandler('load_data.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    """
    df = load_train_data()
    df.to_feather('train_0713.ftr')

    df = load_test_data()  # pd.read_csv('test_0530.csv', parse_dates=['t_activation_date'], float_precision='float32')
    df.to_feather('test_0713.ftr')
    """
    """
    make_etl(pd.read_feather('train_0713.ftr', nthreads=8).astype('float32'), 'train_feat_agg.ftr')
    make_etl(pd.read_feather('test_0713.ftr', nthreads=8).astype('float32'), 'test_feat_agg.ftr')

    make_date_min(pd.read_feather('train_0713.ftr', nthreads=8).astype('float32'), 'train_date_min.ftr')
    make_date_min(pd.read_feather('test_0713.ftr', nthreads=8).astype('float32'), 'test_date_min.ftr')


    """
    # make_etl_section(pd.read_feather('train_0713.ftr', nthreads=8).astype('float32'), 'train_feat_agg_sec.ftr')
    # make_etl_section(pd.read_feather('test_0713.ftr', nthreads=8).astype('float32'), 'test_feat_agg_sec.ftr')
    # make_hash_count_sec(pd.read_feather('train_0713.ftr', nthreads=8).astype('float32'),
    #                    pd.read_feather('test_0713.ftr', nthreads=8).astype('float32'))

    # make_num_pass_sec(pd.read_feather('train_0713.ftr', nthreads=8).astype('float32'), 'train_num_pass_sec.ftr')
    # make_num_pass_sec(pd.read_feather('test_0713.ftr', nthreads=8).astype('float32'), 'test_num_pass_sec.ftr')

    # make_hash_count_no_s38(pd.read_feather('train_0713.ftr', nthreads=8).astype('float32'),
    #                       pd.read_feather('test_0713.ftr', nthreads=8).astype('float32'))

    # make_diff(pd.read_feather('train_0713.ftr', nthreads=8).astype('float32'), 'train_diff.ftr')
    # make_diff(pd.read_feather('test_0713.ftr', nthreads=8).astype('float32'), 'test_diff.ftr')

    # make_time_mean(pd.read_feather('train_0713.ftr', nthreads=8).astype('float32'),
    #               pd.read_feather('test_0713.ftr', nthreads=8).astype('float32'))

    # make_time_mean_norm(pd.read_feather('train_0713.ftr', nthreads=8).astype('float32'),
    #                    pd.read_feather('test_0713.ftr', nthreads=8).astype('float32'))

    make_hash_rank(pd.read_feather('train_0713.ftr', nthreads=8).astype('float32'),
                   pd.read_feather('test_0713.ftr', nthreads=8).astype('float32'))
