import pandas as pd
import numpy as np
from scipy import sparse
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
# import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, log_loss
import gc
from logging import getLogger
logger = getLogger(None)

from tqdm import tqdm

from load_data import load_train_data, load_test_data
import sys
DIR = 'result_tmp/'
DTYPE = 'float32'
print(DIR)
print(DTYPE)

from utils import mcc_optimize


def cst_metric_xgb(pred, dtrain):
    label = dtrain.get_label().astype(np.int)
    best_proba, best_mcc = mcc_optimize(pred, label)
    return 'mcc', best_mcc, True


def callback(data):
    if (data.iteration + 1) % 10 == 0:
        print('progress: ', data.iteration + 1)
        return


from scipy import sparse


def train():

    df = pd.concat([pd.read_feather('train_0713.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('train_feat_agg.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('train_feat_agg_sec.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('train_hash_cnt.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('train_hash_cnt_nos38.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('train_hash_cnt_sec.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('train_date_min.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('train_num_pass_sec.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('train_diff.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('train_time_mean.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('train_time_mean_norm.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('train_magic.ftr', nthreads=8)[
        ['magic1', 'magic2', 'magic3', 'magic4']].astype(DTYPE),
    ], axis=1)

    df_cols = pd.read_csv('result_0715_allfeat/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

    df_cols = pd.read_csv('result_0715_magic/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

    df_cols = pd.read_csv('result_0715_sec/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

    df_cols = pd.read_csv('result_0716_sec_hash/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

    df_cols = pd.read_csv('result_0716_num_sec/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

    df_cols = pd.read_csv('result_0716_rate001/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

    df_cols = pd.read_csv('result_0717_s38/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

    df_cols = pd.read_csv('result_0717_diff/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

    df_cols = pd.read_csv('result_0717_time_mean/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

    df_cols = pd.read_csv('result_0718_time_mean/feature_importances.csv')
    drop_cols = df_cols[df_cols['imp'] == 0]['col'].values
    df.drop(drop_cols, axis=1, errors='ignore', inplace=True)

    logger.info(f'load 1 {df.shape}')

    y_train = df['Response'].values
    df.drop(['Response', 'Id'], axis=1, errors='ignore', inplace=True)

    logger.info(f'load dropcols {df.shape}')
    gc.collect()
    x_train = df.values  # sparse.csc_matrix(df.values, dtype=DTYPE)

    usecols = df.columns.values.tolist()

    del df
    gc.collect()

    logger.info('train data size {}'.format(x_train.shape))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)

    with open(DIR + 'usecols.pkl', 'wb') as f:
        pickle.dump(usecols, f, -1)

    # {'boosting_type': 'gbdt', 'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_bin': 511, 'max_depth': -1, 'metric': 'None', 'min_child_weight': 10, 'min_split_gain': 0, 'num_leaves': 31, 'objective': 'binary', 'reg_alpha': 1, 'scale_pos_weight': 1, 'seed': 114, 'subsample': 0.99, 'subsample_freq': 1, 'verbose': -1, 'xgboost_dart_mode': True}
    all_params = {'min_child_weight': [10],
                  'subsample': [0.99],
                  'subsample_freq': [1],
                  'seed': [114],
                  'colsample_bytree': [0.7],
                  'learning_rate': [0.01],
                  'max_depth': [-1],
                  'min_split_gain': [0],
                  'reg_alpha': [1],
                  'max_bin': [511],
                  'num_leaves': [31],
                  'objective': ['binary'],
                  'scale_pos_weight': [1],
                  'verbose': [-1],
                  'boosting_type': ['gbdt'],
                  'metric': ["None"],
                  'xgboost_dart_mode': [True],
                  # 'device': ['gpu'],
                  }

    use_score = 0
    min_score = (100, 100, 100)
    for params in tqdm(list(ParameterGrid(all_params))):
        cnt = -1
        list_score = []
        list_score2 = []
        list_best_iter = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in cv.split(x_train, y_train):
            cnt += 1
            trn_x = x_train[train]  # [[i for i in range(x_train.shape[0]) if train[i]]]
            val_x = x_train[test]  # [[i for i in range(x_train.shape[0]) if test[i]]]
            trn_y = y_train[train]
            val_y = y_train[test]
            train_data = lgb.Dataset(trn_x,  # .values.astype(np.float32),
                                     label=trn_y,
                                     feature_name=usecols
                                     )
            test_data = lgb.Dataset(val_x,  # .values.astype(np.float32),
                                    label=val_y,
                                    feature_name=usecols
                                    )
            del trn_x
            gc.collect()
            clf = lgb.train(params,
                            train_data,
                            100000,  # params['n_estimators'],
                            early_stopping_rounds=500,
                            valid_sets=[test_data],
                            feval=cst_metric_xgb,
                            # callbacks=[callback],
                            verbose_eval=10
                            )
            pred = clf.predict(val_x).clip(0, 1)

            all_pred[test] = pred
            best_proba, best_mcc = mcc_optimize(pred, val_y)
            _score = - best_mcc
            _score2 = log_loss(val_y, pred)

            logger.info('   _score: %s' % _score)
            logger.info('   _best_proba: %s' % best_proba)
            logger.info('   _score2: %s' % _score2)

            list_score.append(_score)
            list_score2.append(_score2)

            if clf.best_iteration != 0:
                list_best_iter.append(clf.best_iteration)
            else:
                list_best_iter.append(params['n_estimators'])

            with open(DIR + 'train_cv_pred_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(pred, f, -1)
            with open(DIR + 'model_%s.pkl' % cnt, 'wb') as f:
                pickle.dump(clf, f, -1)
            gc.collect()
            break
        with open(DIR + 'train_cv_tmp.pkl', 'wb') as f:
            pickle.dump(all_pred, f, -1)

        logger.info('trees: {}'.format(list_best_iter))
        # trees = np.mean(list_best_iter, dtype=int)
        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))

        logger.info('param: %s' % (params))
        logger.info('cv: {})'.format(list_score))
        logger.info('cv2: {})'.format(list_score2))

        logger.info('loss: {} (avg min max {})'.format(score[use_score], score))
        logger.info('qwk: {} (avg min max {})'.format(score2[use_score], score2))

        if min_score[use_score] > score[use_score]:
            min_score = score
            min_params = params
        logger.info('best score: {} {}'.format(min_score[use_score], min_score))

        logger.info('best params: {}'.format(min_params))

    imp = pd.DataFrame(clf.feature_importance(), columns=['imp'])
    imp['col'] = usecols
    n_features = imp.shape[0]
    imp = imp.sort_values('imp', ascending=False)
    imp.to_csv(DIR + 'feature_importances_0.csv')
    logger.info('imp use {} {}'.format(imp[imp.imp > 0].shape, n_features))

    del val_x
    del trn_y
    del val_y
    del train_data
    del test_data
    gc.collect()

    trees = np.mean(list_best_iter)

    logger.info('all data size {}'.format(x_train.shape))

    train_data = lgb.Dataset(x_train,
                             label=y_train,
                             feature_name=usecols
                             )
    del x_train
    gc.collect()
    logger.info('train start')
    clf = lgb.train(min_params,
                    train_data,
                    int(trees * 1.1),
                    feval=cst_metric_xgb,
                    # valid_sets=[train_data],
                    verbose_eval=10,
                    callbacks=[callback]
                    )
    logger.info('train end')
    with open(DIR + 'model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    # del x_train
    gc.collect()

    logger.info('save end')
    return best_proba


def predict(best_proba):
    with open(DIR + 'model.pkl', 'rb') as f:
        clf = pickle.load(f)

    with open(DIR + 'usecols.pkl', 'rb') as f:
        usecols = pickle.load(f)

    imp = pd.DataFrame(clf.feature_importance(), columns=['imp'])
    imp['col'] = usecols
    n_features = imp.shape[0]
    imp = imp.sort_values('imp', ascending=False)
    imp.to_csv(DIR + 'feature_importances.csv')
    logger.info('imp use {} {}'.format(imp[imp.imp > 0].shape, n_features))

    df = pd.read_feather('test_0713.ftr', nthreads=8)
    ids = df['Id'].values
    df = pd.concat([df.astype(DTYPE),
                    pd.read_feather('test_hash_cnt.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('test_hash_cnt_sec.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('test_feat_agg.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('test_feat_agg_sec.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('test_hash_cnt_nos38.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('test_date_min.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('test_num_pass_sec.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('test_diff.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('test_time_mean.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('test_time_mean_norm.ftr', nthreads=8).astype(DTYPE),
                    pd.read_feather('test_magic.ftr', nthreads=8)[
        ['magic1', 'magic2', 'magic3', 'magic4']].astype(DTYPE),
    ], axis=1)

    logger.info(f'load 1 {df.shape}')
    logger.info('data size {}'.format(df.shape))

    x_test = df[usecols]
    if x_test.shape[1] != n_features:
        raise Exception('Not match feature num: %s %s' % (x_test.shape[1], n_features))

    logger.info('test load end')

    p_test = clf.predict(x_test)
    with open(DIR + 'test_tmp_pred.pkl', 'wb') as f:
        pickle.dump(p_test, f, -1)

    logger.info('test save end')

    sub = pd.DataFrame()
    sub['Id'] = ids
    sub['Response'] = p_test >= best_proba
    sub.to_csv(DIR + 'submit.csv', index=False)
    logger.info('exit')


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.setLevel('INFO')
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'train.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    best_proba = train()
    predict(best_proba)
