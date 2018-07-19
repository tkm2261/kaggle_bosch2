import pandas as pd
import numpy as np
from scipy import sparse
import pickle
import glob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
# import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import gc
from logging import getLogger
logger = getLogger(None)

from tqdm import tqdm


import sys
DIR = 'ens_tmp/'  # sys.argv[1]  # 'result_1008_rate001/'
DTYPE = 'float64'
print(DIR)
print(DTYPE)


from utils import mcc_optimize


def cst_metric_xgb(pred, dtrain):
    label = dtrain.get_label().astype(np.int)
    best_proba, best_mcc = mcc_optimize(pred, label)
    return 'mcc', best_mcc, True


def loss_func(y, pred):
    return


def train(x_train):

    #y_train = pd.read_feather('train_0713.ftr')['Response'].values
    #np.savetxt('y_train.npy', y_train)
    y_train = np.loadtxt('y_train.npy')

    usecols = x_train.columns.values.tolist()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=871)

    with open(DIR + 'usecols.pkl', 'wb') as f:
        pickle.dump(usecols, f, -1)
    for _, test in cv.split(x_train, y_train):
        x_train = x_train.iloc[test].values
        y_train = y_train[test]
        np.savetxt('index.npy', test)
        break

    all_params = {'boosting_type': 'gbdt', 'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_bin': 255, 'max_depth': -1, 'metric': 'None', 'min_child_weight': 50,
                  'min_split_gain': 0.01, 'num_leaves': 15, 'objective': 'xentropy ', 'reg_alpha': 0, 'scale_pos_weight': 1, 'seed': 114514, 'subsample': 1, 'subsample_freq': 0, 'verbose': -1}
    """
    all_params = {'min_child_weight': [80],
                  'subsample': [1],
                  'subsample_freq': [0],
                  'seed': [114514],
                  'colsample_bytree': [0.8],
                  'learning_rate': [0.01],
                  'max_depth': [4],
                  'min_split_gain': [0.01],
                  'reg_alpha': [0.001],
                  'reg_lambda': [0.1],
                  'max_bin': [255],
                  'num_leaves': [15],
                  'objective': ['xentropy'],
                  'scale_pos_weight': [1],
                  'verbose': [-1],
                  'boosting_type': ['gbdt'],
                  'metric': ['rmse'],
                  # 'skip_drop': [0.7],
                  }
    """
    all_params = {k: [v] for k, v in all_params.items()}
    use_score = 0
    min_score = (100, 100, 100)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=871)
    for params in tqdm(list(ParameterGrid(all_params))):
        cnt = -1
        list_score = []
        list_score2 = []
        list_best_iter = []
        list_thresh = []
        all_pred = np.zeros(y_train.shape[0])
        for train, test in cv.split(x_train, y_train):
            cnt += 1
            trn_x = x_train[train]
            val_x = x_train[test]
            trn_y = y_train[train]
            val_y = y_train[test]

            train_data = lgb.Dataset(trn_x,
                                     label=trn_y,
                                     feature_name=usecols
                                     )
            test_data = lgb.Dataset(val_x,
                                    label=val_y,
                                    feature_name=usecols
                                    )
            del trn_x
            gc.collect()
            clf = lgb.train(params,
                            train_data,
                            100000,  # params['n_estimators'],
                            early_stopping_rounds=100,
                            valid_sets=[test_data],
                            feval=cst_metric_xgb,
                            # callbacks=[callback],
                            verbose_eval=10
                            )
            pred = clf.predict(val_x).clip(0, 1)

            all_pred[test] = pred

            best_mcc, _score = mcc_optimize(pred, val_y)
            _score2 = _score  # - roc_auc_score(val_y, pred)

            logger.info('   _score: %s' % _score)
            logger.info('   _mcc: %s' % best_mcc)
            logger.info('   _score2: %s' % _score2)
            list_thresh.append(best_mcc)
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
        with open(DIR + 'train_cv_tmp.pkl', 'wb') as f:
            pickle.dump(all_pred, f, -1)

        logger.info('trees: {}'.format(list_best_iter))
        # trees = np.mean(list_best_iter, dtype=int)
        score = (np.mean(list_score), np.min(list_score), np.max(list_score))
        score2 = (np.mean(list_score2), np.min(list_score2), np.max(list_score2))

        logger.info('param: %s' % (params))
        logger.info('cv: {})'.format(list_score))
        logger.info('mcc: {})'.format(list_thresh))
        logger.info('cv2: {})'.format(list_score2))

        logger.info('loss: {} (avg min max {})'.format(score[use_score], score))
        logger.info('all loss: {}'.format(np.sqrt(mean_squared_error(y_train, all_pred))))
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
                    valid_sets=[train_data],
                    verbose_eval=10
                    )
    logger.info('train end')
    with open(DIR + 'model.pkl', 'wb') as f:
        pickle.dump(clf, f, -1)
    with open(DIR + 'list_thresh', 'wb') as f:
        pickle.dump(list_thresh, f, -1)
    # del x_train
    gc.collect()

    logger.info('save end')


def predict(x_test):
    with open(DIR + 'model.pkl', 'rb') as f:
        clf = pickle.load(f)

    with open(DIR + 'usecols.pkl', 'rb') as f:
        usecols = pickle.load(f)

    with open(DIR + 'list_thresh', 'rb') as f:
        list_thresh = pickle.load(f)

    imp = pd.DataFrame(clf.feature_importance(), columns=['imp'])
    imp['col'] = usecols
    n_features = imp.shape[0]
    imp = imp.sort_values('imp', ascending=False)
    imp.to_csv(DIR + 'feature_importances.csv')
    logger.info('imp use {} {}'.format(imp[imp.imp > 0].shape, n_features))

    x_test = x_test[usecols]
    if x_test.shape[1] != n_features:
        raise Exception('Not match feature num: %s %s' % (x_test.shape[1], n_features))

    logger.info('test load end')

    p_test = clf.predict(x_test)
    with open(DIR + 'test_tmp_pred.pkl', 'wb') as f:
        pickle.dump(p_test, f, -1)

    logger.info('test save end')

    logger.info('test load end')

    p_test = clf.predict(x_test)
    with open(DIR + 'test_tmp_pred.pkl', 'wb') as f:
        pickle.dump(p_test, f, -1)

    logger.info('test save end')

    #ids = pd.read_feather('test_0713.ftr')['Id'].values
    ids = np.loadtxt('ids.npy')

    sub = pd.DataFrame()
    sub['Id'] = ids.astype(int)
    sub['Response'] = p_test >= np.mean(list_thresh)
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
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    paths = [
        'result_0718_time_mean',
        'result_0717_diff',
        'result_0717_tune_rate001',
        'result_0716_rate001',
        'result_0716_num_sec',
        'result_0716_sec_hash',
        'result_0716_sec_hash2',
        'result_0716_tuned',
        'result_0715_magic',
        'result_0715_hash',
    ]

    for path in tqdm(paths):
        with open(path + '/train_cv_tmp.pkl', 'rb') as f:
            df_train[path] = pickle.load(f).clip(0, 1)
        with open(path + '/test_tmp_pred.pkl', 'rb') as f:
            df_test[path] = pickle.load(f).clip(0, 1)
    """
    from itertools import combinations
    for p1, p2 in tqdm(list(combinations(paths, 2))):
        with open(p1 + '/train_cv_tmp.pkl', 'rb') as f1:
            with open(p2 + '/train_cv_tmp.pkl', 'rb') as f2:
                df_train[p1 + '-' + p2] = (pickle.load(f1).clip(0, 1) + pickle.load(f2).clip(0, 1)) / 2
        with open(p1 + '/test_tmp_pred.pkl', 'rb') as f1:
            with open(p2 + '/test_tmp_pred.pkl', 'rb') as f2:
                df_test[p1 + '-' + p2] = (pickle.load(f1).clip(0, 1) + pickle.load(f2).clip(0, 1)) / 2
    """
    train(df_train)
    predict(df_test)
