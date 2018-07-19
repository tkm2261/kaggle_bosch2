import re
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from multiprocessing import Pool
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold, cross_val_predict
import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, log_loss, mean_squared_error
import gc
from logging import getLogger
logger = getLogger(None)

from tqdm import tqdm


DIR = 'ens_tmp2/'

from numba import jit

#a = set(path.split('/')[0] + '/' for path in glob.glob('result_06*/train_cv_tmp.pkl'))
#b = set(path.split('/')[0] + '/' for path in glob.glob('result_06*/test_tmp_pred.pkl'))


DIRS = [
    'result_0718_time_mean/',
    'result_0717_diff/',
    'result_0717_tune_rate001/',
    'result_0716_rate001/',
    'result_0716_num_sec/',
    'result_0716_sec_hash/',
    'result_0716_sec_hash2/',
    'result_0716_tuned/',
    'result_0715_magic/',
    'result_0715_hash/',
]

index = np.loadtxt('index.npy').astype(int)

from utils import mcc_optimize


def loss_func(y, pred):
    best_proba, best_mcc = mcc_optimize(pred, y)
    return - best_mcc


def load_pred(path):
    with open(path + 'train_cv_tmp.pkl', 'rb') as f:
        pred = pickle.load(f)[index]
    return pred


def load_test(path):
    with open(path + 'test_tmp_pred.pkl', 'rb') as f:
        pred = pickle.load(f)
    return pred


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


def optimize(list_path, score_func, trials):
    space = {}
    for i, path in enumerate(list_path):
        # space[i] = hp.choice(path, [True, False])
        space[i] = hp.quniform(path, 0, 1, 0.01)
    np.random.seed(114514)
    min_params = None
    min_score = 100
    for i in range(10):
        trials = Trials()
        best = fmin(score_func, space, algo=tpe.suggest, trials=trials, max_evals=1000)
        sc = score_func({i: best[path] for i, path in enumerate(list_path)})
        if min_score > sc:
            min_score = sc
            min_params = best
        logger.warn('attempt %s: %s' % (i + 1, sc))
    return min_params


if __name__ == '__main__':

    from logging import StreamHandler, DEBUG, Formatter, FileHandler

    log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

    handler = StreamHandler()
    handler.setLevel('WARN')
    handler.setFormatter(log_fmt)
    logger.setLevel('WARN')
    logger.addHandler(handler)

    handler = FileHandler(DIR + 'ens.py.log', 'a')
    handler.setLevel(DEBUG)
    handler.setFormatter(log_fmt)
    logger.setLevel(DEBUG)
    logger.addHandler(handler)

    list_dir = DIRS  # [DIR1, DIR2, DIR3, DIR4]

    list_preds = [load_pred(path) for path in list_dir]
    y_train = np.loadtxt('y_train.npy')[index]

    for i, path in enumerate(list_dir):
        pp = list_preds[i]
        logger.info('{}'.format((i, path, loss_func(y_train, pp))))

    def score_func(params):
        # use_preds = [pred for i, pred in enumerate(list_preds) if params[i]]
        use_preds = [params[i] * pred for i, pred in enumerate(list_preds)]
        if len(use_preds) == 0:
            return 100

        # pred = np.mean(use_preds, axis=0)
        pred = np.sum(use_preds, axis=0)
        pred /= sum(params.values())

        sc = loss_func(y_train, pred)
        return sc

    def pred_func(params):
        # use_preds = [pred for i, pred in enumerate(list_preds) if params[i]]
        use_preds = [params[i] * pred for i, pred in enumerate(list_preds)]
        if len(use_preds) == 0:
            return 100

        # pred = np.mean(use_preds, axis=0)
        pred = np.sum(use_preds, axis=0)
        pred /= sum(params.values())

        return pred.clip(0, 1)

    def predict(params):
        # use_preds = [pred for i, pred in enumerate(list_preds) if params[i]]
        use_preds = [params[path] * load_test(path) for path in list_dir]
        # pred = np.mean(use_preds, axis=0)
        pred = np.sum(use_preds, axis=0)
        pred /= sum(params.values())
        return pred.clip(0, 1)

    trials = Trials()
    min_params = optimize(list_dir, score_func, trials)
    logger.info(f'min params: {min_params}')
    preds = pred_func({i: min_params[path] for i, path in enumerate(list_dir)})

    best_proba, sc = mcc_optimize(preds, y_train)
    logger.warn('search: %s' % sc)

    list_test = [load_test(path) for path in list_dir]
    p_test = predict(min_params)

    ids = np.loadtxt('ids.npy')

    sub = pd.DataFrame()
    sub['Id'] = ids.astype(int)
    sub['Response'] = p_test >= best_proba
    sub.to_csv(DIR + 'submit_ens.csv', index=False)
    logger.info('exit')
