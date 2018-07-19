import numpy as np
import pandas as pd
from numba.decorators import jit


@jit
def mcc_optimize(y_prob, y_true):

    df = pd.DataFrame()
    df['y_prob'] = y_prob
    df['y_true'] = y_true
    df = df.sort_values('y_prob')
    y_prob_sort = df['y_prob'].values
    y_true_sort = df['y_true'].values
    n = y_true.shape[0]
    nump = y_true.sum()
    numn = n - nump

    tn_v = np.cumsum(y_true_sort == 0, dtype=np.int)
    fp_v = np.cumsum(y_true_sort == 1, dtype=np.int)
    fn_v = numn - tn_v
    tp_v = nump - fp_v
    s = (tp_v + fn_v) / n
    p = (tp_v + fp_v) / n
    sup_v = tp_v / n - s * p
    inf_v = np.sqrt(p * s * (1 - p) * (1 - s))
    mcc_v = sup_v / inf_v
    mcc_v[np.isinf(mcc_v)] = -1
    mcc_v[mcc_v != mcc_v] = -1

    df = pd.DataFrame()
    df['mcc'] = mcc_v
    df['pred'] = y_prob_sort
    df = df.sort_values(by='mcc', ascending=False).reset_index(drop=True)

    best_mcc = df.ix[0, 'mcc']
    best_proba = df.ix[0, 'pred']

    return best_proba, best_mcc


def evalmcc_xgb_min(preds, dtrain):
    labels = dtrain.get_label()
    best_proba, best_mcc = mcc_optimize(preds, labels)
    return 'MCC', - best_mcc
