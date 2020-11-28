import numpy as np
from scipy import stats

import ComplexNetwork as Complex

def shannon_divergence(d1, d2):
    min_, max_ = np.min(np.concatenate((d1,d2))), np.max(np.concatenate((d1,d2)))
    x = np.arange(min_, max_, abs(max_-min_)/1000)
    P = stats.kde.gaussian_kde(d1)(x)
    Q = stats.kde.gaussian_kde(d2)(x)
    divergence = np.sum(np.where(P != 0., P * np.log(P / Q), 0))
    divergence += np.sum(np.where(Q != 0., Q * np.log(Q / P), 0))
    return divergence
            