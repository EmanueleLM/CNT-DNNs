import numpy as np
from scipy import stats

import ComplexNetwork as Complex

def shannon_divergence(d1, d2, distr=False):
    min_, max_ = np.min(np.concatenate((d1,d2))), np.max(np.concatenate((d1,d2)))
    x = np.arange(min_, max_, abs(max_-min_)/1000)
    if distr is False:
        P = stats.kde.gaussian_kde(d1)(x)
        Q = stats.kde.gaussian_kde(d2)(x)
    else:
        assert len(d1) == len(d2), print("When distr is True, the distributions are expected to have the same length, but the first's is {}, the latter's {}.".format(len(d1), len(d2)))
        P, Q = d1, d2  # inputs are already distributions
    divergence = np.sum(np.where(P != 0., P * np.log(P / Q), 0))
    divergence += np.sum(np.where(Q != 0., Q * np.log(Q / P), 0))
    return divergence
            