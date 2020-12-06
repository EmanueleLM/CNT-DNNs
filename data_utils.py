import numpy as np

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
