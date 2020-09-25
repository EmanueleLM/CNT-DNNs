# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:15:12 2020

@author: Emanuele
"""
import glob
import numpy as np

def filename_gt(path, prefix='f'):
    files_ = []
    for g in glob.glob(path + prefix + '*.npy'):
        files_.append(g)
    return files_

if __name__ == "__main__":
    topology = 'fc'
    fin = np.zeros(shape=(32,10))
    fin_bias = np.zeros(shape=(10,))
    fin_acc_ge = [str(round(i,3)) for i in  np.arange(0.10, 1.0, 0.025)]
    fin_acc_ge_next = [str(round(i,3)) for i in  np.arange(0.125, 1.025, 0.025)]
    for r, r_next in zip(fin_acc_ge, fin_acc_ge_next):
        acc_prefix = r+'-'+r_next
        for f in filename_gt('results/{}/{}/'.format(topology, acc_prefix), prefix='f'):
            tmp = np.load(f, allow_pickle=True)
            fin += tmp[-2]  # matrix weights
            fin_bias += tmp[-1]  # bias, last layer
        # average the values
        num_weights = len(filename_gt('results/{}/{}/'.format(topology, acc_prefix), prefix='f'))
        print("[logger]: Extracting values of {} weights in range {}-{}, topology {}".format(num_weights, r, r_next, topology))
        fin /= num_weights
        fin_bias /= num_weights
        # save weights
        np.save('./results/{}_weights_npy/fin_weights_acc-{}-{}.npy'.format(topology, r, r_next), fin)
        np.save('./results/{}_weights_npy/fin_bias_acc-{}-{}.npy'.format(topology, r, r_next), fin_bias)
