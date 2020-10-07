# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:49:21 2020

@author: Emanuele

Add 'medium' to the filename of each neural net. 
It has been used to massively renaming files without that prefix, so be careful when using it.
Rename function has been commented out to protect from unintended renamings.
"""

import glob
#import os <-- de-comment

datasets = ['MNIST', 'CIFAR10']
archs = ['fc', 'cnn']
for d in datasets:
    for a in archs:
        files_pattern = "./weights/{}/{}_{}_*.npy".format(d, d, a)
        files = glob.glob(files_pattern)
        prefix = '{}_{}_{}_nlayers-5'.format(d, a, 'medium')
        for file_ in files:
            suffix = file_.split('nlayers-5')[1]
            new_name = "./weights/{}/".format(d) + prefix + suffix
            try:
                print("Renaming:\n{}\nwith:\n{}\n".format(file_, new_name))
                #os.rename(file_, new_name)  <-- de-comment
            except:
                print("An error has occurred")
