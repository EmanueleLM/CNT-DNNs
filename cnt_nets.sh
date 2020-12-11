#!/bin/sh
# Count the number of trained networks inside 'weights/<DATASET>' folder
# arg $1 is the dataset (MNIST, CIFAR10 etc.), arg $2 is the support (0.05, 0.5, 5.0)
# arg $2 is the bins size
# arg $3 is the architecture (fc, cnn, etc.)
# arg $4 is the netsize (small, medium (default), big)
# arg $5 is a further filtering on naming (e.g., on init-distributions)
# Example of usage:  sh ./cnt_nets.sh MNIST 0.05 fc medium normal-gaussian
for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do echo "Accuracy $i: num. nets: "; ls ./weights/$1/ | grep support-$2 | grep binaccuracy-$i | grep $3 | grep $4 | grep $5 | wc -l
done
