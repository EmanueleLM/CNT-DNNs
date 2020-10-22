#!/bin/sh
# $1: dataset (MNIST, CIFAR10)
# $2: bin size
# $3: maxfiles
# Example: sh ./generate_plots.sh MNIST 0.1 2500
dataset=$1
bins=$2
maxfiles=$3
for i in 0.05 0.5 5.0
do
    screen -d -m nice bash -c "python3 plot_generator.py -maxfiles $3 -scale $i --bins $bins -d $dataset -netsize small"
    screen -d -m nice bash -c "python3 plot_generator.py -maxfiles $3 -scale $i --bins $bins -d $dataset -netsize medium"
    screen -d -m nice bash -c "python3 plot_generator.py -maxfiles $3 -scale $i --bins $bins -d $dataset -netsize big"
done