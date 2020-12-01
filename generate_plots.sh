#!/bin/sh
# $1: dataset (MNIST, CIFAR10)
# $2: bin size
# $3: maxfiles
# $4: architecture
# Example: sh ./generate_plots.sh MNIST 0.1 2500 fc
dataset=$1
bins=$2
maxfiles=$3
architecture=$4
for i in 0.05
do
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize small"
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize medium"
    #screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize big"
done