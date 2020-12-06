#!/bin/sh
# $1: dataset (MNIST, CIFAR10)
# $2: bin size
# $3: maxfiles
# $4: architecture
# $5: standard deviation to filter data (see plot_generator_fc.py)
# Example: sh ./generate_plots.sh MNIST 0.1 2500 fc 3
dataset=$1
bins=$2
maxfiles=$3
architecture=$4
std=$5
for i in 0.05 0.5 5.0
do
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize small --rejectoutliers $std"
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize medium --rejectoutliers $std"
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize big --rejectoutliers $std"
done