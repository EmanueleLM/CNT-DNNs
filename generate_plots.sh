#!/bin/sh
# $1: dataset (MNIST, CIFAR10)
# $2: bin size
# $3: architecture
# Example: sh ./generate_plots.sh MNIST 0.1 fc
dataset=$1
bins=$2
architecture=$3

# scale = 0.05
std=0.
maxfiles=1000
echo "Generating plots for networks whose init scale is 0.05"
for i in 0.05
do
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize small --rejectoutliers $std"
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize medium --rejectoutliers $std"
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize big --rejectoutliers $std"
done
sleep 3600  

# scale = 0.5
echo "Generating plots for networks whose init scale is 0.5"
std=3.0
maxfiles=500
for i in 0.5
do
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize small --rejectoutliers $std"
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize medium --rejectoutliers $std"
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize big --rejectoutliers $std"
done
sleep 3600  

# scale = 5.0
echo "Generating plots for networks whose init scale is 5.0"
std=2.5
maxfiles=100
for i in 5.0
do
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize small --rejectoutliers $std"
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize medium --rejectoutliers $std"
    screen -d -m nice bash -c "python3 plot_generator_$architecture.py -maxfiles $maxfiles -scale $i --bins $bins -d $dataset -netsize big --rejectoutliers $std"
done