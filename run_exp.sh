#!/bin/sh
# $1: dataset (MNIST, CIFAR10)
# $2: netsize (small, medium, big)
# $3: architecture (fc, cnn, rnn, attention)
# $4: cut-train (from 0.0 to 1.0)
# $5: bins (0.025, usually)
# $6: scale (0.05, 0.5, 5.0)
# $7: sims (1000 usually)
# $8, $9: min-max values of accuracy considered (all the other are discarded)
# $10: GPUs binded to train the networks
# Example sh ./run_exp.sh MNIST medium fc 1.0 0.1 5.00 1000 0.0 1.0 '0,1,2'
dataset=$1
netsize=$2
architecture=$3
cut_train=$4
bins=$5
scale=$6
sims=$7
min=$8
max=$9
gpus=$10
for i in $(seq 1 2 20)
do
    seed=$(( 10000*i ))
    echo "Args: dataset: $1, netsize: $2, arch: $3, cut_train:$4, binc: $5, scale: $6, sims: $7, min: $8, max: $9, gpus: $10"
    echo "Command: python3 train_vision.py -d $dataset --netsize $netsize  -a $architecture --cut-train $cut_train --seed $seed --bins $bins --scale $scale --sims $sims --min $min --max $max --gpus $gpus"
    screen -d -m nice bash -c "python3 train_vision.py -d $dataset --netsize $netsize  -a $architecture --cut-train $cut_train --seed $seed --bins $bins --scale $scale --sims $sims --min $min --max $max --gpus $gpus"
done