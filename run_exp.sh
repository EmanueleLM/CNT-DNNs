#!/bin/sh
# $1: dataset (MNIST, CIFAR10)
# $2: architecture (fc, cnn, rnn, attention)
# $3: cut-train (from 0.0 to 1.0)
# $4: bins (0.025, usually)
# $5: scale (0.05, 0.5, 5.0)
# $6: sims (1000 usually)
# $7, $8: min-max values of accuracy considered (all the other are discarded)
# $9: GPUs binded to train the networks
# Example sh ./run_exp.sh MNIST fc 1.0 0.025 5.00 1000 0.0 1.0 '0,1,2'
dataset=$1
architecture=$2
cut_train=$3
bins=$4
scale=$5
sims=$6
min=$7
max=$8
gpus=$9
for i in $(seq 1 2 20)
do
    seed=$(( 10000*i ))
    echo "Args: dataset: $1, arch: $2, cut_train:$3, binc: $4, scale: $5, sims: $6, min: $7, max: $8, gpus: $9"
    screen -d -m nice bash -c "python3 train_vision.py -d $dataset -a $architecture --cut-train $cut_train --seed $seed --bins $bins --scale $scale --sims $sims --min $min --max $max --gpus $gpus"
done