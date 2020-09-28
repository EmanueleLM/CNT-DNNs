#!/bin/sh
# $1: dataset (MNIST, CIFAR10)
# $2: architecture (fc, cnn, rnn, attention)
# $3: cut-train (from 0.0 to 1.0)
# $4: bins (0.025, usually)
# $5: scale (0.05, 0.5, 5.0)
# $6: sims (1000 usually)
# $7, $8: min-max values of accuracy considered (all the other are discarded)
# Example sh ./run_exp.sh MNIST fc 1.0 0.025 0.05 1000 0.5 0.9
for i in {0..9}
do
    let seed="i*1000"
    echo 'python3 train_vision.py -d $1 -a $2 --cut-train $3 --seed $seed --bins $4 --scale $5 --sims $6 --min $7 --max $8'
    screen -d -m nice bash -c 'python3 train_vision.py -d $1 -a $2 --cut-train $3 --seed $seed --bins $4 --scale $5 --sims $6 --min $7 --max $8'
done