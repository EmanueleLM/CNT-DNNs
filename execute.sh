screen -d -m nice bash -c 'python3 mnist.py -a fc -cut-training 1.0 --seed 0 -bins 0.025 --scale 0.05 --sims 10000'
screen -d -m nice bash -c 'python3 mnist.py -a fc -cut-training 1.0 --seed 10001 -bins 0.025 --scale 0.05 --sims 10000'
screen -d -m nice bash -c 'python3 mnist.py -a fc -cut-training 1.0 --seed 20001 -bins 0.025 --scale 0.05 --sims 10000'
screen -d -m nice bash -c 'python3 mnist.py -a fc -cut-training 1.0 --seed 30001 -bins 0.025 --scale 0.05 --sims 10000'
screen -d -m nice bash -c 'python3 mnist.py -a fc -cut-training 1.0 --seed 40001 -bins 0.025 --scale 0.05 --sims 10000'