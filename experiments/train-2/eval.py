
import random
import time

from itertools import product
import os
import sys

import numpy as np

Ds = np.power(2, np.linspace(np.log2(100), np.log2(90000), 15))
seeds = [0, 1, 2, 3, 4, 5]

configs = list(product(Ds, seeds)) # 15 x 6 = 90

if __name__ == '__main__':

    task_idx = int(sys.argv[1])
    # time.sleep(5 * task_idx)

    D, seed = configs[task_idx]
    D = int(D)

    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/class/8.316/final/scripts/train.py \
                                -F /om/user/ericjm/results/class/8.316/final/train-2 \
                                run with width=100 \
                                depth=3 \
                                activation='Tanh' \
                                dropout=0.0 \
                                size=32 \
                                D={D} \
                                batch_size=4096 \
                                epochs=500 \
                                lr=0.0001 \
                                test_samples=5000 \
                                seed={seed} \
              """)


