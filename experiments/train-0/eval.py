
import random
import time

from itertools import product
import os
import sys

import numpy as np

sizes = [4, 6, 8, 12, 16, 24, 32, 48, 64]
seeds = [0, 1, 2, 3, 4, 5]

configs = list(product(sizes, seeds)) # 9 x 6 = 54

if __name__ == '__main__':

    task_idx = int(sys.argv[1])
    # time.sleep(5 * task_idx)

    size, seed = configs[task_idx]

    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/class/8.316/final/scripts/train.py \
                                -F /om/user/ericjm/results/class/8.316/final/train-0 \
                                run with width=100 \
                                depth=3 \
                                activation='Tanh' \
                                dropout=0.0 \
                                size={size} \
                                D=90000 \
                                batch_size=4096 \
                                epochs=500 \
                                lr=0.0001 \
                                test_samples=5000 \
                                seed={seed} \
              """)


