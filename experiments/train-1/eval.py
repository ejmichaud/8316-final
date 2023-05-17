
import random
import time

from itertools import product
import os
import sys

import numpy as np

widths = np.power(2, np.linspace(np.log2(5), np.log2(500), 10))
seeds = [0, 1, 2, 3, 4, 5]

configs = list(product(widths, seeds)) # 10 x 6 = 60

if __name__ == '__main__':

    task_idx = int(sys.argv[1])
    # time.sleep(5 * task_idx)

    width, seed = configs[task_idx]
    width = int(width)

    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/class/8.316/final/scripts/train.py \
                                -F /om/user/ericjm/results/class/8.316/final/train-1 \
                                run with width={width} \
                                depth=3 \
                                activation='Tanh' \
                                dropout=0.0 \
                                size=32 \
                                D=90000 \
                                batch_size=4096 \
                                epochs=500 \
                                lr=0.0001 \
                                test_samples=5000 \
                                seed={seed} \
              """)


