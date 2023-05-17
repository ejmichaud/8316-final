
import random
import time

from itertools import product
import os
import sys

import numpy as np

sizes = [4, 6, 8, 12, 16, 24, 32, 48, 64, 128]
temperatures = np.linspace(1, 3.5, 50)

configs = list(product(sizes, temperatures)) # 10 x 50 = 500

if __name__ == '__main__':

    task_idx = int(sys.argv[1])
    # time.sleep(5 * task_idx)

    size, temperature = configs[task_idx]

    # run a command from the commandline with the os package
    os.system(f"""python /om2/user/ericjm/class/8.316/final/scripts/simulate.py \
                                --save_dir /om/user/ericjm/results/class/8.316/final/all-0 \
                                --temperature {temperature} \
                                --size {size} \
                                --nsim {2000} \
                                """)


