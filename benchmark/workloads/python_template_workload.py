import os 
import sys
import time
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from benchmark.executer.loaders.python import *

threads = ###NUM_THREADS###
os.system('echo off > /sys/devices/system/cpu/smt/control')
os.environ['OPENBLAS_NUM_THREADS'] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads
os.environ["OMP_NUM_THREADS"] = threads

import numpy as np

# define a handler for @pytond decorator that has some args does nothing
def pytond(*args, **kwargs):
    def decorator(func):
        return func
    return decorator


##FUNC###



if __name__ == '__main__':

    bench_path = ###BENCH_PATH###
    workload = ###WORKLOAD###

    ###DATA_ARGS### = ###DATA_LOADER###

    for func in ###FUNCS_LIST###:

        # Warmup
        for i in range(5):
            print(f"Execution #{i}")
            res = globals()[func]()
            # if i == 0:
            #     print(res)

        # Timed Execution
        times = []
        for i in range(5):
            print(f"Execution #{5+i}")
            start = time.time()
            res = globals()[func]()
            end = time.time()
            times.append(1000 * (end - start))

        mean = np.mean(times).round(0)
        print(f"Avg Time: {mean} ms")

        if not os.path.exists(f'{bench_path}/results/{workload}'):
            os.makedirs(f'{bench_path}/results/{workload}')

        if type(res) in [float, int, np.int64, np.float64]:
            res = np.array([res])
        else:
            res = np.array(res)

        extra_dim_added = False
        if len(res.shape) == 1:
            res = np.array([res])
            extra_dim_added = True

        for i in range(res.shape[0]):
            for j in range(res.shape[1]):
                if isinstance(res[i, j], (float, np.float64)):
                    if res[i, j].is_integer():
                        res[i, j] = int(res[i, j])
                    else:
                        res[i, j] = round(res[i, j], 4)
                elif isinstance(res[i, j], str):
                    res[i, j] = res[i, j].strip()
                elif isinstance(res[i, j], pd.Timestamp):
                    res[i, j] = res[i, j].strftime('%Y-%m-%d')

        if extra_dim_added:
            res = res[0]

        np.savetxt(f'{bench_path}/results/{workload}/{func}_python_{threads}T.csv', res, delimiter='|', fmt='%s')

        with open(f'{bench_path}/results/all_results.csv', 'a') as f:
            f.write(f"{workload},{func},python,{threads},{mean}\n")
        
        print(f"{func} Execution Done.")

