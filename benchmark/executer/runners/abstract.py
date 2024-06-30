import os
import math
import statistics
import numpy as np
import pandas as pd
import tableauhyperapi
from abc import abstractmethod
from typing import Dict, Tuple


class AbstractRunner:
    @abstractmethod
    def __init__(self, db_args):
        raise NotImplementedError

    def run(
            self,
            query: str,
            warmup_iters: int = 5,
            iters: int = 5,
            verbose: int = 0,
            results_path: str = None,
            db_name: str = None,
            query_name: str = None,
            query_set_name: str = None,
            theads: int = 1
    ):
        if verbose and not results_path:
            raise Exception("you should pass results_path")

        res = self.query_with_result(query)

        set_path_name = results_path + '/' + query_set_name
        if not os.path.exists(set_path_name):
            os.makedirs(set_path_name)

        if res is not None:
            try:
                res.drop(res.columns[0], axis=1, inplace=True)
            except:
                pass

        if type(res) in [float, int, np.float64]:
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
                elif isinstance(res[i, j], tableauhyperapi.date.Date):
                    res[i, j] = str(res[i, j]).split(' ')[0]
                elif isinstance(res[i, j], pd.Timestamp):
                    res[i, j] = res[i, j].strftime('%Y-%m-%d')
 
        if extra_dim_added:
            res = res[0]

        np.savetxt(f'{set_path_name}/{query_name}_{db_name}_{theads}T.csv', res, delimiter='|', fmt='%s')
        
        counter = 0
        for _ in range(warmup_iters):
            print(f"Iteration {counter}")
            self.query(query)
            counter += 1

        times = []
        for _ in range(iters):
            print(f"Iteration {counter}")
            times.append(self.query_with_time(query))
            counter += 1

        stats = self.calculate_stats(times)
        print(stats)

        return stats

    @abstractmethod
    def query(self, query: str):
        raise NotImplementedError

    @abstractmethod
    def query_with_time(self, query: str):
        raise NotImplementedError

    @abstractmethod
    def query_with_result(self, query: str):
        raise NotImplementedError

    @staticmethod
    def calculate_stats(times):
        return round(statistics.mean(times), 2)
