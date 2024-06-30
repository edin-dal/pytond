import re
from timeit import default_timer as timer

import pandas as pd

from .abstract import AbstractRunner


class HyperRunner(AbstractRunner):
    def __init__(self, db_args):
        self.server, self.conn = db_args

    def run(self, *args, **kwargs):
        try:
            return super().run(*args, **kwargs)
        except Exception as e:
            print(e)
            raise e
        # finally:
        #     self.conn.close()
        #     self.server.close()

    def query(self, query: str):
        query = self._revise_query(query)
        hyper_res = self.conn.execute_query(query)
        hyper_res.close()

    def query_with_result(self, query: str):
        query = self._revise_query(query)
        hyper_res = self.conn.execute_query(query)
        result_df = pd.DataFrame(list(hyper_res))
        hyper_res.close()
        return result_df

    def query_with_time(self, query: str):
        query = self._revise_query(query)
        timer_start = timer()
        hyper_res = self.conn.execute_query(query)
        timer_end = timer()
        hyper_res.close()
        return float("{:.2f}".format((timer_end - timer_start) * 1000))

    @staticmethod
    def _revise_query(query: str):
        query = re.sub("MEAN\((\w+|\w+\.\w+)\)", r'AVG(\1)', query)
        query = HyperRunner._revise_slice(query)
        return query

    @staticmethod
    def _revise_slice(query: str):
        matches = re.findall("(\w+|\w+\.\w+)\[(\d+):(\d+)\]", query)
        for match in matches:
            var_name, start_idx, end_idx = match[0], int(match[1]), int(match[2])
            query = query.replace(
                f"{var_name}[{start_idx}:{end_idx}]",
                f"substring({var_name} from {start_idx + 1} for {end_idx - start_idx})"
            )
        return query
