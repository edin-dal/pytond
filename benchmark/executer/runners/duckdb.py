from timeit import default_timer as timer

from .abstract import AbstractRunner


class DuckdbRunner(AbstractRunner):
    def __init__(self, db_args):
        self.conn = db_args

    def query(self, query: str):
        self.conn.execute(query)

    def query_with_result(self, query: str):
        return self.conn.execute(query).df()

    def query_with_time(self, query: str):
        timer_start = timer()
        self.conn.execute(query)
        timer_end = timer()
        return float("{:.2f}".format((timer_end - timer_start) * 1000))
