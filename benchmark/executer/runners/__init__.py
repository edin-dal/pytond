from importlib import import_module
from benchmark.bench_1_datasets import DBTypes


class Runner:
    def __init__(self, db_type: DBTypes, db_args):
        module = import_module(f"benchmark.executer.runners.{db_type.value}")
        db_class = getattr(module, f'{db_type.value.title().replace(" ", "").replace("_", "")}Runner')
        self.db_obj = db_class(db_args)

    def run(
            self,
            query,
            **kwargs
    ):
        return self.db_obj.run(query, **kwargs)
