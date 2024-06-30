from importlib import import_module
from benchmark.bench_1_datasets import DBTypes, Workloads

class DataLoader:
    def __init__(self, db_type: DBTypes, **kwargs):
        self.db_type = db_type

        module = import_module(f"benchmark.executer.loaders.{self.db_type.value}")
        db_class = getattr(module, f'{self.db_type.value.title().replace(" ", "").replace("_", "")}DataLoader')
        self.db_obj = db_class(**kwargs)

    def load(self, workload: Workloads, **kwargs):
        return self.db_obj.load(workload, **kwargs)

    def close(self):
        self.db_obj.close()
