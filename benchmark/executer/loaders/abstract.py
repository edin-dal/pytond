import os
from abc import abstractmethod
from typing import List, Tuple, Dict
from benchmark.bench_1_datasets import *


class AbstractDataLoader:
    @abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def db_args(self):
        raise NotImplementedError

    def load(self, workload: Workloads, *args, **kwargs):
        data_name, data_path = workload.value
        self.schema = kwargs['schema']['database']

        try:
            data_load_func = getattr(self, data_name)
            data_load_func(data_path=data_path, *args, **kwargs)
            return self.db_args
        except Exception as e:
            self.close()
            raise e
        
    def convert_schema(self, schema_dict, db, table):
        table_name = table
        schema = []
        primary_key = []
        foreign_keys = {}

        table_schema = schema_dict[db][table]
        rel_cols = table_schema.cols
        rel_types = table_schema.types
        rel_pks = table_schema.pks
        rel_fks = table_schema.fks

        for item in zip(rel_cols, rel_types):
            schema.append((item[0], item[1]))

        return table_name, schema, rel_pks, rel_fks

    @abstractmethod
    def close(self):
        raise NotImplementedError

    def tpch(self, data_path: str, **kwargs):
        
        table_name, schema, primary_key, foreign_keys = self.convert_schema(self.schema, 'tpch', 're')
        self._table_loader(table_name, schema, file_path=f"{data_path}region.tbl", primary_key=primary_key, foreign_keys=foreign_keys, delimiter="|")

        table_name, schema, primary_key, foreign_keys = self.convert_schema(self.schema, 'tpch', 'na')
        self._table_loader(table_name, schema, file_path=f"{data_path}nation.tbl", primary_key=primary_key, foreign_keys=foreign_keys, delimiter="|")

        table_name, schema, primary_key, foreign_keys = self.convert_schema(self.schema, 'tpch', 'pa')
        self._table_loader(table_name, schema, file_path=f"{data_path}part.tbl", primary_key=primary_key, foreign_keys=foreign_keys, delimiter="|")

        table_name, schema, primary_key, foreign_keys = self.convert_schema(self.schema, 'tpch', 'su')
        self._table_loader(table_name, schema, file_path=f"{data_path}supplier.tbl", primary_key=primary_key, foreign_keys=foreign_keys, delimiter="|")

        table_name, schema, primary_key, foreign_keys = self.convert_schema(self.schema, 'tpch', 'cu')
        self._table_loader(table_name, schema, file_path=f"{data_path}customer.tbl", primary_key=primary_key, foreign_keys=foreign_keys, delimiter="|")

        table_name, schema, primary_key, foreign_keys = self.convert_schema(self.schema, 'tpch', 'ps')
        self._table_loader(table_name, schema, file_path=f"{data_path}partsupp.tbl", primary_key=primary_key, foreign_keys=foreign_keys, delimiter="|")

        table_name, schema, primary_key, foreign_keys = self.convert_schema(self.schema, 'tpch', 'ord')
        self._table_loader(table_name, schema, file_path=f"{data_path}orders.tbl", primary_key=primary_key, foreign_keys=foreign_keys, delimiter="|")

        table_name, schema, primary_key, foreign_keys = self.convert_schema(self.schema, 'tpch', 'li')
        self._table_loader(table_name, schema, file_path=f"{data_path}lineitem.tbl", primary_key=primary_key, foreign_keys=foreign_keys, delimiter="|")

    def birth(self, data_path: str, **kwargs):
        table_name, schema, primary_key, foreign_keys = self.convert_schema(self.schema, 'birth', 'top1000')
        self._table_loader(table_name=table_name, schema=schema, file_path=f"{data_path}", primary_key=primary_key, foreign_keys=foreign_keys, header=True, delimiter=",")

    def crime(self, data_path: str, **kwargs):
        table_name, schema, primary_key, foreign_keys = self.convert_schema(self.schema, 'crime', 'data')
        self._table_loader(table_name=table_name, schema=schema, file_path=f"{data_path}", primary_key=primary_key, foreign_keys=foreign_keys, header=True)

    def n3(self, data_path: str, **kwargs):

        table_name, schema, primary_key, foreign_keys = self.convert_schema(self.schema, 'n3', 'df')
        self._table_loader(table_name=table_name, schema=schema, file_path=f"{data_path}", header=True, delimiter=",")

    def n9(self, data_path: str, **kwargs):
        table_name, schema, primary_key, foreign_keys = self.convert_schema(self.schema, 'n9', 'trainingdata')
        self._table_loader(table_name=table_name, schema=schema, file_path=f"{data_path}", primary_key=primary_key, foreign_keys=foreign_keys, header=True, delimiter=",")

    def synthetic(self, data_path: str, **kwargs):
        # for filename in os.listdir(data_path):
        for filename in ["R_4_5.csv", "R_5_5.csv"]:
            table_name = filename[:-4].lower()
            self._dense_matrix_loader(
                table_name=table_name,
                file_path=os.path.join(data_path, filename)
            )

    @abstractmethod
    def _table_loader(
            self,
            table_name: str,
            schema: List[Tuple[str, str]],
            file_path: str,
            primary_key: List[str] = None,
            foreign_keys: Dict[str, Tuple[str, str]] = None,
            header: bool = False,
            delimiter: str = "|"
    ):
        raise NotImplementedError

    @abstractmethod
    def _dense_matrix_loader(self, table_name: str, file_path: str):
        raise NotImplementedError