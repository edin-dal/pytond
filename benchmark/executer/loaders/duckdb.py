import os
from typing import List, Tuple, Dict

import duckdb
import numpy as np
import pandas as pd

from .abstract import AbstractDataLoader

class DuckdbDataLoader(AbstractDataLoader):
    def __init__(self, threads=1, **kwargs):
        self.conn = duckdb.connect(database=":memory:", read_only=False)
        self.conn.execute(f"PRAGMA threads={threads}")
        os.system("echo off > /sys/devices/system/cpu/smt/control")

    @property
    def db_args(self):
        return self.conn

    def close(self):
        self.conn.close()

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
        for idx in range(len(schema)):
            schema[idx] = (f'"{schema[idx][0]}"', schema[idx][1])
        schema_with_constraints = ', '.join([f"{col_n} {col_t}" for col_n, col_t in schema])
        if primary_key:
            schema_with_constraints += f", PRIMARY KEY ({', '.join(primary_key)})"
        if foreign_keys:
            for src_col, dest in foreign_keys.items():
                dest_tbl, dest_col = dest
                schema_with_constraints += f", FOREIGN KEY ({src_col}) REFERENCES {dest_tbl}({dest_col})"

        self.conn.execute(f"CREATE TABLE {table_name}({schema_with_constraints})")
        self.conn.execute(f"COPY {table_name} FROM '{file_path}' (AUTO_DETECT true{', HEADER' if header else ''})")


    def _dense_matrix_loader(self, table_name: str, file_path: str):

        mat = np.loadtxt(file_path, delimiter="|")
        _, num_cols = mat.shape
        del mat

        schema_with_constraints = "row_no INTEGER PRIMARY KEY, "
        schema_with_constraints += ', '.join([f'col{idx} DOUBLE' for idx in range(num_cols - 1)])

        self.conn.execute(f"CREATE TABLE {table_name}({schema_with_constraints})")
        self.conn.execute(f"COPY {table_name} FROM '{file_path}' (AUTO_DETECT true)")