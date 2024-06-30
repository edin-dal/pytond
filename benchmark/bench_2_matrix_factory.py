import os
import scipy
import duckdb
import numpy as np
import pandas as pd
from tableauhyperapi import *

class MatrixFactory:
    def __init__(self, threads=1):
        os.system("echo off > /sys/devices/system/cpu/smt/control")
        self.threads = threads
        self.server_hyper = None
        self.conn_hyper = None
        self.conn_duckdb = None

        self._init_hyper()
        self._init_duckdb()

        self.data = {}


    def __del__(self):
        os.system("echo on > /sys/devices/system/cpu/smt/control")
        if self.conn_duckdb:
            self.conn_duckdb.close()
        if self.conn_hyper:
            self.conn_hyper.close()
            self.server_hyper.close()

        
    def _init_hyper(self):
        parameters = {
            "log_config": "",
            "max_query_size": "10000000000",
            "hard_concurrent_query_thread_limit": str(self.threads),
            "initial_compilation_mode": "o"
        }
        self.server_hyper = HyperProcess(Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU, 'User', None, parameters)
        self.conn_hyper = Connection(self.server_hyper.endpoint, f'matfac.hyper', CreateMode.CREATE_AND_REPLACE)


    def _init_duckdb(self):
        self.conn_duckdb = duckdb.connect(database=":memory:", read_only=False)
        self.conn_duckdb.execute(f"PRAGMA threads={self.threads}")


    def create_matrix(self, name, rows_count, cols_count, sparsity):
        if name not in self.data:
            print(f"Generating matrix {name}")
            rand_mat = scipy.sparse.random(rows_count, cols_count, sparsity)
            self.data[name] = {"dense": rand_mat.todense()}
            self.data[name]["coo"] = np.array([rand_mat.row, rand_mat.col, rand_mat.data]).T

            print(f"-- Saving")
            self.save_matrix(name)
            self._populate_hyper(name)
            self._populate_duckdb(name)
            print(f"-- Removing saved file")
            self.remove_saved_matrix(name)
            print(f"Done.")
            return True
        else:
            raise ValueError(f"Matrix with name {name} already exists")


    def delete_matrix(self, name):
        if name in self.data:
            del self.data[name]
            self.conn_hyper.execute_command(f"DROP TABLE {name}_dense")
            self.conn_hyper.execute_command(f"DROP TABLE {name}_coo")
            self.conn_duckdb.execute(f"DROP TABLE {name}_dense")
            self.conn_duckdb.execute(f"DROP TABLE {name}_coo")
        else:
            raise ValueError(f"Matrix with name {name} does not exist")


    def save_matrix(self, name):
        if name in self.data:
            dense_mat = np.zeros((self.data[name]['dense'].shape[0], self.data[name]['dense'].shape[1] + 1))
            dense_mat[:, 0] = list(range(self.data[name]['dense'].shape[0]))
            dense_mat[:, 1:] = self.data[name]['dense']
            dense_df = pd.DataFrame(dense_mat, columns=["id"] + [f'col{i}' for i in range(self.data[name]['dense'].shape[1])])
            dense_df["id"] = dense_df["id"].astype(int)
            dense_df.to_csv(f"{os.path.dirname(__file__)}/data/bench_2/{name}_dense.csv", index=False)

            if self.data[name]['dense'].shape[0] == 1 or self.data[name]['dense'].shape[1] == 1:
                coo_df = pd.DataFrame(self.data[name]['coo'], columns=["rid", "cid", "value"])
                
                if self.data[name]['dense'].shape[0] == 1:
                    coo_df.rename(columns={"cid": "id"}, inplace=True)
                    coo_df = coo_df.drop(columns=["rid"])
                else:
                    coo_df.rename(columns={"rid": "id"}, inplace=True)
                    coo_df.drop(columns=["cid"], inplace=True)
                coo_df["id"] = coo_df["id"].astype(int)
                coo_df.to_csv(f"{os.path.dirname(__file__)}/data/bench_2/{name}_coo.csv", index=False)
            else:
                coo_df = pd.DataFrame(self.data[name]['coo'], columns=["rid", "cid", "value"])
                coo_df["rid"] = coo_df["rid"].astype(int)
                coo_df["cid"] = coo_df["cid"].astype(int)
                coo_df.to_csv(f"{os.path.dirname(__file__)}/data/bench_2/{name}_coo.csv", index=False)
        else:
            raise ValueError(f"Matrix with name {name} does not exist")


    def remove_saved_matrix(self, name):
        if name in self.data:
            os.remove(f"{os.path.dirname(__file__)}/data/bench_2/{name}_dense.csv")
            os.remove(f"{os.path.dirname(__file__)}/data/bench_2/{name}_coo.csv")
        else:
            raise ValueError(f"Matrix with name {name} does not exist")


    def _populate_hyper(self, name):
        print(f"-- Populating Hyper")

        dense_mat = self.data[name]["dense"]
        cols_count = dense_mat.shape[1]
        dense_table_def = "(id INTEGER ASSUMED PRIMARY KEY, " + ", ".join([f"col{i} FLOAT" for i in range(cols_count)]) + ")"
        self.conn_hyper.execute_command(f"CREATE TABLE {name}_dense {dense_table_def}")
        self.conn_hyper.execute_command(
            f"COPY {name}_dense from {escape_string_literal(f'{os.path.dirname(__file__)}/data/bench_2/{name}_dense.csv')} "
            f"with (format => 'csv', delimiter => ',', header => true)"
        )

        coo_table_def = "(rid INTEGER, cid INTEGER, value FLOAT, ASSUMED PRIMARY KEY (rid, cid))"
        self.conn_hyper.execute_command(f"CREATE TABLE {name}_coo {coo_table_def}")
        self.conn_hyper.execute_command(
            f"COPY {name}_coo from {escape_string_literal(f'{os.path.dirname(__file__)}/data/bench_2/{name}_coo.csv')} "
            f"with (format => 'csv', delimiter => ',', header => true)"
        )


    def _populate_duckdb(self, name):
        print(f"-- Populating DuckDB")
        if name not in self.data:
            raise ValueError(f"Matrix with name {name} does not exist")
        
        dense_mat = self.data[name]["dense"]
        cols_count = dense_mat.shape[1]
        dense_table_def = "(id INTEGER PRIMARY KEY, " + ", ".join([f"col{i} DOUBLE" for i in range(cols_count)]) + ")"
        self.conn_duckdb.execute(f"CREATE TABLE {name}_dense {dense_table_def}")
        self.conn_duckdb.execute(f"COPY {name}_dense FROM '{os.path.dirname(__file__)}/data/bench_2/{name}_dense.csv' (AUTO_DETECT true)")

        if dense_mat.shape[0] == 1 or dense_mat.shape[1] == 1:
            coo_table_def = "(id INTEGER, value FLOAT, PRIMARY KEY (id))"
        else:
            coo_table_def = "(rid INTEGER, cid INTEGER, value FLOAT, PRIMARY KEY (rid, cid))"
        self.conn_duckdb.execute(f"CREATE TABLE {name}_coo {coo_table_def}")
        self.conn_duckdb.execute(f"COPY {name}_coo FROM '{os.path.dirname(__file__)}/data/bench_2/{name}_coo.csv' (AUTO_DETECT true)")


    def get_matrix(self, name, format):
        if name in self.data:
            return self.data[name][format]
        else:
            raise ValueError(f"Matrix with name {name} does not exist")


    def get_all_matrix_names(self):
        return list(self.data.keys())
    

    def query_hyper(self, query):
        return pd.DataFrame(list(self.conn_hyper.execute_query(query)))
    

    def query_duckdb(self, query):
        return self.conn_duckdb.execute(query).df()