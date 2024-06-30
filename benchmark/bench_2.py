import os
import sys
import time
import statistics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmark.bench_2_matrix_factory import MatrixFactory
from benchmark.bench_1_datasets import *
from pytond.translator import *

################################################################################################

verbose = False
threads = [1, 4]
Warmup = 5
Repetitions = 5

benchmarks = []

# Sparsity Chart (changing sparsity)
b1 = {
    'rows_list': [1000000],
    'cols_list': [32],
    'sprs_list': [0.0001, 0.001, 0.01, 0.1, 1.0]
}

# Sparsity Chart (changing sparsity)
b2 = {
    'rows_list': [10000, 50000, 100000, 500000],
    'cols_list': [32],
    'sprs_list': [1.0]
}

# Features Chart (changing cols)
b3 = {
    'rows_list': [1000000],
    'cols_list': [2, 4, 8, 16],
    'sprs_list': [1.0]
}

benchmarks.append(b1)
benchmarks.append(b2)
benchmarks.append(b3)

bench_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

database_context = {
    'database': 
        {
            'layouts': 
                {
                    'm_dense': Relation(name='m_dense', cols=[], types=[], pks = ['id'], fks = {}, index_col_idx=0),
                    'm_coo': Relation(name='m_coo', cols=[], types=[], pks = ['rid', 'cid'], fks = {}, index_col_idx=None)
                }
        }
    }

################################################################################################

def benchmark(query, query_name):
    print("=============================================")
    print("\033[94m" + ">>> Experiment: " + query_name + "\033[0m")
    print("=============================================")

    for _ in range(Warmup):
        res = query()
        if verbose:
            print("Result: " + str(res))

    times = []
    for i in range(Repetitions):
        start = time.time()
        query()
        end = time.time()
        times.append(1000 * (end - start))

    average = statistics.mean(times)
    stdev = statistics.stdev(times)

    print("Mean: " + str(average) + " ms | Stdev: " + str(stdev) + " ms")

    return res, average, stdev

def coo_df_to_dense_array(df, rows, cols):
    df.drop(df.columns[0], axis=1, inplace=True)
    dense = np.zeros((rows, cols))
    for _, row in df.sort_values(by=[df.columns[0], df.columns[1]]).iterrows():
        dense[int(row.iloc[0])][int(row.iloc[1])] = row.iloc[2]
    return dense

def define_database(cols):
    global database_context
    database_context = {
        'database': 
            {
                'layouts': 
                    {
                        'm_dense': 
                            Relation(
                                name= 'm_dense', 
                                cols= ['id'] + [f'col{i}' for i in range(cols)], 
                                types=[Integer] + [Float for _ in range(cols)],
                                pks = ['id'], 
                                fks = {}, 
                                index_col_idx=0
                            ),
                        'm_coo': 
                            Relation(
                                name='m_coo', 
                                cols=['rid', 'cid', 'value'], 
                                types=[Integer, Integer, Float], 
                                pks = ['rid', 'cid'], 
                                fks = {}, 
                                index_col_idx=None
                            )
                    }
            }
        }

################################################################################################

def generate_queries():
    python_to_sql(
        bench_dir=bench_dir,
        source='bench_2.py',
        used_database='layouts',
        pytond_context=database_context,
        verbose=False
    )

################################################################################################

def covar_numpy(m):
    return np.einsum('ij,ik->jk', m, m)

################################################################################################
base_query_path = bench_dir + "/outputs/bench_2/"
dense_query_path = base_query_path + "covar_dense.sql"
coo_query_path = base_query_path + "covar_coo.sql"

if os.path.exists(dense_query_path):
    os.remove(dense_query_path)
if os.path.exists(coo_query_path):
    os.remove(coo_query_path)


with open(f"{os.path.dirname(__file__)}/results/results_bench_2.csv", "w") as f:
    f.write("threads,exp,rows,cols,sprs,time\n")
    with open(f"{os.path.dirname(__file__)}/results/errors_bench_2.log", "w") as fe:
        
        for NUM_THREADS in threads:

            os.system('echo off > /sys/devices/system/cpu/smt/control')
            os.environ['OPENBLAS_NUM_THREADS'] = str(NUM_THREADS)
            os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(NUM_THREADS)
            os.environ["NUMEXPR_NUM_THREADS"] = str(NUM_THREADS)
            os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)

            import numpy as np

            for b in benchmarks:

                rows_list = b['rows_list']
                cols_list = b['cols_list']
                sprs_list = b['sprs_list']
                total_experiments = len(rows_list) * len(cols_list) * len(sprs_list)
                current_experiment = 0

                for cols in cols_list:
                    
                    define_database(cols)


                    if not os.path.exists(dense_query_path) or not os.path.exists(coo_query_path):
                        generate_queries()
                    dense_query = open(dense_query_path, "r").read()
                    coo_query = open(coo_query_path, "r").read()
                    
                    for rows in rows_list:

                        for sparsity in sprs_list:
                            
                            fac = MatrixFactory(NUM_THREADS)
                            fac.create_matrix("m", rows, cols, sparsity)

                            if verbose:
                                print("Matrix:")
                                print(fac.data["m"]["dense"])

                            print("#########################################################################")
                            print("\033[94m" + ">>> Exp: " + str(current_experiment) + "(" + str(NUM_THREADS) + "T) | Rows=" + str(rows) + " | Cols=" + str(cols) + " | Sparsity=" + str(sparsity) + "\033[0m")

                            #################################################################
                        
                            res_covar_numpy, avg, stdev = benchmark(lambda: covar_numpy(fac.data["m"]["dense"]), "Covar Benchmark (Numpy)")
                            f.write(str(NUM_THREADS) + ",covar-numpy," + str(rows) + "," + str(cols) + "," + str(sparsity) + "," + str(round(avg, 2)) + "\n")
                            f.flush()

                            res_covar_duck_dense, avg, stdev = benchmark(lambda: fac.query_duckdb(dense_query), "Covar Benchmark (DuckDB-Dense)")
                            f.write(str(NUM_THREADS) + ",covar-duckdb-dense," + str(rows) + "," + str(cols) + "," + str(sparsity) + "," + str(round(avg, 2)) + "\n")
                            f.flush()

                            res_covar_hyper_dense, avg, stdev = benchmark(lambda: fac.query_hyper(dense_query), "Covar Benchmark (Hyper-Dense)")
                            f.write(str(NUM_THREADS) + ",covar-hyper-dense," + str(rows) + "," + str(cols) + "," + str(sparsity) + "," + str(round(avg, 2)) + "\n")
                            f.flush()

                            res_covar_duck_coo, avg, stdev = benchmark(lambda: fac.query_duckdb(coo_query), "Covar Benchmark (DuckDB-COO)")
                            f.write(str(NUM_THREADS) + ",covar-duckdb-coo," + str(rows) + "," + str(cols) + "," + str(sparsity) + "," + str(round(avg, 2)) + "\n")
                            f.flush()

                            #################################################################

                            print("Verifying results")

                            res_covar_duck_dense = res_covar_duck_dense.to_numpy()[:, 1:]
                            res_covar_hyper_dense = res_covar_hyper_dense.to_numpy()[:, 1:]
                            res_covar_duck_coo_densed = coo_df_to_dense_array(res_covar_duck_coo, cols, cols)

                            covar_duck_dense_ver = np.allclose(res_covar_numpy, res_covar_duck_dense, rtol=1e-05, atol=1e-06)
                            covar_hyper_dense_ver = np.allclose(res_covar_numpy, res_covar_hyper_dense, rtol=1e-05, atol=1e-06)
                            covar_duck_coo_ver = np.allclose(res_covar_numpy, res_covar_duck_coo_densed, rtol=1e-05, atol=1e-06)

                            if not covar_duck_dense_ver:
                                fe.write(f"covar-duckdb-dense,{rows},{cols},{sparsity}\n")
                            if not covar_hyper_dense_ver:
                                fe.write(f"covar-hyper-dense,{rows},{cols},{sparsity}\n")
                            if not covar_duck_coo_ver:
                                fe.write(f"covar-duckdb-coo,{rows},{cols},{sparsity}\n")

                            fe.flush()

                            if (covar_duck_dense_ver and  covar_hyper_dense_ver and covar_duck_coo_ver):
                                print("\033[92m" + "ALL VERIFIED :)" + "\033[0m")
                            else:
                                print("\033[91m" + f"THERE WERE ERRORS :( Check errors_{NUM_THREADS}T.log" + "\033[0m")
                            #################################################################
                            
                            fac.delete_matrix("m")
                            del fac

                            current_experiment += 1

                    os.remove(dense_query_path)
                    os.remove(coo_query_path)


print("All done.")
with open(f"{os.path.dirname(__file__)}/results/errors_bench_2.log", "r") as f:
    if f.read() == "":
        print("\033[92m" + "ALL VERIFIED :)" + "\033[0m")
    else:
        print("\033[91m" + f"THERE WERE ERRORS :( Check results/errors_bench_2.log" + "\033[0m")
        print(f.read())