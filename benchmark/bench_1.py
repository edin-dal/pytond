import os
import sys
import time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmark.bench_1_datasets import *
from pytond.translator import *
from executer.runners import Runner
from benchmark.executer.loaders import DataLoader
bench_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))

#################################################################################################
#### BENCHMARK CONFIG ###########################################################################
#################################################################################################
# Benchmark Configurations
PYTHON_BENCHMARK = True
PYTOND_BENCHMARK = True
USE_PREGENERATED_QUERIES = False

# Threads to be used
BENCHMARK_THREADS = [1, 2, 3, 4]

# Query engines to be used as PyTond backend
BENCHMARK_DB_TYPES = [DBTypes.Hyper, DBTypes.DuckDB]

# Workloads to be used
BENCHMARK_WORKLOADS = [
    Workloads.TPCH,
    Workloads.Crime,
    Workloads.Birth,
    Workloads.Synthetic,
    Workloads.N3,
    Workloads.N9,
]
#################################################################################################
#################################################################################################
#################################################################################################
RESET_OUTPUTS_AND_RESULTS = True
if RESET_OUTPUTS_AND_RESULTS:
    prompt = input("Do you want to re-generate outputs and results? (y/n): ")
    if prompt == 'y':
        RESET_OUTPUTS_AND_RESULTS = True
    else:
        RESET_OUTPUTS_AND_RESULTS = False
#################################################################################################
#################################################################################################

def iter_python(workload: Workloads, threads: int = 1):
        workload = workload.value[0]
    
        # Preparing Python file
        if not os.path.exists(f'{bench_dir}/outputs/{workload}'):
            os.makedirs(f'{bench_dir}/outputs/{workload}')

        with open(f'{bench_dir}/outputs/{workload}/{workload}_workload.py', 'w') as f:
            with open(f'{bench_dir}/workloads/python_template_workload.py', 'r') as template:
                content = template.read()
                
                content = content.replace('###BENCH_PATH###', "'" + bench_dir + "'")

                content = content.replace('###WORKLOAD###', "'" + workload + "'")

                content = content.replace('###NUM_THREADS###', "'" + str(threads) + "'")
                
                content = content.replace('###DATA_LOADER###', f'load_{workload}()')
                args = ""
                if workload == 'tpch':
                    args = 'li, ord, cu, pa, na, su, ps, re'
                elif workload == 'synthetic':
                    args = 'm45, m55'
                elif workload == 'crime':
                    args = 'data'
                elif workload == 'birth':
                    args = 'top1000'
                elif workload == 'n3':
                    args = 'df'
                elif workload == 'n9':
                    args = 'trainingdata'
                content = content.replace('###DATA_ARGS###', args)

                with open(f'{bench_dir}/workloads/{workload}.py', 'r') as workload_file:
                    workload_code = workload_file.read()
                    funcs = []
                    for line in workload_code.split('\n'):
                        if line.startswith('def '):
                            funcs.append(line.split(' ')[1].split('(')[0])
                    content = content.replace('##FUNC##', workload_code)
                    content = content.replace('###FUNCS_LIST###', str(funcs))

                f.write(content)

        print(f"\033[92m### PYTHON BENCHMARK | Execution... ###\033[0m")

        print(f"\033[92m### {str.upper(workload)} | {threads}T ###\033[0m")
        try:
            os.system(f'python3.10 {bench_dir}/outputs/{workload}/{workload}_workload.py')
        except:
            print(f"\033[91m### {str.upper(workload)} | {threads}T | FAILED ###\033[0m")

#################################################################################################

def iter_main(db_type: DBTypes, workload: Workloads, threads: int = 1):

    print(f"\033[92m### {str.upper(workload.value[0])} | {threads}T ###\033[0m")

    print(f"\033[94m >>> {str.upper(db_type.value)} Loading Data...")  
    data_loader = DataLoader(db_type, threads=threads)
    db_args = data_loader.load(workload, schema=data_context_original)
    print(f"\033[94m >>> Data Loaded.")

    print(f"\033[94m >>> {str.upper(db_type.value)} Execution...")  
    runner = Runner(db_type, db_args)
    sql_files = []
    dir = bench_dir + f'/outputs/{workload.value[0]}/'

    with open(f'{bench_dir}/results/all_results.csv', 'a') as stats_file:

        for file in os.listdir(dir):
            if file.endswith(".sql"):
                sql_files.append(file)

        for file in sorted(sql_files):
            print(f"\033[94m >>>>>> {file} Execution...")  
            with open(dir+file, 'r') as f:
                query = f.read()
                try:
                    stat = runner.run(db_name=db_type.value , query_set_name=workload.value[0], query_name=file[:-4], query=query, verbose=False, results_path=bench_dir+'/results', theads=threads)
                    stats_file.write(f'{workload.value[0]},{file[:-4]},{db_type.value},{threads},{stat}\n')
                    stats_file.flush()
                except(Exception) as e:
                    print(f"\033[91m >>>>>> {file} Execution Failed.")
                    print(e)

        print(f"\033[94m >>> Execution Done.")
    print(f"\033[94m >>> Execution Done.")
    data_loader.close()

################################################

def compare_df(df1, df2):
    if df1.shape != df2.shape:
        return False
    
    if df1.equals(df2):
        return True

    diff = df1.values != df2.values

    for i in range(df1.shape[0]):
        for j in range(df1.shape[1]):
            if diff[i][j] == True:
                if isinstance(df1.iloc[i, j], (float, np.float64)) or isinstance(df2.iloc[i, j], (float, np.float64)):
                    if np.isnan(df1.iloc[i, j]):
                        df1.iloc[i, j] = 0
                    if np.isnan(df2.iloc[i, j]):
                        df2.iloc[i, j] = 0
                    if np.isclose(df1.iloc[i, j], df2.iloc[i, j], atol=1e-4):
                        diff[i][j] = False
    if diff.any():
        return False

    return True

def verify_workload(workload):
    print(f"\033[92m### {str.upper(workload.value[0])} | Verification ###\033[0m")
    # list all files in the directory
    dir = bench_dir + f'/results/{workload.value[0]}/'
    files = {}
    for file in os.listdir(dir):
        if file.endswith(".csv"):
            tokens = ['_opt_hyper', '_opt_duckdb', '_hyper', '_duckdb', '_python']
            for token in tokens:
                if token in file:
                    query_part = file.split(token)[0]
                    break
            if query_part not in files:
                files[query_part] = []
            files[query_part].append(file)

    # sort the files
    for query in files:
        files[query].sort()

    files_keys = sorted(files.keys())

    failed_verifications = []
    for query in files_keys:
        base_data = pd.DataFrame()
        data = pd.DataFrame()
        base_name = query + '_python_1T.csv' 
        if base_name in files[query]:
            print(f"\033[94m>>> {base_name} | Base\033[0m")
            # read the base file
            base_data = pd.read_csv(dir + base_name, delimiter='|', header=None)
                 
            for file in files[query]:
                if file != base_name:
                    verfied = False
                    try:
                        data = pd.read_csv(dir + file, delimiter='|', header=None)
                    except:
                        pass

                    if compare_df(base_data, data):
                        verfied = True

                    if verfied:
                        print(f"\033[92m>>> {file} | Passed\033[0m")
                    else:
                        failed_verifications.append(file)
                        print(f"\033[91m>>> {file} | Failed\033[0m")

                        print("BASE:")
                        print(base_data)

                        print("RESULT:")
                        print(data)

                        if base_data.shape == data.shape:
                            print("DIFF:")
                            print(base_data != data)


        print("--------------------------------------------------")

    if len(failed_verifications) == 0:
        print(f"\033[92m******************************************** All Verifications Passed ###\033[0m")
    else:
        print(f"\033[91m******************************************** Failed Verifications: {failed_verifications} ###\033[0m")

################################################

if __name__ == '__main__':

    if not os.path.exists(f'{bench_dir}/results'):
        os.makedirs(f'{bench_dir}/results')
    else:
        if RESET_OUTPUTS_AND_RESULTS:
            os.system(f'rm -rf {bench_dir}/outputs/*')
            os.system(f'rm -rf {bench_dir}/results/*')

    mode = 'a'
    if RESET_OUTPUTS_AND_RESULTS:
        mode = 'w'
    with open(f'{bench_dir}/results/all_results.csv', mode) as stats_file:
        if mode == 'w':
            stats_file.write('workload,query,engine,thread,time\n')

    ##################################################################################

    # Python Benchmark

    if PYTHON_BENCHMARK:
        print(f"\033[92m### BENCHMARK EXECUTION ###\033[0m")
        for workload in BENCHMARK_WORKLOADS:
            for th in BENCHMARK_THREADS:
                iter_python(workload=workload, threads=th)
                time.sleep(1)
        print(f"\033[92m### BENCHMARK EXECUTION DONE ###\033[0m")
    else:
        print(f"\033[91m### PYTHON BENCHMARK DISABLED ###\033[0m")

    ##################################################################################

    # Pytond Benchmark

    if PYTOND_BENCHMARK:

        if not USE_PREGENERATED_QUERIES:
            # Translation Pipeline
            print(f"\033[92m### CODE GENERATION ###\033[0m")

            if not os.path.exists(f'{bench_dir}/outputs'):
                os.makedirs(f'{bench_dir}/outputs')
            else:
                if RESET_OUTPUTS_AND_RESULTS:
                    os.system(f'rm -rf {bench_dir}/outputs/*')

            for workload in BENCHMARK_WORKLOADS:
                print(f"\033[92m### {str.upper(workload.value[0])} ###\033[0m")
                python_to_sql(bench_dir=bench_dir , source=workload.value[0]+'.py', used_database=workload.value[0], pytond_context=data_context, verbose=False)

            print(f"\033[92m### CODE GENERATION DONE ###\033[0m")
        else:
            print(f"\033[91m### USING CURRENT QUERIES ###\033[0m")

        ##################################################################################

        # Execution Pipeline
        print(f"\033[92m### BENCHMARK EXECUTION ###\033[0m")
        for workload in BENCHMARK_WORKLOADS:
            for db in BENCHMARK_DB_TYPES:
                for th in BENCHMARK_THREADS:
                    iter_main(db_type=db, workload=workload, threads=th)
                    time.sleep(1)
        print(f"\033[92m### BENCHMARK EXECUTION DONE ###\033[0m")

    else:
        print(f"\033[91m### PYTOND BENCHMARK DISABLED ###\033[0m")


    # Verification
    if os.path.exists(f'{bench_dir}/results'):
        for workload in BENCHMARK_WORKLOADS:
            verify_workload(workload)