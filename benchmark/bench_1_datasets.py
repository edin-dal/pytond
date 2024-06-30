import os
import copy
import enum
import numpy as np
import pandas as pd
from pytond.tondir import Relation

main_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

class Workloads(enum.Enum):
    TPCH = ("tpch", main_data_path + "/tpch/tpch-dbgen/")
    Birth = ("birth", main_data_path + "/birth/birth_analysis_top1000.csv")
    Crime = ("crime", main_data_path + "/crime/weld-benchmarks/us_cities_states_counties_sf=100.csv")
    Synthetic = ("synthetic", main_data_path + "/synthetic/")
    N3 = ("n3", main_data_path + "/n3/2009.csv")
    N9 = ("n9", main_data_path + "/n9/weather.csv")

data_context = {
    'database':
    {
        'tpch':
        {
            'li': 
            Relation(name='li',
                    cols=['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment', 'empty_col'],
                    types=['INTEGER', 'INTEGER', 'INTEGER', 'INTEGER', 'DOUBLE', 'DOUBLE', 'DOUBLE', 'DOUBLE', 'CHAR(1)', 'CHAR(1)', 'DATE', 'DATE', 'DATE', 'CHAR(25)', 'CHAR(10)', 'VARCHAR(44)', 'INTEGER'],
                    pks = ['l_orderkey', 'l_linenumber'],
                    fks = {'l_orderkey': ('ord', 'o_orderkey'), 'l_partkey': ('pa', 'p_partkey'), 'l_suppkey': ('su', 's_suppkey')},
                    index_col_idx=None
            ),
            'ord':
            Relation(name='ord',
                    cols=['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment', 'empty_col'],
                    types=['INTEGER', 'INTEGER', 'CHAR(1)', 'DOUBLE', 'DATE', 'CHAR(15)', 'CHAR(15)', 'INTEGER', 'VARCHAR(79)', 'INTEGER'],
                    pks = ['o_orderkey'],
                    fks = {'o_custkey': ('cu', 'c_custkey')},
                    index_col_idx=None
            ),
            'pa':
            Relation(name='pa',
                    cols=['p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size', 'p_container', 'p_retailprice', 'p_comment', 'empty_col'],
                    types=['INTEGER', 'VARCHAR(55)', 'CHAR(25)', 'CHAR(10)', 'VARCHAR(25)', 'INTEGER', 'CHAR(10)', 'DOUBLE', 'VARCHAR(23)', 'INTEGER'],
                    pks = ['p_partkey'],
                    fks = None,
                    index_col_idx=None
            ),
            'cu':
            Relation(name='cu',
                    cols=['c_custkey', 'c_name', 'c_address', 'c_nationkey', 'c_phone', 'c_acctbal', 'c_mktsegment', 'c_comment', 'empty_col'],
                    types=['INTEGER', 'VARCHAR(25)', 'VARCHAR(40)', 'INTEGER', 'CHAR(15)', 'DOUBLE', 'CHAR(10)', 'VARCHAR(117)', 'INTEGER'],
                    pks = ['c_custkey'],
                    fks = {'c_nationkey': ('na', 'n_nationkey')},
                    index_col_idx=None
            ),
            'su':
            Relation(name='su',
                    cols=['s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal', 's_comment', 'empty_col'],
                    types=['INTEGER', 'CHAR(25)', 'VARCHAR(40)', 'INTEGER', 'CHAR(15)', 'DOUBLE', 'VARCHAR(101)', 'INTEGER'],
                    pks = ['s_suppkey'],
                    fks = {'s_nationkey': ('na', 'n_nationkey')},
                    index_col_idx=None
            ),
            'ps':
            Relation(name='ps',
                    cols=['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost', 'ps_comment', 'empty_col'],
                    types=['INTEGER', 'INTEGER', 'INTEGER', 'DOUBLE', 'VARCHAR(199)', 'INTEGER'],
                    pks = ['ps_partkey', 'ps_suppkey'],
                    fks = {'ps_partkey': ('pa', 'p_partkey'), 'ps_suppkey': ('su', 's_suppkey')},
                    index_col_idx=None
            ),
            'na':
            Relation(name='na',
                    cols=['n_nationkey', 'n_name', 'n_regionkey', 'n_comment', 'empty_col'],
                    types=['INTEGER', 'CHAR(25)', 'INTEGER', 'VARCHAR(152)', 'INTEGER'],
                    pks = ['n_nationkey'],
                    fks = {'n_regionkey': ('re', 'r_regionkey')},
                    index_col_idx=None
            ),
            're':
            Relation(name='re',
                    cols=['r_regionkey', 'r_name', 'r_comment', 'empty_col'],
                    types=['INTEGER', 'CHAR(25)', 'VARCHAR(152)', 'INTEGER'],
                    pks = ['r_regionkey'],
                    fks = None,
                    index_col_idx=None
            )
        },
        'crime':
        {
            'data':
            Relation(name='data',
                    cols=['state_short', 'total_population', 'total_adult_population', 'number_of_robberies'] + [('col' + str(idx), 'DOUBLE') for idx in range(20)],
                    types=['CHAR(2)', 'INTEGER', 'INTEGER', 'INTEGER'] + ['DOUBLE' for _ in range(20)],
                    index_col_idx=None
            )
        },
        'birth':
        {
            'top1000':
            Relation(name='top1000',
                    cols=['year', 'sex', 'id', 'name', 'births'],
                    types=['INTEGER', 'CHAR(1)', 'INTEGER', 'VARCHAR(255)', 'INTEGER'],
                    index_col_idx=None
            )
        },
        'n3':
        {
            'df':
            Relation(name='df',
                    cols = ['FL_DATE', 'OP_CARRIER', 'OP_CARRIER_FL_NUM', 'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'DEP_TIME', 'DEP_DELAY', 'TAXI_OUT', 'WHEELS_OFF', 'WHEELS_ON', 'TAXI_IN', 'CRS_ARR_TIME', 'ARR_TIME', 'ARR_DELAY', 'CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME', 'DISTANCE', 'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'UNNAMED_27'],
                    types=['DATE', 'CHAR(2)', 'INTEGER', 'CHAR(3)', 'CHAR(3)', 'DOUBLE', 'DOUBLE','DOUBLE', 'DOUBLE', 'DOUBLE', 'DOUBLE','DOUBLE', 'DOUBLE', 'DOUBLE', 'DOUBLE','DOUBLE', 'CHAR(1)', 'DOUBLE', 'DOUBLE','DOUBLE', 'DOUBLE', 'DOUBLE', 'DOUBLE','DOUBLE', 'DOUBLE', 'DOUBLE', 'DOUBLE','DOUBLE'],
                    index_col_idx=None
            )
        },
        'n9':
        {
            'trainingdata':
            Relation(name='trainingdata',
                    cols=['sensor_id', 'location', 'lat', 'lon', 'timestamp', 'pressure', 'temperature', 'humidity'],
                    types=['INTEGER', 'INTEGER', 'DOUBLE', 'DOUBLE', 'CHAR(19)', 'DOUBLE', 'DOUBLE', 'DOUBLE'],
                    index_col_idx=None
            )
        },
        'synthetic':
        {
            'm45':
            Relation(name='R_4_5',
                    cols=['row_no'] + ['col' + str(idx) for idx in range(4 * 2)],
                    types=['DOUBLE' for _ in range(5)],
                    pks=['row_no'],
                    index_col_idx=0
            ),
            'm55':
            Relation(name='R_5_5',
                    cols=['row_no'] + ['col' + str(idx) for idx in range(5 * 2)],
                    types=['DOUBLE' for _ in range(5)],
                    pks=['row_no'],
                    index_col_idx=0
            ),
        }
    }
}

data_context_original = copy.deepcopy(data_context)

class DBTypes(enum.Enum):
    DuckDB = "duckdb"
    Hyper = "hyper"