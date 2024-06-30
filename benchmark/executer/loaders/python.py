import pandas as pd
from benchmark.bench_1_datasets import *


pd.options.mode.chained_assignment = None  # default='warn'

def load_tpch():

    path = Workloads.TPCH.value[1]

    # Lineitem

    l_columnnames = ["L_ORDERKEY", "L_PARTKEY", "L_SUPPKEY", "L_LINENUMBER", "L_QUANTITY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_TAX",
                    "L_RETURNFLAG", "L_LINESTATUS", "L_SHIPDATE", "L_COMMITDATE", "L_RECEIPTDATE", "L_SHIPINSTRUCT", "L_SHIPMODE", "L_COMMENT"]

    for i in range(len(l_columnnames)):
        l_columnnames[i] = l_columnnames[i].lower()

    l_data_types = {
        'l_orderkey': int,
        'l_partkey': int,
        'l_suppkey': int,
        'l_linenumber': int,
        'l_quantity': float,
        'l_extendedprice': float,
        'l_discount': float,
        'l_tax': float,
        'l_returnflag': str,
        'l_linestatus': str,
        'l_shipinstruct': str,
        'l_shipmode': str,
        'l_comment': str
    }

    l_parse_dates = ['l_shipdate', 'l_commitdate', 'l_receiptdate']

    li = pd.read_table(path + "lineitem.tbl", sep="|", names=l_columnnames, dtype=l_data_types, parse_dates=l_parse_dates, index_col=False)

    # Order

    o_columnnames = ["O_ORDERKEY", "O_CUSTKEY", "O_ORDERSTATUS", "O_TOTALPRICE", "O_ORDERDATE", "O_ORDERPRIORITY", "O_CLERK", "O_SHIPPRIORITY", "O_COMMENT"]

    for i in range(len(o_columnnames)):
        o_columnnames[i] = o_columnnames[i].lower()
        
    o_data_types = {
        'o_orderkey': int,
        'o_custkey': int,
        'o_orderstatus': str,
        'o_totalprice': float,
        'o_orderpriority': str,
        'o_clerk': str,
        'o_shippriority': int,
        'o_comment': str
    }

    o_parse_dates = ['o_orderdate']

    ord = pd.read_table(path + "orders.tbl", sep="|", names=o_columnnames, dtype=o_data_types, parse_dates=o_parse_dates, index_col=False)

    # Customer

    c_columnnames = ["C_CUSTKEY", "C_NAME", "C_ADDRESS", "C_NATIONKEY", "C_PHONE", "C_ACCTBAL", "C_MKTSEGMENT", "C_COMMENT"]

    for i in range(len(c_columnnames)):
        c_columnnames[i] = c_columnnames[i].lower()
        
    c_data_types = {
        'c_custkey': int,
        'c_name': str,
        'c_address': str,
        'c_nationkey': int,
        'c_phone': str,
        'c_acctbal': float,
        'c_mktsegment': str,
        'c_comment': str
    }

    c_parse_dates = []

    cu = pd.read_table(path + "customer.tbl", sep="|", names=c_columnnames, dtype=c_data_types, parse_dates=c_parse_dates, index_col=False)

    # Part

    p_columnnames = ["P_PARTKEY", "P_NAME", "P_MFGR", "P_BRAND", "P_TYPE", "P_SIZE", "P_CONTAINER", "P_RETAILPRICE", "P_COMMENT"]

    for i in range(len(p_columnnames)):
        p_columnnames[i] = p_columnnames[i].lower()
        
    p_data_types = {
        'p_partkey': int, 
        'p_name': str,
        'p_mfgr': str,
        'p_brand': str,
        'p_type': str,
        'p_size': int,
        'p_container': str,
        'p_retailprice': float,
        'p_comment': str
    }

    p_parse_dates = []

    pa = pd.read_table(path + "part.tbl", sep="|", names=p_columnnames, dtype=p_data_types, parse_dates=p_parse_dates, index_col=False)

    # Nation

    n_columnnames = ["N_NATIONKEY", "N_NAME", "N_REGIONKEY", "N_COMMENT"]

    for i in range(len(n_columnnames)):
        n_columnnames[i] = n_columnnames[i].lower()
        
    n_data_types = {
        'n_nationkey': int,
        'n_name': str,
        'n_regionkey': int,
        'n_comment': str,
    }

    n_parse_dates = []

    na = pd.read_table(path + "nation.tbl", sep="|", names=n_columnnames, dtype=n_data_types, parse_dates=n_parse_dates, index_col=False)

    # Supplier

    s_columnnames = ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_NATIONKEY", "S_PHONE", "S_ACCTBAL", "S_COMMENT"]

    for i in range(len(s_columnnames)):
        s_columnnames[i] = s_columnnames[i].lower()

    s_data_types = {
        's_suppkey': int,
        's_name': str,
        's_address': str,
        's_nationkey': int,
        's_phone': str,
        's_acctbal': float,
        's_comment': str
    }

    s_parse_dates = []

    su = pd.read_table(path + "supplier.tbl", sep="|", names=s_columnnames, dtype=s_data_types, parse_dates=s_parse_dates, index_col=False)

    # Partsupp

    ps_columnnames = ["PS_PARTKEY", "PS_SUPPKEY", "PS_AVAILQTY", "PS_SUPPLYCOST", "PS_COMMENT"]

    for i in range(len(ps_columnnames)):
        ps_columnnames[i] = ps_columnnames[i].lower()

    ps_data_types = {
        'ps_partkey': int,
        'ps_suppkey': int,
        'ps_availqty': int,
        'ps_supplycost': float,
        'ps_comment': str
    }

    ps_parse_dates = []

    ps = pd.read_table(path + "partsupp.tbl", sep="|", names=ps_columnnames, dtype=ps_data_types, parse_dates=ps_parse_dates, index_col=False)

    # Region

    r_columnnames = ["R_REGIONKEY", "R_NAME", "R_COMMENT"]

    for i in range(len(r_columnnames)):
        r_columnnames[i] = r_columnnames[i].lower()

    r_data_types = {
        'r_regionkey': int,
        'r_name': str,
        'r_comment': str
    }

    r_parse_dates = []

    re = pd.read_table(path + "region.tbl", sep="|", names=r_columnnames, dtype=r_data_types, parse_dates=r_parse_dates, index_col=False)

    return li, ord, cu, pa, na, su, ps, re


def load_crime():
    path = Workloads.Crime.value[1]
    crime = pd.read_csv(path, delimiter='|')
    crime.columns = ['state_short', 'total_population', 'total_adult_population', 'number_of_robberies', "col0", "col1",
            "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10", "col11", "col12", "col13",
            "col14", "col15", "col16", "col17", "col18", "col19"]
    return crime


def load_birth():
    path = Workloads.Birth.value[1]
    res = pd.read_csv(path)
    return res


def load_synthetic():
    data_path = Workloads.Synthetic.value[1]
    data_list = []
    for d in ["R_4_5.csv", "R_5_5.csv"]:
        cols_count = 2 * int(d.split("_")[1])
        col_names = ["row_no"] + ["col" + str(i) for i in range(0, cols_count)]
        data_list.append(pd.read_csv(data_path + d, delimiter='|', header=None, names=col_names))
    m45 = data_list[0]
    m55 = data_list[1]
    return m45, m55


def load_n3():
    path = Workloads.N3.value[1]
    return pd.read_csv(path)


def load_n9():
    path = Workloads.N9.value[1]
    return pd.read_csv(path)



