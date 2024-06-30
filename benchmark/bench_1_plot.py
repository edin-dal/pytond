import os
import enum
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

##################################################################################################

class FigsizeConfigs(enum.Enum):
    two_columns = (24, 4)
    one_column = (12, 4)
    half_column = (6, 4)

class ColorConfigs(enum.Enum):
    Python = ("#1f78b4", "#a6cee3")
    Grizzly = ("#33a02c", "#b2df8a")
    PyTond = ("#e31a1c", "#fb9a99")

plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

##################################################################################################

def tpch_end2end(stats):
    plt.rcParams["figure.figsize"] = FigsizeConfigs.two_columns.value

    stats = stats[stats["experiment"] == "tpch"]
    stats = stats[stats["constraints"] == "_"]
    stats = stats[stats["id_generation"] == "_"]
    stats = stats[["threads", "query", "database", "naive", "opt"]]

    for idx, threads in enumerate([1, 4]):
        idf = stats[stats["threads"] == threads]

        idf = pd.DataFrame(
            np.array([
                idf[idf["database"] == "python"]["query"].apply(lambda x: f"Q{x[2]}" if x[1] == "0" else x.upper()),
                idf[idf["database"] == "python"]["naive"],
                idf[idf["database"] == "duckdb"]["naive"],
                idf[idf["database"] == "hyper"]["naive"],
                idf[idf["database"] == "duckdb"]["opt"],
                idf[idf["database"] == "hyper"]["opt"],
            ]).T,
            columns=["query", "Python", "Grizzly/DuckDB", "Grizzly/Hyper", "PyTond/DuckDB", "PyTond/Hyper"]
        )

        colors = [ColorConfigs.Python.value[0], *ColorConfigs.Grizzly.value, *ColorConfigs.PyTond.value]
        ax = idf.plot.bar(logy=True, color=colors, edgecolor="white", lw=0)
        ax.set_ylabel("Run Time (ms)")
        ax.set_xticklabels(idf["query"], rotation=0, fontsize=14)
        ax.legend(loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.17), frameon=False)
        ax.grid(axis="y", linestyle="--")
        ax.set_axisbelow(True)

        plt.savefig(f"charts/tpch_end2end_{threads}.pdf", bbox_inches="tight")

def others_end2end(stats):
    plt.rcParams["figure.figsize"] = FigsizeConfigs.two_columns.value

    stats = stats[stats["query"].isin(["crime_index_100", "birth_analysis", "hybrid_covar_nofilt", "hybrid_covar_filt", "hybrid_mv_nofilt", "hybrid_mv_filt", "n3", "n9"])]
    stats = stats[stats["constraints"] == "_"]
    stats = stats[stats["id_generation"] == "_"]
    stats = stats[["threads", "query", "database", "naive", "opt"]]

    stats = stats.sort_values(by="query").reset_index(drop=True)

    for idx, threads in enumerate([1, 4]):

        idf = pd.DataFrame(
            np.array([
                stats[(stats["threads"] == threads) & (stats["database"] == "python")]["query"],
                stats[(stats["threads"] == threads) & (stats["database"] == "python")]["naive"],
                stats[(stats["threads"] == threads) & (stats["database"] == "duckdb")]["naive"],
                stats[(stats["threads"] == threads) & (stats["database"] == "hyper")]["naive"],
                stats[(stats["threads"] == threads) & (stats["database"] == "duckdb")]["opt"],
                stats[(stats["threads"] == threads) & (stats["database"] == "hyper")]["opt"],
            ]).T,
            columns=["query", "Python", "Grizzly/DuckDB", "Grizzly/Hyper", "PyTond/DuckDB", "PyTond/Hyper"]
        )

        idf['order'] = idf['query'].apply(lambda x:
                                            0 if x == "crime_index_100" else
                                            1 if x == "birth_analysis" else
                                            2 if x == "hybrid_covar_nofilt" else
                                            3 if x == "hybrid_covar_filt" else
                                            4 if x == "hybrid_mv_nofilt" else
                                            5 if x == "hybrid_mv_filt" else
                                            6 if x == "n3" else
                                            7 if x == "n9" else -1)


        idf = idf.sort_values(by='order').reset_index(drop=True)
        idf = idf.drop(columns='order') 

        colors = [ColorConfigs.Python.value[0], *ColorConfigs.Grizzly.value, *ColorConfigs.PyTond.value]
        ax = idf.plot.bar(logy=True, color=colors, edgecolor="white", lw=0)
        ax.set_ylabel("Run Time (ms)")
        ax.set_xticklabels(["Crime Index", "Birth Analysis", "Hybrid Covar (NF)", "Hybrid Covar (F)", "Hybrid MV (NF)", "Hybrid MV (F)", "N3", "N9"], rotation=0, fontsize=14)
        ax.legend(loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.17), frameon=False)
        ax.grid(axis="y", linestyle="--")
        ax.set_axisbelow(True)

        # Show numbers over bars
        horizontal_offset = -0.20
        for i, col in enumerate(idf.columns[2:]):
            for j, val in enumerate(idf[col]):
                x = horizontal_offset + (i+1)*0.09+j
                y = val * 1.18
                if val > 0:
                    scaleup = "{:.2f}".format(idf['Python'][j] / val) + r"$\times$"
                    ax.text(x, y, scaleup, color='black', rotation=90, fontsize=10)

        plt.savefig(f"charts/others_end2end_{threads}.pdf", bbox_inches="tight")

def tpch_scaling(stats):
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 16

    stats = stats[stats["experiment"] == "tpch"]
    stats = stats[stats["constraints"] == "_"]
    stats = stats[stats["id_generation"] == "_"]
    stats = stats[["query", "database", "threads", "opt", "naive"]]

    fig, axes = plt.subplots(nrows=1, ncols=4)

    for q_idx, q_name in enumerate(["q04", "q06", "q13", "q22"]):
        ax = axes[q_idx]
        python_stats = stats[(stats["query"] == q_name) & (stats["database"] == "python")]
        duckdb_stats = stats[(stats["query"] == q_name) & (stats["database"] == "duckdb")]
        hyper_stats = stats[(stats["query"] == q_name) & (stats["database"] == "hyper")]
        idf = pd.DataFrame(
            np.array([
                python_stats[python_stats["threads"] == 1]["naive"].values[0] / python_stats["naive"],
                duckdb_stats[duckdb_stats["threads"] == 1]["naive"].values[0] / duckdb_stats["naive"],
                hyper_stats[hyper_stats["threads"] == 1]["naive"].values[0] / hyper_stats["naive"],
                duckdb_stats[duckdb_stats["threads"] == 1]["opt"].values[0] / duckdb_stats["opt"],
                hyper_stats[hyper_stats["threads"] == 1]["opt"].values[0] / hyper_stats["opt"],
            ]).T,
            columns=["Python", "Grizzly/DuckDB", "Grizzly/Hyper", "PyTond/DuckDB", "PyTond/Hyper"]
        )
        colors = [ColorConfigs.Python.value[0], *ColorConfigs.Grizzly.value, *ColorConfigs.PyTond.value]
        idf.plot(ax=ax, color=colors, style=".-", markersize=14, legend=None)
        ax.plot([0, 1, 2, 3], [1, 2, 3, 4], color="black", linestyle="dashed")

        ax.set_xticks([0, 1, 2, 3], labels=[])
        title = q_name.title()
        if title[0] == "Q" and title[1] == "0":
            title = f"{title[0]}{title[2]}"
        ax.set_title(title, y=-0.35, fontsize=16, fontweight="bold")
        ax.grid(axis="y", linestyle="--")
        ax.set_axisbelow(True)
        ax.set_aspect(1)
    plt.setp(axes, xticklabels=[1, 2, 3, 4])
    plt.setp(axes[0], ylabel='Speedup')
    plt.subplots_adjust(wspace=0.4)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 3, 1, 4, 2]  # Define the desired order
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc="upper center", ncol=3, bbox_to_anchor=(-1.6, 1.5), frameon=False)
    plt.savefig(f"charts/tpch_scaling.pdf", bbox_inches="tight")
    plt.rcParams["font.size"] = 14

def others_scaling(stats):
    plt.rcParams["figure.figsize"] = FigsizeConfigs.two_columns.value

    stats = stats[stats["experiment"] != "tpch"]
    stats = stats[stats["constraints"] == "_"]
    stats = stats[stats["id_generation"] == "_"]
    stats = stats[["query", "database", "threads", "opt", "naive"]]

    fig, axes = plt.subplots(nrows=1, ncols=8)
    for q_idx, q_name in enumerate(["crime_index_100", "birth_analysis", "hybrid_covar_nofilt", "hybrid_covar_filt", "hybrid_mv_nofilt", "hybrid_mv_filt", "n3", "n9"]):
        ax = axes[q_idx]
        python_stats = stats[(stats["query"] == q_name) & (stats["database"] == "python")]
        duckdb_stats = stats[(stats["query"] == q_name) & (stats["database"] == "duckdb")]
        hyper_stats = stats[(stats["query"] == q_name) & (stats["database"] == "hyper")]

        idf = pd.DataFrame(
            np.array([
                python_stats[python_stats["threads"] == 1]["naive"].values[0] / python_stats["naive"],
                duckdb_stats[duckdb_stats["threads"] == 1]["naive"].values[0] / duckdb_stats["naive"],
                hyper_stats[hyper_stats["threads"] == 1]["naive"].values[0] / hyper_stats["naive"],
                duckdb_stats[duckdb_stats["threads"] == 1]["opt"].values[0] / duckdb_stats["opt"],
                hyper_stats[hyper_stats["threads"] == 1]["opt"].values[0] / hyper_stats["opt"],
            ]).T,
            columns=["Python", "Grizzly/DuckDB", "Grizzly/Hyper","PyTond/DuckDB", "PyTond/Hyper"]
        )
        colors = [ColorConfigs.Python.value[0], *ColorConfigs.Grizzly.value, *ColorConfigs.PyTond.value]
        idf.plot(ax=ax, color=colors, style=".-", markersize=14, legend=None)
        ax.plot([0, 1, 2, 3], [1, 2, 3, 4], color="black", linestyle="dashed")
        if q_idx == 3:
            ax.legend(loc="upper center", ncol=8, bbox_to_anchor=(1.2, 1.3), frameon=False)
        ax.set_xticks([0, 1, 2, 3], labels=[])

        title_dict = {
            "crime_index_100": "Crime Index",
            "birth_analysis": "Birth Analysis",
            "hybrid_covar_nofilt": "Hybrid Covar (NF)",
            "hybrid_covar_filt": "Hybrid Covar (F)",
            "hybrid_mv_nofilt": "Hybrid MV (NF)",
            "hybrid_mv_filt": "Hybrid MV (F)",
            "n3": "N3",
            "n9": "N9",
        }
        title = title_dict[q_name]

        ax.set_title(title, y=-0.4, fontsize=14, fontweight="bold")

        ax.grid(axis="y", linestyle="--")
        ax.set_axisbelow(True)

    for ax in axes:
        ax.set_aspect(1)

    plt.setp(axes, xticklabels=[1, 2, 3, 4])
    plt.setp(axes[0], ylabel='Speedup')
    plt.subplots_adjust(wspace=0.5)
    # plt.tight_layout()
    plt.savefig(f"charts/others_scaling.pdf", bbox_inches="tight")

##################################################################################################

if __name__ == '__main__':

    workload_types = {
        'tpch': ['q01', 'q02', 'q03', 'q04', 'q05', 'q06', 'q07', 'q08', 'q09', 'q10', 'q11', 'q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20', 'q21', 'q22'],
        'birth': ['birth_analysis'],
        'crime': ['crime'],
        'n3': ['n3'],
        'n9': ['n9'],
        'synthetic': ['hybrid_mv_f', 'hybrid_mv_nf', 'hybrid_covar_f', 'hybrid_covar_nf']
    }

    tmp_workload_types = {}
    for workload in workload_types:
        tmp_workload_types[workload] = []
        for query in workload_types[workload]:
            tmp_workload_types[workload].append(query)
            tmp_workload_types[workload].append(query + '_opt')
    workload_types = tmp_workload_types

    engine_types = ['python', 'hyper', 'duckdb']

    threads = [1, 2, 3, 4]

    df = pd.read_csv('results/all_results.csv', header=0, index_col=None, sep=',')

    tmp_df = df.copy()[['workload', 'query', 'engine', 'thread']]
    duplicates = tmp_df[tmp_df.duplicated()]

    if duplicates.empty:
        print("\033[92m" + "No duplicates found!" + "\033[0m")
    else:
        print("\033[91m" + "Duplicates found! Remove them and re-run." + "\033[0m")
        print(duplicates)
        exit(1)

    for workload in workload_types:
        for query in workload_types[workload]:
            for engine in engine_types:
                for thread in threads:
                    if df[(df['workload'] == workload) & (df['query'] == query) & (df['engine'] == engine) & (df['thread'] == thread)].empty:
                        if engine == 'python' and query.endswith('_opt'):
                            continue
                        if engine == 'python' and workload == 'tpch' and thread > 1:
                            tmp_df = df[(df['workload'] == workload) & (df['query'] == query) & (df['engine'] == engine) & (df['thread'] == 1)]
                            tmp_df.loc[:, 'thread'] = thread
                            if not tmp_df.empty:
                                df = pd.concat([df, tmp_df], ignore_index=True)
                                continue
                        tmp_df = pd.DataFrame.from_dict({'workload': [workload], 'query': [query], 'engine': [engine], 'thread': [thread], 'time': [float('nan')]})
                        df = pd.concat([df, tmp_df], ignore_index=True)

    df.sort_values(by=['workload', 'query', 'engine', 'thread'], inplace=True)

    df.rename(columns={
        'workload': 'experiment',
        'engine': 'database',
        'thread': 'threads',
    }, inplace=True)

    df['constraints'] = '_'
    df['id_generation'] = '_'
    df['naive'] = float('nan')
    df['opt'] = float('nan')

    df = df[[
        'experiment', 
        'database', 
        'threads', 
        'id_generation', 
        'constraints', 
        'query', 
        'time']]

    df['query'] = df['query'].replace({
        'crime': 'crime_index_100',
        'crime_opt': 'crime_index_100_opt',
        'hybrid_covar_f': 'hybrid_covar_filt',
        'hybrid_covar_f_opt': 'hybrid_covar_filt_opt',
        'hybrid_covar_nf': 'hybrid_covar_nofilt',
        'hybrid_covar_nf_opt': 'hybrid_covar_nofilt_opt',
        'hybrid_mv_f': 'hybrid_mv_filt',
        'hybrid_mv_f_opt': 'hybrid_mv_filt_opt',
        'hybrid_mv_nf': 'hybrid_mv_nofilt',
        'hybrid_mv_nf_opt': 'hybrid_mv_nofilt_opt'
    })

    for index, row in df.iterrows():
        if not row['query'].endswith('_opt'):
            df.loc[(df['experiment'] == row['experiment']) & (df['database'] == row['database']) & (df['threads'] == row['threads']) & (df['query'] == row['query']), 'naive'] = row['time']
        else:
            df.loc[(df['experiment'] == row['experiment']) & (df['database'] == row['database']) & (df['threads'] == row['threads']) & (df['query'] == row['query']), 'opt'] = row['time']

    df.drop(columns=['time'], inplace=True)


    tmp_df = pd.DataFrame(columns=['experiment', 'database', 'threads', 'query', 'constraints', 'id_generation', 'naive', 'opt'])
    for index, row in df.iterrows():
        if not row['query'].endswith('_opt'):
            tmp_df = pd.concat([tmp_df, row.to_frame().T], ignore_index=True)
        else:
            tmp_df.loc[(tmp_df['experiment'] == row['experiment']) & (tmp_df['database'] == row['database']) & (tmp_df['threads'] == row['threads']) & (tmp_df['query'] == row['query'].replace('_opt', '')), 'opt'] = row['opt']

    df = tmp_df

    df.to_csv('results/all_results_prepared.csv', index=False)
    print("\033[92m" + "Prepared results saved to results/all_results_prepared.csv" + "\033[0m")
    print("\033[92m" + "Generating plots..." + "\033[0m")

    tpch_end2end(df)
    print("tpch_end2end done")
    others_end2end(df)
    print("others_end2end done")
    tpch_scaling(df)
    print("tpch_scaling done")
    others_scaling(df)
    print("others_scaling done")


