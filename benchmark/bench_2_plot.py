import os
import pandas as pd
import matplotlib.pyplot as plt

# main
def main():
    data = pd.read_csv(f'{os.path.dirname(__file__)}/results/results_bench_2.csv')
    data = data.sort_values(by=['threads', 'exp', 'rows', 'cols', 'sprs']).reset_index(drop=True)
    # Define the data types
    data['threads'] = data['threads'].astype(int)
    data['exp'] = data['exp'].astype(str)
    data['rows'] = data['rows'].astype(int)
    data['cols'] = data['cols'].astype(int)
    data['sprs'] = data['sprs'].astype(float)
    data['time'] = data['time'].astype(float)

    data['time'] = data['time'].apply(lambda x: int(x) + 1)

    views = {}

    # The Sparsity Chart View
    df_spr_covar_1t = data[((data.rows == 1000000) & (data.cols == 32) & (data.threads == 1) & (data.exp.apply(lambda x: str.startswith(x, 'covar'))))].reset_index(drop=True)
    df_spr_covar_4t = data[((data.rows == 1000000) & (data.cols == 32) & (data.threads == 4) & (data.exp.apply(lambda x: str.startswith(x, 'covar'))))].reset_index(drop=True)
    views['Sparsity'] = {
        'Covar': {
            '1 Thread': df_spr_covar_1t,
            '4 Threads': df_spr_covar_4t
        }
    }

    # The Rows Chart View
    df_rows_covar_1t = data[((data.sprs == 1) & (data.cols == 32) & (data.threads == 1) & (data.exp.apply(lambda x: str.startswith(x, 'covar'))))].reset_index(drop=True)
    df_rows_covar_4t = data[((data.sprs == 1) & (data.cols == 32) & (data.threads == 4) & (data.exp.apply(lambda x: str.startswith(x, 'covar'))))].reset_index(drop=True)
    views['Number of Rows'] = {
        'Covar': {
            '1 Thread': df_rows_covar_1t,
            '4 Threads': df_rows_covar_4t
        }
    }

    # The Cols Chart View
    df_cols_covar_1t = data[((data.sprs == 1) & (data.rows == 1000000) & (data.threads == 1) & (data.exp.apply(lambda x: str.startswith(x, 'covar'))))].reset_index(drop=True)
    df_cols_covar_4t = data[((data.sprs == 1) & (data.rows == 1000000) & (data.threads == 4) & (data.exp.apply(lambda x: str.startswith(x, 'covar'))))].reset_index(drop=True)
    views['Number of Columns'] = {
        'Covar': {
            '1 Thread': df_cols_covar_1t,
            '4 Threads': df_cols_covar_4t
        }
    }

    ####################################################################################

    view_t = ['Sparsity', 'Number of Rows', 'Number of Columns']
    exp_t = ['Covar']
    thread_t = ['1 Thread', '4 Threads']
    names_to_df_cols = {
        'Number of Rows': 'rows',
        'Number of Columns': 'cols',
        'Sparsity': 'sprs'
    }
    exp_to_title = {
        'covar-numpy': 'Numpy',
        'covar-duckdb-dense': 'PyTond/DuckDB (Dense)',
        'covar-duckdb-coo': 'PyTond/DuckDB (Sparse)',
        'covar-hyper-dense': 'PyTond/Hyper (Dense)'
    }

    i = 0
    j = 0

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))

    chart_counter = 0
    for i, t in enumerate(thread_t):
        for j, e in enumerate(exp_t):
            for k, v in enumerate(view_t):
                df = views[v][e][t]
                for item in ([axs[i, j*3+k].title, axs[i, j*3+k].xaxis.label, axs[i, j*3+k].yaxis.label] +
                            axs[i, j*3+k].get_xticklabels() + axs[i, j*3+k].get_yticklabels()):
                    item.set_fontname('Times New Roman')
                    item.set_fontweight('bold')
                    item.set_fontsize(14)

                axs[i, j*3+k].xaxis.label.set_fontsize(16)

                axs[i, j*3+k].set_yscale('log')
                axs[i, j*3+k].set_xscale('log')
                axs[i, j*3+k].set_xlabel(v)
                
                if (i,j,k) == (0,0,0) or (i,j,k) == (1,0,0):
                    label = 'Run Time (ms) - '
                    if i == 0:
                        label += '1 Thread'
                    else:
                        label += '4 Threads'
                    axs[i, j*3+k].set_ylabel(label)
                else:
                    axs[i, j*3+k].set_yticklabels([])

                for exp in df.exp.unique():
                    df_exp = df[df.exp == exp].reset_index(drop=True)
                    label = exp_to_title[exp]
                    axs[i, j*3+k].plot(df_exp[names_to_df_cols[v]], df_exp.time, label=label, marker='.', markersize=14, linestyle='-', linewidth=2)

                axs[i, j*3+k].grid(True, linestyle="--")
                axs[i, j*3+k].set_ylim(1, 10000)

                chart_counter += 1

    handles, labels = axs[0,0].get_legend_handles_labels()
    order = [3, 1, 0, 2]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]

    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1), frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, top=0.9)

    plt.savefig(f'{os.path.dirname(__file__)}/charts/bench_2.pdf')

if __name__ == '__main__':
    main()