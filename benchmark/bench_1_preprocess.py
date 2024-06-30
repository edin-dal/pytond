import os
import enum
import numpy as np
import pandas as pd

main_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

class Workloads(enum.Enum):
    Birth = ("birth", main_data_path + "/birth/birth_analysis_top1000.csv")
    Synthetic = ("synthetic", main_data_path + "/synthetic/")
    N9 = ("n9", main_data_path + "/n9/weather.csv")

def prepare_synthetic_data():

    ts = [5]
    fs = [4, 5]
    nr = 1000 * 1000
    ds = 2
    
    def generate_dataset(f, t):
        dr = ds * f
        r = np.random.randint(100, size=(nr, dr))
        idx = np.arange(0, nr)[..., None]
        mat = np.hstack((idx, r))
        np.savetxt(Workloads.Synthetic.value[1] + "R_%d_%d.csv" % (f, t), mat,"%d|" * dr + "%d")
        
    for t in ts:
     for f in fs:
        generate_dataset(f, t)

def prepare_birth_data():
    years = range(1880, 2011)
    names = pd.DataFrame()
    pieces = []
    for year in years:
        path = 'yob%d.txt' % year
        frame = pd.read_csv(main_data_path + '/birth/' + path, names=['name', 'sex', 'births'])
        frame['year'] = year
        pieces.append(frame)
        names = pd.concat(pieces, ignore_index=True)

    def get_top1000(group):
        return group.drop('year', axis=1).drop('sex', axis=1).sort_values(by='births', ascending=False)[:1000]

    grouped = names.groupby(['year', 'sex'])
    top1000 = grouped.apply(get_top1000)

    top1000.to_csv(Workloads.Birth.value[1])
    print('Dataset is prepared and saved to a csv file.')

def prepare_n9_data():
    df = pd.DataFrame()
    for i in ["07", "08", "09", "10", "11", "12"]:
        df_tmp = pd.read_csv(main_data_path + "/n9/" + "2017-" + i + "_bme280sof.csv")
        df_tmp = df_tmp.drop(df_tmp.columns[0], axis=1)
        df = pd.concat([df, df_tmp])
    df.to_csv(Workloads.N9.value[1], index=False)

if __name__ == '__main__':
    if not os.path.exists(Workloads.Synthetic.value[1] + "R_4_5.csv"):
        print("preparing synthetic data...")
        prepare_synthetic_data()
    if not os.path.exists(Workloads.Birth.value[1]):
        print("pre-processing of birth data...")
        prepare_birth_data()
    if not os.path.exists(Workloads.N9.value[1]):
        print("pre-processing of weather data...")
        prepare_n9_data()
    print("All data is prepared and saved to csv files.")