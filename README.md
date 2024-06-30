# PyTond: Efficient Python Data Science on the Shoulders of Databases

This is an implementation of **PyTond**, an approach to translate Python data science into efficient SQL code. The work has been presented in the 40th IEEE International Conference on Data Engineering (**ICDE'24**) for the first time. The code here is a experimental prototype that has differences (mostly improvements) over what is presented in the paper.

## Requirements
The implementation is tested with the following software packages/versions:

```Python 3.10.13``` [the exact version is required.]

```pandas 2.2.0```

```numpy 1.25.2```

```duckdb 0.10.1```

```tableauhyperapi 0.0.17782```

```matplotlib 3.8.0```



## Preparing Datasets
To run the experiments, we need to prepare the datasets from different sources. Here is the step-by-step explanation of how to prepare them:

```cd benchmark/data```

```mkdir tpch birth crime synthetic n3 n9```

### tpch dataset
```cd tpch```

```git clone https://github.com/electrum/tpch-dbgen```

```cd tpch-dbgen```

```make```

```./dbgen -s 1```

### birth dataset
```cd birth```

```wget https://www.ssa.gov/oact/babynames/names.zip```

```unzip names.zip```

### crime dataset
```cd crime```

```git clone https://github.com/weld-project/weld-benchmarks```

```cd weld-benchmarks```

```mkdir data```

```export TEST_HOME=$(pwd)```

```./download-data.sh```

Note: If you see "xrange" undefined exception, copy the following snippet to the first line of all scripts in the "scripts" folder:

```
try:
    xrange
except NameError:
    xrange = range
```

### n3 dataset
```cd n3```

Login to your Kaggle account, open the following page (https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018), and download the "2009.csv" file into the current folder.

### n9 dataset
```cd n9```

Login to your Kaggle account, open the following page (https://www.kaggle.com/datasets/hmavrodiev/sofia-air-quality-dataset), and download these files: "2017-07_bme280sof.csv", "2017-08_bme280sof.csv", "2017-09_bme280sof.csv", "2017-10_bme280sof.csv", "2017-11_bme280sof.csv", "2017-12_bme280sof.csv"

### pre-processing of birth, n9, and synthetic datasets

```python3.10 bench_1_preprocess.py```

## Running the Benchmarks

### bench_1

The first benchmark runs the end-to-end experiments of the ICDE paper:

```cd benchmarks```

Set the configuration parameters at the top of "bench1.py" to your based on your needs.

Run the benchmark:

```python3.10 bench_1.py```

Plot the charts:

```python3.10 bench_1_plot.py```

Note: the outputs folder contains the translation pipeline outputs. The results folder contains the execution results and timings. The charts folder contains the generated charts similar to the figures in our ICDE paper.

### bench_2
The second benchmark is related to our layouts micro-benchmark in the ICDE paper.

Run the benchmark:

```python3.10 bench_2.py```

Plot the charts:

```python3.10 bench_2_plot.py```
