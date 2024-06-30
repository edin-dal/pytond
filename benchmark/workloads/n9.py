@pytond()
def n9():
    data = trainingdata
    data = data[data.sensor_id == 6698]
    data = data.dropna(subset=["pressure","humidity"])
    data = data.sort_values(by=['timestamp'], ascending=[True])
    data = data.iloc[int(len(data)*0.33):int(len(data)*0.66)]
    return np.array(data[["pressure"]])