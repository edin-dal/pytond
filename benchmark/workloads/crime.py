@pytond()
def crime():
    data_big_cities = data[data["total_population"] > 500000]
    data_big_cities_stats = data_big_cities[['total_population', 'total_adult_population', 'number_of_robberies']].to_numpy()
    predictions = np.einsum('ij,j->i', data_big_cities_stats, np.array([1.0, 2.0, -2000.0]))
    data_big_cities["Crime index"] = predictions / 100000.0
    data_big_cities["Crime index"][data_big_cities["Crime index"] >= 0.02] = 0.02
    data_big_cities["Crime index"][data_big_cities["Crime index"] < 0.01] = 0.01
    return data_big_cities["Crime index"].sum()