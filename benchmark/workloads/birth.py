@pytond(pivot_column_values={'sex': ['F', 'M']})
def birth_analysis():
    all_names = top1000.name.unique()
    mask = np.array(['lesl' in x.lower() for x in all_names])
    lesley_like = all_names[mask]
    filtered = top1000[top1000.name.isin(lesley_like)]
    table = filtered.pivot_table(values='births', index='year', columns='sex', aggfunc=sum)
    table = table.div(table.sum(1), axis=0)
    return table