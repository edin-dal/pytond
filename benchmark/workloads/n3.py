@pytond()
def n3():
    data = df
    data['OP_CARRIER'] = data['OP_CARRIER'].replace({
        'UA':'United Airlines',
        'AS':'Alaska Airlines',
        '9E':'Endeavor Air',
        'B6':'JetBlue Airways',
        'EV':'ExpressJet',
        'F9':'Frontier Airlines',
        'G4':'Allegiant Air',
        'HA':'Hawaiian Airlines',
        'MQ':'Envoy Air',
        'NK':'Spirit Airlines',
        'OH':'PSA Airlines',
        'OO':'SkyWest Airlines',
        'VX':'Virgin America',
        'WN':'Southwest Airlines',
        'YV':'Mesa Airline',
        'YX':'Republic Airways',
        'AA':'American Airlines',
        'DL':'Delta Airlines'
    })
    data = data[(data['ARR_DELAY'] > 0)]
    data['ARR_DELAY'] = data['ARR_DELAY'] / 60
    data = data.groupby(['OP_CARRIER'], sort=False, as_index=False).agg(value=('ARR_DELAY', 'sum')).sort_values(by=['value'], ascending=[False])
    return data