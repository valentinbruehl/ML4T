import requests
import json
import pandas as pd

# this piece of code is for request only, and only needs to be executed when new parameters are being incorporated
headers = {'Content-type': 'application/json'}
data = json.dumps({
    "seriesid": ['LNS11300000', 'CUUR0000SA0'],  # Series IDs for Labor Force Participation Rate and CPI (Consumer Price Index)
    "startyear": "2012",
    "endyear": "2024"
})

# API request
p = requests.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', data=data, headers=headers)
json_data = json.loads(p.text)

data_list = []

# Process the JSON data
for series in json_data['Results']['series']:
    seriesId = series['seriesID']
    for item in series['data']:
        year = item['year']
        period = item['period']
        value = item['value']
        if 'M01' <= period <= 'M12':
            # Convert period to a month number as API stores months as MX
            month = int(period[1:])
            # datetime object so that we can later convert it to a dataframe
            date = pd.Timestamp(year=int(year), month=month, day=1)
            
            data_list.append({"Date": date, "Series ID": seriesId, "Value": float(value)})


df = pd.DataFrame(data_list)

# Date as index, so that it is consistent with dataset.py
df.set_index("Date", inplace=True)

