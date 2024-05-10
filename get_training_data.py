import ee 
import geemap
import numpy as np
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor
import ast

# Function to process each row
def process_row(row):
    coords, roi = row
    cloudscores = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED").filterBounds(roi).filterDate(start_date, end_date)
    cl = cloudscores.toList(cloudscores.size()) # cloudscores.size() specifies how much to grab

    for i in range(cl.size().getInfo()):
        image = ee.Image(cl.get(i))
        t = ee.Date(image.get('system:time_start')).format().getInfo().split('T')[0]  #get rid of the time component
        arr = image.sampleRectangle(roi).get('cs')
        arr = np.array(arr.getInfo())
        avg_cs = np.mean(arr)
        df.at[coords, t] = avg_cs

# get the column as a list of floats we need
data = pd.read_csv("FRT_coords/tank_inventory.csv")
coords = data['Coords']
county = data['County']
state = data['State']

# dates
start_date = "2020-01-01"
end_date = "2020-02-01"
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
date_range = date_range.strftime('%Y-%m-%d')

# create dataframe
df = pd.DataFrame(index=coords, columns=date_range)
df['County'] = county.values 
df['State'] = state.values

print("START")
print(df.shape)

# get working with earth engine
ee.Authenticate()
ee.Initialize(project="cloud-cover-421214")

print("PRE-ARGS")
print(df.shape)

# Create list of arguments for parallel processing
args = [(index, ee.Geometry.Polygon(ast.literal_eval(index))) for index, row in df.iterrows()]

print("NEW")
print(df.shape)
# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor() as executor:
    executor.map(process_row, args)

print("FINAL")
print(df.shape)

df.to_csv('cloud_scores.csv')
