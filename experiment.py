import ee 
import geemap
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import ast


df = pd.read_csv("FRT_coords/tank_inventory.csv")

print(df.shape)

# count frequency and get the most used counties
county_counts = df['County'].value_counts()

# top 20 counties
top_counties = county_counts.head(20).index.tolist()
filtered_df = df[df['County'].isin(top_counties)]
print(df.shape)

# greater than 25 counts
selected_counties = county_counts[county_counts > 25].index.tolist()
filtered_df = df[df['County'].isin(selected_counties)]
print(filtered_df.shape)


print(filtered_df.shape)
# Write the filtered DataFrame to a new CSV file
#filtered_df.to_csv("filtered_tank_inventory.csv", index=False)