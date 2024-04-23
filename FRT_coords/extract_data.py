"""
Converts the tank_inventory.json file into a csv, containing only useful information for 
external floating roof tanks only.
"""
import json
import csv

# read in json 
with open("tank_inventory.json", "r") as file:
    json_data = file.read()
data = json.loads(json_data)

# write data to csv
with open("tank_inventory.csv", "w", newline='') as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow(['Tanker Type', 'Tile', 'NW Latitude', 'NW Longitude', 'SE Latitude', 'SE Longitude', 
                     'Center Latitude', 'Center Longitude', 'Capture Date', 'County', 'State'])

    for feature in data['features']:
        properties = feature['properties']

        tanker_type = properties['object_class']
        if tanker_type == "external_floating_roof_tank":
            tile = properties['tile_name']
            nw_lat_coor = properties["nw_lat_object_coord"]
            nw_lon_coor = properties["nw_lon_object_coord"]
            se_lat_coor = properties["se_lat_object_coord"]
            se_lon_coor = properties["se_lon_object_coord"]
            center_lat_coor = properties["centroid_lat_object_coord"]
            center_lon_coor = properties["centroid_lon_object_coord"]
            capture_date = properties['capture_date']
            county = properties['county']
            state = properties['state_fips']

            writer.writerow([tanker_type, tile, nw_lat_coor, nw_lon_coor, se_lat_coor, se_lon_coor, 
                            center_lat_coor, center_lon_coor, capture_date, county, state])

