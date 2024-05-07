"""
Converts the tank_inventory.json file into a csv, containing only useful information for 
external floating roof tanks only.
"""
import json
import csv

# read in json 
with open("FRT_coords/tank_inventory.json", "r") as file:
    json_data = file.read()
data = json.loads(json_data)

# write data to csv
with open("FRT_coords/tank_inventory.csv", "w", newline='') as csv_file:
    writer = csv.writer(csv_file)

    writer.writerow(['Tanker Type', 'Tile', 'NW Coor (Long, Lat)', 'NE Coor (Long, Lat)', 
                     'SE Coor (Long, Lat)', 'SW Coor (Long, Lat)', 'Capture Date', 'County', 'State'])

    for feature in data['features']:
        properties = feature['properties']
        geometry = feature['geometry']

        tanker_type = properties['object_class']
        if tanker_type == "external_floating_roof_tank":
            tile = properties['tile_name']
            capture_date = properties['capture_date']
            county = properties['county']
            state = properties['state_fips']

            all_coords = geometry['coordinates']
            sw_coor = tuple(all_coords[0][1])
            nw_coor = tuple(all_coords[0][0])
            se_coor = tuple(all_coords[0][2])
            ne_coor = tuple(all_coords[0][3])

            writer.writerow([tanker_type, tile, nw_coor, ne_coor, se_coor, sw_coor, capture_date, 
                             county, state])

