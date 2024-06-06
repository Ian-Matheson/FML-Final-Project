import pandas as pd
import ee
import sys

""" 
This script begins aysnchronous batch processing jobs on Google Earth
Engine to compute and download the mean MODIS internal cloudmask value for 
each bounding box defined in the input file, for each day in the 
observation period. 

The MODIS cloudmask is a binary mask defined by a flag in bit 10 in the QA band 
'state_1km'. 

See: https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GA#bands

usage: python modis_batch_download.py ./path/to/tank_inventory.csv
"""
BATCH_SIZE = 10
START_DATE = "2016-01-01"
END_DATE = "2024-05-01"

# START_DATE = "2020-01-01"
# END_DATE = "2020-01-02"

BITMASK = 1 << 10


def process_batch(batch, images):
    imgs = images.filterBounds(batch)

    def per_feature(feat):
        def per_image(img):

            #  Extract 10th bit from QA channel- this is the internal MODIS
            #  cloud mask flag. 1 = cloud. 0 = no cloud. Convert to float
            #  Then, shift back over. otherwise the 1 is still in the 10th place,
            #  e.g. cloudy = 1024. We want cloudy = 1
            mask = img.bitwiseAnd(BITMASK).toInt().rightShift(10)

            x = mask.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=feat.geometry(),
                scale=100,
                bestEffort=True,
            )

            x = ee.Feature(None, x)
            x = x.copyProperties(feat)
            x = x.set("date", img.date().format())
            return x

        return imgs.map(per_image)

    return batch.map(per_feature).flatten()


def main(bbox_filename):
    ee.Authenticate()
    ee.Initialize(project="cloud-cover-421214")

    # load modis image data
    start_date = START_DATE
    end_date = END_DATE

    modis = (
        ee.ImageCollection("MODIS/061/MOD09GA")
        .filterDate(start_date, end_date)
        .select("state_1km")  #  Select QA band
    )

    # create bounding boxes
    df = pd.read_csv(bbox_filename)
    df = df.dropna(axis=0, how="any")

    features = [
        ee.Feature(
            ee.Geometry.Rectangle(
                coords=[
                    float(r["NW Coor (Long, Lat)"][1:-1].split(", ")[0]),  # nw long
                    float(r["SE Coor (Long, Lat)"][1:-1].split(", ")[1]),  # se lat
                    float(r["SE Coor (Long, Lat)"][1:-1].split(", ")[0]),  # se long
                    float(r["NW Coor (Long, Lat)"][1:-1].split(", ")[1]),  # nw lat
                ]
            ),
            {"state": r["State"], "county": r["County"], "id": r["ID"]},
        )
        for index, r in df.iterrows()
    ]

    # create batches
    batches = []
    for i in range(0, df.shape[0], BATCH_SIZE):
        batches.append(ee.FeatureCollection(features[i : i + BATCH_SIZE]))

    # submit each batch task to Google Earth Engine cloud
    for i, batch in enumerate(batches):
        out = process_batch(batch, modis)
        task = ee.batch.Export.table.toDrive(
            collection=out,
            description=f"modis_batch_{i}",
            folder="modis",
            fileFormat="CSV",
            selectors=["id", "state_1km", "date"],
        )
        task.start()
        print("started task", i)


if __name__ == "__main__":
    main(sys.argv[1])
