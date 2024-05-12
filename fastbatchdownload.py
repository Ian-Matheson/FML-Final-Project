import pandas as pd
import ee
import sys

""" 
This script begins aysnchronous batch processing jobs on Google Earth
Engine to compute and download the mean CloudScore+ value for 
each bounding box defined in the input file, for each day in the 
observation period. 

See: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED

usage: python fastbatchdownload.py ./path/to/tank_inventory.csv
"""
BATCH_SIZE = 10
START_DATE = "2016-01-01"
# END_DATE = "2020-01-01"
# START_DATE = "2020-01-02"
END_DATE = "2024-05-01"


def process_batch(batch, images):
    imgs = images.filterBounds(batch)

    def per_feature(feat):
        def per_image(img):
            x = img.reduceRegion(
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

    # load cloudscore image data
    start_date = START_DATE
    end_date = END_DATE

    cloudscore = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filterDate(start_date, end_date)
        .select(["cs"])
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
        out = process_batch(batch, cloudscore)
        task = ee.batch.Export.table.toDrive(
            collection=out,
            description=f"smallfastbatch_{i}",
            folder="cloudscore",
            fileFormat="CSV",
            selectors=["id", "cs", "date"],
        )
        task.start()


if __name__ == "__main__":
    main(sys.argv[1])
