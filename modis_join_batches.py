import pandas as pd
import glob, os, sys
import warnings
import tqdm

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Usage: python join_batches.py /path/to/tank_data.csv
#        Expects cloudscore files to be joined to be in ../data
START_DATE = "2016-01-01"
END_DATE = "2024-05-01"


def main(bbox_filename):
    csv_files = glob.glob(os.path.join("..", "data", "*.csv"))

    tankdf = pd.read_csv(bbox_filename, index_col=0)
    tankdf = tankdf.dropna(axis=0, how="any")

    print(f"Read {tankdf.shape[0]} rows from bounding box data.")

    clouds = pd.concat(map(pd.read_csv, csv_files))
    clouds = clouds.set_index("id")
    print(f"Read {clouds.shape[0]} rows from {len(csv_files)} mini-batches.")

    df = pd.DataFrame(index=pd.date_range(START_DATE, END_DATE))

    values = 0
    for index, row in tqdm.tqdm(clouds.iterrows(), total=len(clouds.index)):
        cs = row["state_1km"]
        date = row["date"].split("T")[0]
        df.at[date, index] = cs
        values += 1

    #  Assume cloudy (== 1) if no data
    df = df.fillna(1)
    print("Filled with 1s. Current #na: ", df.isna().sum().sum())

    df = df.reindex(sorted(df.columns, key=lambda x: int(x)), axis=1)

    print(f"Saving to csv...")
    df.to_csv("modis_combined.csv")
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1])
