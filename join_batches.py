import pandas as pd
import glob, os, sys

# Usage: python join_batches.py /path/to/tank_data.csv
#        Expects cloudscore files to be joined to be in ./data
START_DATE = "2016-01-01"
END_DATE = "2024-05-01"


def main(bbox_filename):
    csv_files = glob.glob(os.path.join("..", "data", "*.csv"))

    tankdf = pd.read_csv(bbox_filename, index_col=0)
    tankdf = tankdf.dropna(axis=0, how="any")

    print(f"Read {tankdf.shape[0]} rows from bounding box data.")

    clouds = pd.concat(map(pd.read_csv, csv_files))
    clouds = clouds.set_index("id")
    print(f"Read {clouds.shape[0]} rows from {len(csv_files)} cloudscore mini-batches.")

    df = pd.DataFrame(index=pd.date_range(START_DATE, END_DATE))

    values = 0
    for index, row in clouds.iterrows():
        cs = row["cs"]
        date = row["date"].split("T")[0]
        df.at[date, index] = cs
        values += 1

    df = df.groupby(pd.Grouper(freq="W-TUE", key=df.index.name)).min()
    print(
        f"After grouping into time periods, {df.isna().sum().sum()} NA values remaining"
    )
    df = df.fillna(0)
    print("Filled with 0s. Current #na: ", df.isna().sum().sum())

    print(f"Saving to csv...")
    df.to_csv("combined.csv")
    print("Done.")


if __name__ == "__main__":
    main(sys.argv[1])
