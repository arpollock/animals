#! /usr/bin/env python

# Python file for reading and interacting with the vulture data

import pandas as pd
import os


def read_file(visible_only=False) -> pd.DataFrame:
    """Reads the vulture data from the csv file, and returns a pandas
    dataframe with just the useful information. Specify visible_only to
    restrict data to those entries with a True visible column."""

    # Get current working directory
    cd = os.getcwd()
    # Read in the vulture data
    df = pd.read_csv(cd+'/data/turkey_vultures.csv', low_memory=False)

    usefulDf = df[["event-id", "visible", "timestamp",
                   "location-long", "location-lat",
                   "individual-local-identifier"]].copy()

    if visible_only:
        return usefulDf[usefulDf["visible"] is True]

    return usefulDf


def get_names(dataframe) -> pd.DataFrame:
    """Get all bird names. Takes a data frame"""

    return df["individual-local-identifier"].unique()


def get_data_by_name(df, name) -> pd.DataFrame:
    """Filter df to only location entries for the bird named by name"""

    return df[df["individual-local-identifier"] == name]


if __name__ == "__main__":
    df = read_file()
    print(get_names(df))

    name = None
    while name != "exit":
        name = input("Input a name from the list (or type exit): ")

        if name == "exit":
            quit()

        print(get_data_by_name(df, name))

    print("Thanks for playing!")
