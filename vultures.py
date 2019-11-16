#! /usr/bin/env python

# Python file for reading and interacting with the vulture data

import pandas as pd
import os
import numpy as np
import model


def read_file(visible_only=False) -> pd.DataFrame:
    """Reads the vulture data from the csv file, and returns a pandas
    dataframe with just the useful information. Specify visible_only to
    restrict data to those entries with a True visible column."""

    # Get current working directory
    cd = os.getcwd()
    # Read in the vulture data
    df = pd.read_csv(cd+'/data/turkey_vultures.csv', low_memory=False)

    usefulDf = df[["visible",  # "timestamp,"
                   "location-long", "location-lat",
                   "individual-local-identifier"]].copy()

    if visible_only:
        return usefulDf[usefulDf["visible"] is True]

    usefulDf = usefulDf.dropna(subset=["location-long",
                                       "location-lat"], axis=0)
    return usefulDf


def read_analysis() -> pd.DataFrame:
    """Read Seth and Zach's analysis spreadsheet in as a pandas dataframe.
    This data can be used to filter the vulture data into more useful sections.
    """

    # Get current working directory
    cd = os.getcwd()
    # Read in the vulture data
    df = pd.read_excel(cd+'/data/Animals.xlsx')

    return df


def get_west_names() -> pd.DataFrame:
    """Gets the names of the birds from the west"""

    df = read_analysis()

    good_enough = df[df["West"] >= 0.2]

    return list(good_enough["Name"])


def get_names(dataframe) -> pd.DataFrame:
    """Get all bird names. Takes a data frame"""

    return dataframe["individual-local-identifier"].unique()


def get_data_by_name(df, name) -> pd.DataFrame:
    """Filter df to only location entries for the bird named by name.
    Returns a df if a string name is passed, or a list of df if a list
    of names is passed. Is recursive in the list case"""

    print(f"Getting data for {name}")

    if isinstance(name, str):
        return df[df["individual-local-identifier"] == name]
    elif isinstance(name, list):
        return [get_data_by_name(df, n) for n in name if (
            df[df["individual-local-identifier"] == n].shape[0] > 1)]


if __name__ == "__main__":
    df = read_file()
    print(get_names(df))
    print(get_west_names())
    print(get_data_by_name(df, get_west_names()))
    my_df = get_data_by_name(df, 'Edgar')
    print(my_df)
    model.add_pixels(my_df)
    print(my_df)
    print(get_data_by_name(df, 'Edgar'))
    last_x = None
    last_y = None
    for pair in list(model.get_coords(my_df)):
        x = pair[0]
        y = pair[1]
        if last_x is None or last_y is None:
            last_x = x
            last_y = y
            continue
        print(f"({x}, {y}) from ({last_x}, {last_y})")
        assert abs(x - last_x) <= 1 and abs(y - last_y) <= 1
        last_x = x
        last_y = y

    name = None
    while name != "exit":
        name = input("Input a name from the list (or type exit): ")

        if name == "exit":
            quit()

        print(get_data_by_name(df, name))

    print("Thanks for playing!")
