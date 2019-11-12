#! /usr/bin/env python

# Python file for reading and interacting with the vulture data

import pandas as pd
import os
from return_pixel import return_pixel
import numpy as np


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

    if isinstance(name, str):
        return df[df["individual-local-identifier"] == name]
    elif isinstance(name, list):
        return [get_data_by_name(df, n) for n in name]


def add_pixels(df) -> pd.DataFrame:
    """Uses the return_pixel function to add x and y pixel locations
    to the data frame"""

    if "x" in df and "y" in df:
        return

    df["x"] = df.apply(lambda row: return_pixel(row["location-lat"],
                                                row["location-long"])[0],
                       axis=1)
    df["y"] = df.apply(lambda row: return_pixel(row["location-lat"],
                                                row["location-long"])[1],
                       axis=1)


def interpolate(last_x, last_y, dest_x, dest_y) -> list:
    """Provides a list of x,y pairs in tuple form that bridge from last_x,
    lasy_y and include dest_x, dest_y. The path will move one step at a
    time in the dimension that changes the most, to guarantee continuity
    """

    dx = dest_x - last_x
    dy = dest_y - last_y

    if (dy == 0 and dx == 0):
        return []

    max_val = max(abs(dx), abs(dy))

    x_step = float(dx)/max_val
    y_step = float(dy)/max_val

    return list(
            [(round(last_x + x_step * (i + 1)),
              round(last_y + y_step * (i + 1))) for i in range(max_val)])


def get_coords(df) -> list:
    """Takes a data frame representing the path of a single bird, and
    generates (x, y) coords that represent this bird's path.
    Makes use of the interpolate function to fill in gaps"""

    add_pixels(df)
    last_x = None
    last_y = None

    for row in df.itertuples(index=False):
        x = getattr(row, "x")
        y = getattr(row, "y")
        if last_x is None or last_y is None:
            last_x = x
            last_y = y
            yield (last_x, last_y)
            continue
        # print(f"Interpolating ({last_x}, {last_y})->({x}, {y})")
        for coord in interpolate(last_x, last_y, x, y):
            yield coord
        # print("Finished interpolation")
        last_x = x
        last_y = y


if __name__ == "__main__":
    df = read_file()
    print(get_names(df))
    print(get_west_names())
    print(get_data_by_name(df, get_west_names()))
    my_df = get_data_by_name(df, get_west_names())[0]
    add_pixels(my_df)
    print(my_df)
    last_x = None
    last_y = None
    for pair in list(get_coords(my_df)):
        x = pair[0]
        y = pair[1]
        if last_x is None or last_y is None:
            last_x = x
            last_y = y
            continue
        print(f"({x}, {y}) from ({last_x}, {last_y})")
        assert(abs(x - last_x) <= 1 and abs(y - last_y) <= 1)
        last_x = x
        last_y = y

    name = None
    while name != "exit":
        name = input("Input a name from the list (or type exit): ")

        if name == "exit":
            quit()

        print(get_data_by_name(df, name))

    print("Thanks for playing!")
