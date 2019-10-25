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


if __name__ == "__main__":
    df = read_file()
    print(get_names(df))
    print(get_west_names())
    print(get_data_by_name(df, get_west_names()))

    name = None
    while name != "exit":
        name = input("Input a name from the list (or type exit): ")

        if name == "exit":
            quit()

        print(get_data_by_name(df, name))

    print("Thanks for playing!")
