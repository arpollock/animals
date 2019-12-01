#! /usr/bin/env python

# Python file for reading and interacting with the vulture data

import pandas as pd
import os
import numpy as np
import model


def read_file(file, visible_only=False) -> pd.DataFrame:
    """Reads the vulture data from the csv file, and returns a pandas
    dataframe with just the useful information. Specify visible_only to
    restrict data to those entries with a True visible column."""

    # Get current working directory
    cd = os.getcwd()
    # Read in the vulture data
    df = None
    if file is None:
        df = pd.read_csv(cd+'/data/turkey_vultures.csv', low_memory=False)
    else:
        df = pd.read_csv(cd+'/'+file, low_memory=False)

    usefulDf = df[["visible",  # "timestamp,"
                   "location-long", "location-lat",
                   "individual-local-identifier"]].copy()

    if visible_only:
        return usefulDf[usefulDf["visible"] is True]

    usefulDf = usefulDf.dropna(subset=["location-long",
                                       "location-lat"], axis=0)
    return usefulDf


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
    else:
        return [get_data_by_name(df, n) for n in name if (
            df[df["individual-local-identifier"] == n].shape[0] > 1)]
