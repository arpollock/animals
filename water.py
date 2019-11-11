#! /usr/bin/env python

# Python file for reading and interacting with the topographic data

import pandas as pd
import numpy as np
import shapefile as shp # pip3 install pyshp
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from return_pixel import return_lat_lon

WATER_DF = None

def read_shapefile():
    """
    Read a shapefile into a Pandas dataframe with a 'coords'
    column holding the geometry information. This uses the pyshp
    package
    """
    shp_path = './data/ne10mlakes/ne_10m_lakes.shp'
    sf = shp.Reader(shp_path,encoding="ISO-8859-1")

    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

def read_water_data():
    global WATER_DF
    WATER_DF = read_shapefile()
    WATER_DF.drop(axis=1, columns=['note', 'delta',	'dam_name', 'year', 'admin', 'name_abb', 'name_alt'], inplace=True)

    WATER_DF["polygon"] = WATER_DF["coords"].apply(lambda coords: Polygon(coords))
    #for index, row in WATER_DF.iterrows():
        #polygon = Polygon(row['coords'])
        # for coord in row['coords']:
        #     print(coord)
        # print(' ')
        # if(index > 5):
        #     return
    # print(WATER_DF.head())

def get_if_water(latitude=0, longitude=0):
    for index, row in WATER_DF.iterrows():
        if row["polygon"].contains(Point(latitude, longitude)):
            return 1
    # TODO: consider ocean
    return 0 # assume if did not encounter in lake data then is land


def get_if_water_xy(x=0, y=0):
    latitude, longitude = return_lat_lon(x, y)
    return get_if_water(latitude, longitude)


read_water_data()  # Always do this, even on import

if __name__ == "__main__":

    latitude = None
    longitude = None
    while latitude != "exit":
        latitude = input("Input a latitude value (or type exit): ")

        if latitude == "exit":
            quit()

        longitude = input("Input a longitude value (or type exit): ")

        if longitude == "exit":
            quit()

        print( get_if_water( float(latitude), float(longitude) ) )

    print("Thanks for playing!")
