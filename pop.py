#! /usr/bin/env python

# Python file for reading and interacting with the topographic data

import pandas as pd

CITY_DF = None
CITY_LAT_RADIUS = 25.0
CITY_LONG_RADIUS = 25.0
MAX_POP = 0.0

def read_cities_csv():
    global CITY_DF
    global MAX_POP
    CITY_DF = pd.read_csv('./data/world_cities.csv', encoding="ISO-8859-1", header=0)
    CITY_DF.drop(axis=1, columns=['city_ascii', 'country', 'iso2', 'iso3', 'admin_name', 'capital', 'id'], inplace=True)
    for index, row in CITY_DF.iterrows():
        if row['population'] > MAX_POP:
            MAX_POP = row['population']
    print(CITY_DF.head())
    print(MAX_POP)
    

# TODO: decide if want a gradient or just 0 or 1 as return
def get_if_city(latitude=0, longitude=0) -> float:
    for index, row in CITY_DF.iterrows():
        if float(row['lat']) + CITY_LAT_RADIUS >= latitude and float(row['lat']) - CITY_LAT_RADIUS <= latitude and float(row['lng']) + CITY_LONG_RADIUS >= longitude and float(row['lng']) - CITY_LONG_RADIUS <= longitude:
            return 1
    return 0



if __name__ == "__main__":

    read_cities_csv()
    latitude = None
    longitude = None
    while latitude != "exit":
        latitude = input("Input a latitude value (or type exit): ")

        if latitude == "exit":
            quit()

        longitude = input("Input a longitude value (or type exit): ")

        if longitude == "exit":
            quit()

        print( get_if_city( float(latitude), float(longitude) ) )

    print("Thanks for playing!")
