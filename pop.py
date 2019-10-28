#! /usr/bin/env python

# Python file for reading and interacting with the topographic data

import pandas as pd
import numpy as np

CITY_DF = None
MAX_CITY_LAT_RADIUS = 25.0
MAX_CITY_LONG_RADIUS = 25.0
MAX_POP = 0.0

def read_cities_csv():
    global CITY_DF
    global MAX_POP
    CITY_DF = pd.read_csv('./data/world_cities.csv', encoding="ISO-8859-1", header=0)
    CITY_DF.drop(axis=1, columns=['city_ascii', 'country', 'iso2', 'iso3', 'admin_name', 'capital', 'id'], inplace=True)
    for index, row in CITY_DF.iterrows():
        if row['population'] > MAX_POP:
            MAX_POP = float(row['population'])
    # print(CITY_DF.head())
    # print(MAX_POP)
    

# TODO: decide if want a gradient or just 0 or 1 as return
def get_if_city(latitude=0, longitude=0) -> float:
    ''' Takes in a latitude/longitude coordinate and provides a population density estimation at that
        point based off distance city center latitude/longitude coordinate.
        The radius of a city is estimated based off of its population.
        1 is max population (city center) and 0 is no city (provides value between 0 and 1) ''' 
    closest = 0.0
    for index, row in CITY_DF.iterrows():
        city_lat_rad = float(row['population']) / MAX_POP * MAX_CITY_LAT_RADIUS
        city_long_rad = float(row['population']) / MAX_POP * MAX_CITY_LONG_RADIUS
        if float(row['lat']) + city_lat_rad >= latitude and float(row['lat']) - city_lat_rad <= latitude and float(row['lng']) + city_long_rad >= longitude and float(row['lng']) - city_long_rad <= longitude:
            lat_dist = abs( float(row['lat']) - latitude )
            long_dist = abs( float(row['lng']) - longitude )
            if lat_dist > city_lat_rad:
                lat_dist = city_lat_rad
            if long_dist > city_long_rad:
                long_dist = city_long_rad
            # print(f'lat and long dist from city center of {row["city"]}: ({lat_dist},{long_dist})')
            lat_density = 1.0 - (lat_dist / city_lat_rad)
            long_density = 1.0 - (long_dist / city_long_rad)
            # print(f'lat and long density estimation: {lat_density}, {long_density}')
            density = np.mean([lat_density, long_density])
            if density > closest:
                closest = density
                # print(f'New closest city is: {row["city"]} with density: {density}')
            # return 1
            if closest == 1.0: # at city center (won't get better -> exit)
                return closest
            # print(' ')
    return closest



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
