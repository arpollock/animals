#! /usr/bin/env python

# Python file for reading and interacting with the population/ city data

import pandas as pd
import numpy as np
import top
import cv2
from scipy.ndimage import gaussian_filter


CITY_DF = None
MAX_CITY_LAT_RADIUS = 25.0
MAX_CITY_LONG_RADIUS = 25.0
MAX_POP = 0.0


def read_cities_csv():
    global CITY_DF
    global MAX_POP
    top.read_image()
    CITY_DF = pd.read_csv('./data/world_cities.csv', encoding="ISO-8859-1",
                          header=0)
    CITY_DF.drop(axis=1, columns=['city_ascii', 'country', 'iso2', 'iso3',
                                  'admin_name', 'capital', 'id'], inplace=True)
    for index, row in CITY_DF.iterrows():
        if row['population'] > MAX_POP:
            MAX_POP = float(row['population'])
    x = []
    y = []
    for index, row in CITY_DF.iterrows():
        city_x, city_y = top.return_pixel(row['lat'], row['lng'])
        x.append(city_x)
        y.append(city_y)
    CITY_DF['x'] = x
    CITY_DF['y'] = y
    CITY_DF = CITY_DF[np.isfinite(CITY_DF['population'])]
    # print(CITY_DF.head())
    # print(MAX_POP)


# TODO: decide if want a gradient or just 0 or 1 as return
def get_city_scores(width, height):
    '''Takes in (x, y) coords and convert each city to pixel coords and add
       the population divided by distance if it is within the thresh'''
    ''' Takes in a latitude/longitude coordinate and provides a population
        density estimation at that
        point based off distance city center latitude/longitude coordinate.
        The radius of a city is estimated based off of its population.
        1 is max population (city center) and 0 is no city (provides value
        between 0 and 1) '''
    city_scores = np.zeros((width, height), dtype=np.float64)
    impact_size = 101
    radius = int(impact_size / 2)
    norm = np.random.normal(0, 0.1, impact_size)
    norm = np.sort(norm)
    # print(norm)
    norm = np.absolute(norm)
    norm = np.negative(norm)
    norm = np.add(norm, -np.min(norm))

    cross_norm = np.outer(norm, norm)
    padding = 100
    side_padding = int(padding / 2)
    black = np.zeros((impact_size + padding, impact_size + padding))
    black[side_padding:side_padding+impact_size, side_padding:side_padding+impact_size] = cross_norm
    cross_norm = black
    cross_norm = gaussian_filter(cross_norm, 25)
    cross_norm_show = np.multiply(np.divide(cross_norm, np.max(cross_norm)), 255).astype(np.uint8)
    # print(cross_norm_show)
    # main_window = cv2.namedWindow('heatmap', cv2.WINDOW_NORMAL)
    # cv2.imshow('heatmap', cross_norm_show)
    # cv2.resizeWindow('heatmap', 1280, 800)
    # while cv2.waitKey(33) != 27:
    #     pass
    for index, row in CITY_DF.iterrows():
        population = row['population']
        x = row['x']
        y = row['y']
        population_norm = np.multiply(cross_norm, population)
        x_left_bound = min(radius + side_padding, x - 1)
        x_right_bound = min(radius + side_padding + 1, width - x - 1)
        y_up_bound = min(radius + side_padding, y - 1)
        y_down_bound = min(radius + side_padding + 1, height - y - 1)
        city_scores[x-x_left_bound:x+x_right_bound, y-y_up_bound:y+y_down_bound] += population_norm[(radius+side_padding)-x_left_bound:(radius+side_padding)+x_right_bound, (radius+side_padding)-y_up_bound:(radius+side_padding)+y_down_bound]
    return city_scores


if __name__ == "__main__":
    read_cities_csv()
    width = 1928
    height = 1378
    coast_points = top.get_coast_points()
    city_scores = get_city_scores(width, height).astype(int)
    np.save("population.npy", np.transpose(city_scores))
    print(np.max(city_scores), np.min(city_scores))
    heatmap = np.divide(city_scores, np.max(city_scores))
    heatmap = np.multiply(heatmap, 255)
    heatmap = np.transpose(heatmap).astype(np.uint8)

    main_window = cv2.namedWindow('heatmap', cv2.WINDOW_NORMAL)
    for coast in coast_points:
        heatmap[coast[1]][coast[0]] = 255
    cv2.imshow('heatmap', heatmap)
    cv2.resizeWindow('heatmap', 1280, 800)
    while cv2.waitKey(33) != 27:
        pass
    np.savetxt('city_scores.csv', city_scores, ',')

    print("Thanks for playing!")
