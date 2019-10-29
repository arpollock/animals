#! /usr/bin/env python

# Python file for reading and interacting with the topographic data

# import pandas as pd
import cv2
import imutils
import os
import numpy as np
import math

top_image = None
ocean_image = None
color_elv_image = None
h = 0
w = 0
scale_thousand_km = 0
color_scale_height = 0

MIN_ELEVATION = 0
MAX_ELEVATION = 13.3209  # max elevation of topo from scale image (in km)

UPPER_LEFT_LAT = 55.61  # latitude of upper left corner of topo image
UPPER_LEFT_LONG = -136.18  # longitude of upper left corner of topo image
LOWER_RIGHT_LAT = 6.72  # latitude of lower right corner of topo image
LOWER_RIGHT_LONG = -51.55  # longitude of lower right corner of topo image

MAP_LON_DELTA = LOWER_RIGHT_LONG - UPPER_LEFT_LONG
MAP_LAT_BOTTOM_DEGREE = LOWER_RIGHT_LAT * math.pi / 180

# https://databasin.org/maps/new#datasets=588bcc4e1f2646e589e0fc9593498e3d


def read_image():
    """Uses OpenCV to read the various images and get the needed constants for
    lat/long to px conversions."""

    # Get current working directory
    cd = os.getcwd()

    # load the input image and show its dimensions, keeping in mind that
    # images are represented as a multi-dimensional NumPy array with
    # shape no. rows (height) x no. columns (width) x no. channels (depth)
    global top_image
    global ocean_image
    global h
    global w
    global scale_thousand_km
    global color_elv_image
    global color_scale_height

    top_image = cv2.imread(cd + '/data/top_map_black_ocean.png')
    ocean_image = cv2.imread(cd + '/data/land_water.png')
    (h, w, d) = top_image.shape
    print("width={}, height={}, depth={}".format(w, h, d))

    # display the image to our screen -- we will need to click the window
    # open by OpenCV and press a key on our keyboard to continue execution
    # cv2.imshow("Image", top_image)
    # cv2.waitKey(0)

    thousand_km_image = cv2.imread(cd + '/data/top_1000km.png')
    (dummy1, scale_thousand_km, dummy2) = thousand_km_image.shape
    print("1000 km is {} px".format(scale_thousand_km))

    color_elv_image = cv2.imread(cd + '/data/top_color_elv_1px.png')
    (color_scale_height, dummy1, dummy2) = color_elv_image.shape
    print("the height of the color scale image is {} px".format(
        color_scale_height))
    # for i in range(0, color_scale_height): # print debug make sure works
    #     print(f'{i}: {color_elv_image[i, 0]}')

def return_pixel(lat, lon):
    """Takes in a latitude and longitude coord and returns a pixel location"""

    x = (lon - UPPER_LEFT_LONG) * (w / MAP_LON_DELTA)

    lat = lat * math.pi / 180
    
    world_map_width = ((w / MAP_LON_DELTA) * 360) / (2 * math.pi)

    map_offset_y = (world_map_width / 2 * math.log(
        (1 + math.sin(MAP_LAT_BOTTOM_DEGREE))
        / (1 - math.sin(MAP_LAT_BOTTOM_DEGREE))))

    y = h - ((world_map_width / 2 * math.log(
        (1 + math.sin(lat))
        / (1 - math.sin(lat)))) - map_offset_y)

    return (round(x), round(y))

def check_lat(latitude=0) -> bool:
    if latitude > UPPER_LEFT_LAT or latitude < LOWER_RIGHT_LAT:
        print(f'ERROR: invalid latitude value for current area.')
        return False
    else:
        return True

def check_long(longitude=0) -> bool:
    if longitude < UPPER_LEFT_LONG and longitude > LOWER_RIGHT_LONG:
        print(f'ERROR: invalid longitude value for current area.')
        return False
    else:
        return True


def get_elevation(latitude=0, longitude=0) -> float:
    
    if((not check_lat(latitude)) or (not check_long(longitude)) ):
        exit()
    
    x, y = return_pixel(latitude, longitude)
    print(f'px: ({x}, {y})')
    # note on accessing individual pixels
    # OpenCV color images in the RGB (Red, Green, Blue) color space
    # have a 3-tuple associated with each pixel: (B, G, R) not RGB!!
    (top_b, top_g, top_r) = top_image[y, x]
    # print(f'({top_b}, {top_g}, {top_r})')
    return closet_color_elv(top_b, top_g, top_r)

def get_ocean_or_land(latitude=0, longitude=0):
    ''' returns 0 for ocean (black pixel) or 1 for land (white), returns -1 if unknown'''
    if((not check_lat(latitude)) or (not check_long(longitude)) ):
        exit()
    x, y = return_pixel(latitude, longitude)
    (b, g, r) = ocean_image[y, x]
    if b > 250 and g > 250 and r > 250: # assume white -> land
        return 1
    if b < 5 and g < 5 and r < 5: # assume black -> ocean
        return 0
    print(f'Could not determine land or water with RGB: ({r},{g},{b})')
    return -1


def closet_color_elv(b=0, g=0, r=0) -> float:
    closest_color = -1
    closest_color_dif = 300  # bigger than 0-255 range so will get replaced
    if b < 5 and g < 5 and r < 5: # black -> ocean
        return 0
    for i in range(0, color_scale_height):  # print debug make sure works
        (b_c, g_c, r_c) = color_elv_image[i, 0]
        color_diff = (abs(r - r_c)**2 + abs(g - g_c)
                      ** 2 + abs(b - b_c)**2)**(1 / 2)
        if color_diff < closest_color_dif:
            closest_color_dif = color_diff
            closest_color = i
    # print(f'closest color row: {closest_color}')
    return (MAX_ELEVATION
            - ((closest_color / color_scale_height) * MAX_ELEVATION))


if __name__ == "__main__":

    read_image()
    latitude = None
    longitude = None
    while latitude != "exit":
        latitude = input("Input a latitude value (or type exit): ")

        if latitude == "exit":
            quit()

        longitude = input("Input a longitude value (or type exit): ")

        if longitude == "exit":
            quit()

        print(f'elevation: {get_elevation(float(latitude), float(longitude))}')
        print(f'ocean (0) or land (1): { get_ocean_or_land( float(latitude), float(longitude) )}')

    print("Thanks for playing!")
