#! /usr/bin/env python

# Python file for reading and interacting with the topographic data

# import pandas as pd
import cv2
import imutils
import os
import numpy as np

top_image = None
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
CORR_MATRIX = None

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
    global h
    global w
    global scale_thousand_km
    global color_elv_image
    global color_scale_height
    global CORR_MATRIX

    top_image = cv2.imread(cd + '/data/top_map.png')
    (h, w, d) = top_image.shape
    print("width={}, height={}, depth={}".format(w, h, d))
    CORR_MATRIX = np.matrix([[0, 0, UPPER_LEFT_LAT, UPPER_LEFT_LONG],
                             [0, h, UPPER_LEFT_LAT, LOWER_RIGHT_LONG],
                             [w, h, LOWER_RIGHT_LAT, LOWER_RIGHT_LONG],
                             [w, 0, LOWER_RIGHT_LAT, UPPER_LEFT_LONG]])

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

# TODO: make so is actually accurate


"""https://www.accuware.com/support/convert-coordinates-latitude-and-longitude-into-pixels-xy-of-a-floor-plan-image/"""


def convert_lat_to_px(latitude=0) -> int:
    if latitude <= UPPER_LEFT_LAT and latitude >= LOWER_RIGHT_LAT:
        percent = (UPPER_LEFT_LAT - latitude) / \
            (UPPER_LEFT_LAT - LOWER_RIGHT_LAT)
        return round(percent * h)
    else:
        print(f'ERROR: invalid latitude value for current area.')
        exit()


def convert_long_to_px(longitude=0) -> int:
    if longitude >= UPPER_LEFT_LONG and longitude <= LOWER_RIGHT_LONG:
        percent = abs((UPPER_LEFT_LONG - longitude)
                      / (UPPER_LEFT_LONG - LOWER_RIGHT_LONG))
        return round(percent * w)
    else:
        print(f'ERROR: invalid longitude value for current area.')
        exit()


def get_elevation(latitude=0, longitude=0) -> float:
    y = convert_lat_to_px(latitude)
    x = convert_long_to_px(longitude)
    # note on accessing individual pixels
    # OpenCV color images in the RGB (Red, Green, Blue) color space
    # have a 3-tuple associated with each pixel: (B, G, R) not RGB!!
    (top_b, top_g, top_r) = top_image[y, x]
    # print(f'({top_b}, {top_g}, {top_r})')
    return closet_color_elv(top_b, top_g, top_r)


def closet_color_elv(b=0, g=0, r=0) -> float:
    closest_color = -1
    closest_color_dif = 300  # bigger than 0-255 range so will get replaced
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

        print(get_elevation(float(latitude), float(longitude)))

    print("Thanks for playing!")
