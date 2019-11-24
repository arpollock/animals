import cv2
import os
import numpy as np
import math

top_image = None
top_image_hsv = None
ocean_image = None
color_elv_image = None
color_elv_image_hsv = None
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
    global top_image_hsv
    global ocean_image
    global h
    global w
    global scale_thousand_km
    global color_elv_image
    global color_scale_height
    global color_elv_image_hsv

    top_image = cv2.imread(cd + '/data/top_map_black_ocean.png')
    top_image_hsv = cv2.cvtColor(top_image, cv2.COLOR_BGR2HSV)
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
    color_elv_image_hsv = cv2.cvtColor(color_elv_image, cv2.COLOR_BGR2HSV)
    (color_scale_height, dummy1, dummy2) = color_elv_image.shape
    print("the height of the color scale image is {} px".format(
        color_scale_height))
    # for i in range(0, color_scale_height): # print debug make sure works
    #     print(f'{i}: {color_elv_image[i, 0]}')

def in_bounds(x, y):
    if x >= 0 and y >= 0 and x < w and y < h:
        return True
    else:
        return False

def get_ocean_or_land(x, y):
    ''' returns 0 for ocean (black pixel) or 1 for land (white),
    returns -1 if unknown'''
    if in_bounds(x, y):
        (b, g, r) = ocean_image[y, x]
        if b > 127 and g > 127 and r > 127:  # assume white -> land
            return 1
        if b < 127 and g < 127 and r < 127:  # assume black -> ocean
            return 0
        print(f'Could not determine land or water with RGB: ({r},{g},{b})')
        return -1
    print(f'Invalid pixel value(s)')
    return -1

def get_elevation(x, y) -> float:
    # note on accessing individual pixels
    # OpenCV color images in the RGB (Red, Green, Blue) color space
    # have a 3-tuple associated with each pixel: (B, G, R) not RGB!!
    if in_bounds(x, y):
        (top_b, top_g, top_r) = top_image_hsv[y, x]
        # print(f'({top_b}, {top_g}, {top_r})')
        return closet_color_elv(top_b, top_g, top_r)
    print(f'Invalid pixel value(s)')
    return -1

def closet_color_elv(b=0, g=0, r=0) -> float:
    closest_color = -1
    closest_color_dif = 300  # bigger than 0-255 range so will get replaced
    if b < 5 and g < 5 and r < 5:  # black -> ocean
        return 0
    for i in range(0, color_scale_height):  # print debug make sure works
        (b_c, g_c, r_c) = color_elv_image_hsv[i, 0]
        color_diff = (abs(r - r_c)**2 + abs(g - g_c)**2 + abs(b - b_c)**2)**0.5
        if color_diff < closest_color_dif:
            closest_color_dif = color_diff
            closest_color = i
    # print(f'closest color row: {closest_color}')
    return (MAX_ELEVATION
            - ((closest_color / color_scale_height) * MAX_ELEVATION))

if __name__ == "__main__":

    read_image()
    option = None
    while option != "-1":
        print("Options: ")
        print("\t0 - ocean/land")
        print("\t1 - elevation")
        print("\t-1 - EXIT")
        print("\t-2 - read file")
        option = int(input("What feature matrix would you like to generate? "))

        out_arr = np.full((h, w), None)
        out_file = None

        if option == -1:
            quit()
        elif option == 0:
            out_file = "ocean_or_land.npy"
            for y in range(h):
                for x in range(w):
                    out_arr[y,x] = get_ocean_or_land(x, y)
        elif option == 1:
            out_file = "elevation.npy"
            for y in range(h):
                if y % 50 == 0:
                    print(y)
                for x in range(w):
                    out_arr[y,x] = get_elevation(x, y)
        elif option == -2:
            file_name = input("What is the file name/path? ")
            read_arr = np.load(file_name)
            print('\nShape: ', read_arr.shape)
            # print("[")
            # for y in range(read_arr.shape[0]):
            #     print("[", end = '')
            #     for x in range(read_arr.shape[1]):
            #         print(read_arr[y,x], end = ' ')
            #     print("]")
            # print("]")
        else:
            print("INVALID OPTION. Please try again.")
            continue
        if not option == -2:
            np.save(out_file, out_arr)
