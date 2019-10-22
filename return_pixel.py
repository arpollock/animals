import math

def return_pixel(lat, lon):
    map_width = 1928
    map_height = 1378

    map_lon_left = -136.18
    map_lon_right = -51.55
    map_lon_delta = map_lon_right - map_lon_left

    map_lat_bottom = 6.72
    map_lat_bottom_degree = map_lat_bottom * math.pi / 180

    x = (lon - map_lon_left) * (map_width / map_lon_delta)

    lat = lat * math.pi / 180
    world_map_width = ((map_width / map_lon_delta) * 360) / (2 * math.pi)
    map_offset_y = (world_map_width / 2 * math.log((1 + math.sin(map_lat_bottom_degree)) / (1 - math.sin(map_lat_bottom_degree))))
    y = map_height - ((world_map_width / 2 * math.log((1 + math.sin(lat)) / (1 - math.sin(lat)))) - map_offset_y)

    return (x, y)
