import numpy as np
import cv2

# reward_1_50 location:
end_x_1 = 979
end_y_1 = 1172
start_x_1 = end_x_1 - 50
start_y_1 = end_y_1 - 50
start_1 = (start_x_1, start_y_1)
end_1 = (end_x_1, end_y_1)

# reward_2_50 location:
end_x_2 = 1185
end_y_2 = 1300
start_x_2 = end_x_2 - 50
start_y_2 = end_y_2 - 50
start_2 = (start_x_2, start_y_2)
end_2 = (end_x_2, end_y_2)

color_2 = (254, 155, 162)
color_1 = (0, 0, 0)
thickness = 5


def draw_boxes(heatmap):
    heatmap = cv2.rectangle(heatmap, start_1, end_1, color_1, thickness)
    heatmap = cv2.rectangle(heatmap, start_2, end_2, color_2, thickness)
    return heatmap


city_scores = np.load('population.npy', allow_pickle=True)
city_scores_heatmap = np.divide(city_scores, np.max(city_scores))
city_scores_heatmap = np.multiply(city_scores_heatmap, 255).astype(np.uint8)

cv2.imwrite('visuals_box/population.png', city_scores_heatmap)
cv2.imwrite('visuals_box/population_heatmap.png', draw_boxes(cv2.applyColorMap(city_scores_heatmap, cv2.COLORMAP_JET)))

water = np.load('ocean_or_land.npy', allow_pickle=True)
water_heatmap = np.divide(water, np.max(water))
water_heatmap = np.multiply(water_heatmap, 255).astype(np.uint8)

cv2.imwrite('visuals_box/ocean_or_land.png', water_heatmap)
cv2.imwrite('visuals_box/ocean_or_land_heatmap.png', draw_boxes(cv2.applyColorMap(water_heatmap, cv2.COLORMAP_JET)))

elevation = np.load('elevation.npy', allow_pickle=True)
elevation_heatmap = np.divide(elevation, np.max(elevation))
elevation_heatmap = np.multiply(elevation_heatmap, 255).astype(np.uint8)

cv2.imwrite('visuals_box/elevation.png', elevation_heatmap)
cv2.imwrite('visuals_box/elevation_heatmap.png', draw_boxes(cv2.applyColorMap(elevation_heatmap, cv2.COLORMAP_JET)))

coast = (np.load('coast.npy', allow_pickle=True)).astype(np.float32)
# for iteration in range(10):
#     print("iteration", iteration + 1)
#     for i in range(coast.shape[0]):
#         for j in range(coast.shape[1]):
#             if i - 1 >= 0 and coast[i][j] > coast[i-1][j] + 1:
#                 was = coast[i][j]
#                 coast[i][j] = coast[i-1][j] + 1
#                 # print(f'dist at ({j}, {i}) was {was} and now is {coast[i][j]}')
#             if j - 1 >= 0 and i - 1 >= 0 and coast[i][j] > coast[i-1][j-1] + (2 ** 0.5):
#                 was = coast[i][j]
#                 coast[i][j] = coast[i-1][j-1] + (2 ** 0.5)
#                 # print(f'dist at ({j}, {i}) was {was} and now is {coast[i][j]}')
#             if i + 1 < coast.shape[0] and coast[i][j] > coast[i+1][j] + 1:
#                 was = coast[i][j]
#                 coast[i][j] = coast[i+1][j] + 1
#                 # print(f'dist at ({j}, {i}) was {was} and now is {coast[i][j]}')
#             if j + 1 < coast.shape[1] and i + 1 < coast.shape[0] and coast[i][j] > coast[i+1][j+1] + (2 ** 0.5):
#                 was = coast[i][j]
#                 coast[i][j] = coast[i+1][j+1] + (2 ** 0.5)
#                 # print(f'dist at ({j}, {i}) was {was} and now is {coast[i][j]}')
#             if j - 1 >= 0 and coast[i][j] > coast[i][j-1] + 1:
#                 was = coast[i][j]
#                 coast[i][j] = coast[i][j-1] + 1
#                 # print(f'dist at ({j}, {i}) was {was} and now is {coast[i][j]}')
#             if i + 1 < coast.shape[0] and j - 1 >= 0 and coast[i][j] > coast[i+1][j-1] + (2 ** 0.5):
#                 was = coast[i][j]
#                 coast[i][j] = coast[i+1][j-1] + (2 ** 0.5)
#                 # print(f'dist at ({j}, {i}) was {was} and now is {coast[i][j]}')
#             if j + 1 < coast.shape[1] and coast[i][j] > coast[i][j+1] + 1:
#                 was = coast[i][j]
#                 coast[i][j] = coast[i][j+1] + 1
#                 # print(f'dist at ({j}, {i}) was {was} and now is {coast[i][j]}')
#             if i - 1 < coast.shape[0] and j + 1 < coast.shape[1] and coast[i][j] > coast[i-1][j+1] + (2 ** 0.5):
#                 was = coast[i][j]
#                 coast[i][j] = coast[i-1][j+1] + (2 ** 0.5)
#                 # print(f'dist at ({j}, {i}) was {was} and now is {coast[i][j]}')
coast_heatmap = np.divide(coast, np.max(coast))
coast_heatmap = np.multiply(coast_heatmap, 255).astype(np.uint8)
cv2.imwrite('visuals_box/coast.png', coast_heatmap)
cv2.imwrite('visuals_box/coast_heatmap.png', draw_boxes(cv2.applyColorMap(coast_heatmap, cv2.COLORMAP_JET)))
