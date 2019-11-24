import numpy as np
import cv2


city_scores = np.load('population.npy', allow_pickle=True)
city_scores_heatmap = np.divide(city_scores, np.max(city_scores))
city_scores_heatmap = np.multiply(city_scores_heatmap, 255).astype(np.uint8)
cv2.imwrite('visuals/population.png', city_scores_heatmap)
cv2.imwrite('visuals/population_heatmap.png', cv2.applyColorMap(city_scores_heatmap, cv2.COLORMAP_JET))

water = np.load('ocean_or_land.npy', allow_pickle=True)
water_heatmap = np.divide(water, np.max(water))
water_heatmap = np.multiply(water_heatmap, 255).astype(np.uint8)
cv2.imwrite('visuals/ocean_or_land.png', water_heatmap)
cv2.imwrite('visuals/ocean_or_land_heatmap.png', cv2.applyColorMap(water_heatmap, cv2.COLORMAP_JET))

elevation = np.load('elevation.npy', allow_pickle=True)
elevation_heatmap = np.divide(elevation, np.max(elevation))
elevation_heatmap = np.multiply(elevation_heatmap, 255).astype(np.uint8)
cv2.imwrite('visuals/elevation.png', elevation_heatmap)
cv2.imwrite('visuals/elevation_heatmap.png', cv2.applyColorMap(elevation_heatmap, cv2.COLORMAP_JET))

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
cv2.imwrite('visuals/coast.png', coast_heatmap)
cv2.imwrite('visuals/coast_heatmap.png', cv2.applyColorMap(coast_heatmap, cv2.COLORMAP_JET))
