import numpy as np
import model as mod
import vultures
import cv2

# Create the vultures model
model = mod.Model()

try:
    with open('bird_coords.dat', 'r') as file:
        traj_list = eval(file.read())
        trajectories = model.get_trajectories(traj_list)
        file.close()
except IOError:
    print("File for birds not found, will generate")
    # Read the vulture data
    df = vultures.read_file()
    # Narrow down birds to those deemed acceeptable
    west_birds = vultures.get_data_by_name(df, vultures.get_west_names())
    file = open('bird_coords.dat', 'w')
    coords_list = list([list(mod.get_coords(bird)) for bird in west_birds])
    file.write(str(coords_list))
    file.close()
    trajectories = model.get_trajectories(coords_list)


for traj in trajectories:
    assert traj is not None

# Create a giant empty grid for the gridworld
grid = np.zeros((model.shape[0], model.shape[1]), dtype=np.uint32)

for episode in trajectories:
    for step in set(map(lambda e: e.cur_state, episode)):
        location = model.idx2pos(step)
        try:
            assert isinstance(location, tuple)
            assert len(location) == 2
        except AssertionError as e:
            print(location)
            raise e
        grid[(location[0], location[1])] += 1

heatmap = np.divide(grid, np.max(grid))
heatmap = np.multiply(heatmap, 255).astype(np.uint8)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

cv2.imwrite("bird_heat.png", heatmap)
