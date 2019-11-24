import numpy as np
import model as mod
import vultures
from irl3.mdp import gridworld
from irl3 import maxent_irl
from irl3 import img_utils
import matplotlib.pyplot as plt
import time

# Create the vultures model
model = mod.Model(954, 979, 1157, 1172)

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

feature_matrix = model.get_feature_matrix()
assert feature_matrix is not None

for traj in trajectories:
    assert traj is not None
# Find the terminal points from the trajectories
terminals = list(map(lambda traj: traj[-1].next_state, trajectories))

# Create a giant empty grid for the gridworld
grid = [[0 for i in range(model.shape[0])] for j in range(model.shape[1])]
# Create the gridworld
print("Creating the GridWorld")
gw = gridworld.GridWorld(grid, terminals)
print("Getting Transition Probabilities")
# Get Transition Probabilities
P_a = gw.get_transition_mat()
# Create matrix for rewards
print("Running MaxEnt IRL")
start = time.time()
rewards_maxent = maxent_irl.maxent_irl(feature_matrix, P_a, 0.8,
                                       trajectories, 0.02, 20, error=0.1)
end = time.time()
print(end - start)

np.save("50x50rewards.npy", rewards_maxent)
