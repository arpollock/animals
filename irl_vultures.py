import numpy as np
import model
import vultures
from irl3.mdp import gridworld
from irl3 import maxent_irl

# Create the vultures model
model = model.Model(275, 325, 475, 525)
# Read the vulture data
df = vultures.read_file()
# Narrow down birds to those deemed acceeptable
west_birds = vultures.get_data_by_name(df, vultures.get_west_names())
# Get the feature matrix
feature_matrix = model.get_feature_matrix()
assert feature_matrix is not None
# Get the Trajectories
trajectories = model.get_trajectories(west_birds)
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
rewards_maxent = maxent_irl.maxent_irl(feature_matrix, P_a, 0.8,
                                       trajectories, 0.02, 20)
