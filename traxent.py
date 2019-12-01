# Implementation of command line interface for animal tracking 2

import numpy as np
import os
import argparse
import json
import time
import datetime
import glob
from irl3.mdp import gridworld
from irl3 import maxent_irl
import model as mod
import move_data
from draw_plot import draw_plot

parser = argparse.ArgumentParser()
parser.add_argument(
    "-d", "--data", action="append", nargs=2,
    help="Specify a data filename followed by #buckets")
parser.add_argument("-c", "--config", help="Specify a configuration file")
args = parser.parse_args()


def read_config(filename="traxent.cfg"):
    """Attempts to read the traxent.cfg file to discover data configurations,
    hyperparameters, etc. Raises an IOError on failure to do so. Otherwise,
    Returns a model object which has been properly configured, and a
    dictionary of hyperparameters
    """

    try:
        file = open(filename, "r")
        config = json.load(file)
        file.close()
    except IOError:
        raise IOError(f"File {filename} was not found or could not be read")

    model_config = config["model"]
    new_model = mod.Model(*model_config.values())

    for feature in config["features"].keys():
        details = config["features"][feature]
        obj = mod.Feature(*details.values())
        new_model.feature_dict[feature] = obj

    hyperparams = config["hyperparams"]
    if hyperparams is None:
        hyperparams = {"learning rate": 0.02,
                       "discount factor": 0.8,
                       "iterations": 20}

    data_file = config["data file"]

    return new_model, hyperparams, data_file


def write_config(model, data_file, hyperparams={}, filename="traxent.cfg"):
    """Writes model with features and a dictionary of hyperparameters to
    a json dictionary file specified by filename, which defaults to traxent.cfg
    """

    existing = None
    try:
        existing = json.load(open(filename, "r"))
    except IOError:
        print("No existing configuration found")

    file = open(filename, "w")

    try:
        if existing is not None:
            print(f"Found existing configuration file at {filename}")
            print(f"Backing up to {filename}.bak")
            backup = open(f"{filename}.bak", "w")
            json.dump(existing, backup)
            backup.close()
    except IOError:
        print("Could not write backup file")
        exit(1)

    config = {}
    config["hyperparams"] = hyperparams
    model_config = {}
    model_config["x_start"] = model.x_start
    model_config["x_end"] = model.x_end
    model_config["y_start"] = model.y_start
    model_config["y_end"] = model.y_end
    model_config["shape"] = model.shape
    config["model"] = model_config

    features_config = {}

    for feature in model.feature_dict:
        obj = model.feature_dict[feature]
        feature_config = {}
        feature_config["name"] = obj.name
        feature_config["buckets"] = obj.buckets
        feature_config["max"] = int(obj.max) + 1
        feature_config["min"] = int(obj.min)
        feature_config["custom_func"] = obj.custom_func
        feature_config["file"] = obj.file
        features_config[feature] = feature_config

    config["features"] = features_config

    config["data file"] = data_file

    json.dump(config, file)
    file.close()


def irl_rewards(model, filename, hyperparams):
    """Do everything irl_vultures does
    """

    try:
        with open(filename, 'r') as file:
            traj_list = eval(file.read())
            trajectories = model.get_trajectories(traj_list)
            file.close()
    except IOError:
        raise ValueError("File for animals not found, please generate using option 6")

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
    learning_rate = hyperparams["learning rate"]
    gamma = hyperparams["discount factor"]
    iterations = hyperparams["iterations"]
    rewards_maxent = maxent_irl.maxent_irl(feature_matrix, P_a, gamma,
                                           trajectories, learning_rate,
                                           iterations,
                                           error=0.1)
    end = time.time()
    print("Time Elapsed: ", end - start)

    first_dim = model.shape[0]
    last_dim = model.shape[1]
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    timestamp = now

    save_file = f"rewards_{first_dim}x{last_dim}_{timestamp}"

    np.save(save_file, rewards_maxent)
    return (save_file + ".npy")


# Start The CLI
config_file = "trackr.cfg" if args.config is None else args.config

try:
    model, hyperparams, data_file = read_config(config_file)
except IOError:
    print(f"Configuration file {config_file} not found. Using fresh model")
    model = mod.Model()
    hyperparams = {"learning rate": 0.02,
                   "discount factor": 0.8,
                   "iterations": 20}
    data_file = None

rewards_file = None

print("Welcome to the Animal Tracking CLI!")
options = {1:   'Load Features',
           2:   'Generate Reward Map',
           3:   'Display Reward Map',
           4:   'Modify Hyperparameters',
           5:   'Modify Search Area',
           6:   'Change Animal Data',
           7:   'Save and Exit'}

while True:
    os.system("clear")
    print('Please choose an option below to start')

    for i, option in zip(options.keys(), options.values()):
        print(f"{i}: {option}")

    option = None
    while option not in options.keys():
        try:
            option = int(input("> "))
        except TypeError as e:
            print("Option was not an int")
            continue

    if option == 1:
        # load features
        feature_opts = {1: "Load a new feature",
                        2: "Remove an existing feature",
                        3: "Change a bin number"}

        for i, feature_option in zip(feature_opts.keys(),
                                     feature_opts.values()):
            print(f"{i}: {feature_option}")

        feature_option = None
        while feature_option not in feature_opts.keys():
            try:
                feature_option = int(input("> "))
            except TypeError as e:
                print("Option was not an int")
                continue

        if feature_option == 1:
            # load feature
            name = input("Feature name: ")
            filename = input("Enter npy filename: ")
            bins = input("How many bins for the feature: ")
            feature = mod.Feature(name, bins, file=filename)
            model.feature_dict[name] = feature
        elif feature_option == 2:
            # remove feature
            if len(model.feature_dict.keys()) == 0:
                print("There are no features to remove")
                continue
            for feature in model.feature_dict.keys():
                print(f"{feature}: {model.feature_dict[feature]}")
            to_remove = None
            while to_remove not in model.feature_dict.keys():
                to_remove = input("Enter name of feature to remove: ")
            del model.feature_dict[to_remove]
        elif feature_option == 3:
            # change bin number
            for feature in model.feature_dict.keys():
                print(f"{feature}: {model.feature_dict[feature]}")
            to_change = None
            while to_change not in model.feature_dict.keys():
                to_change = input("Enter name of feature to change: ")
            bins = None
            while not isinstance(bins, int):
                try:
                    bins = int(input("Enter number of bins: "))
                except TypeError as e:
                    print("Option was not an int")
                    continue
            model.feature_dict[to_change].buckets = bins

    elif option == 2:
        # generate the reward map
        rewards_file = irl_rewards(model, data_file, hyperparams)
        print(f"Rewards file generated at {rewards_file}")
    elif option == 3:
        # display the reward map
        plot_file = rewards_file
        reward_options = {1: "Display reward map from current session",
                          2: "Display previous reward map"}
        for reward_option, reward_option_index in zip(reward_options.values(),
                                                      reward_options.keys()):
            print(f'{reward_option_index}: {reward_option}')
        reward_option = None
        while reward_option not in reward_options.keys():
            try:
                reward_option = int(input("> "))
            except TypeError as e:
                print("Option was not an int")
                continue

        if plot_file is None or reward_option == 2:
            if plot_file is None and reward_option != 2:
                print("No reward map has been generated this session")
            possible_rewards = {}
            for i, filename in enumerate(glob.glob("rewards_*.npy")):
                possible_rewards[i+1] = filename
            if len(possible_rewards.keys()) == 0:
                print("There are no existing reward maps")
                continue
            print("Please select a reward map from below:")
            for reward_map, reward_map_index in zip(
                    possible_rewards.values(),
                    possible_rewards.keys()):
                print(f'{reward_map_index}: {reward_map}')
                reward_map_index = None
            while reward_map_index not in possible_rewards.keys():
                try:
                    reward_map_index = int(input("Reward Map: "))
                except TypeError as e:
                    print("Option was not an int")
                    continue
            plot_file = possible_rewards[reward_map_index]
        print(f"Drawing plot for {plot_file}")
        draw_plot(plot_file)
    elif option == 4:
        # Deal with hyperparams
        for param in hyperparams.keys():
            print(f"{param}: {hyperparams[param]}")
        to_change = None
        while to_change not in hyperparams.keys():
            to_change = input("Enter name of hyperparameter to change: ")
        try:
            new_value = int(input("What is the new desired value: "))
        except TypeError as e:
            print("Option was not an int")
            continue
        hyperparams[to_change] = new_value
    elif option == 5:
        # Modify search area
        print("Existing values:")
        current_vals = {"x_start":  model.x_start,
                        "x_end":    model.x_end,
                        "y_start":  model.y_start,
                        "y_end":    model.y_end}

        for val in current_vals.keys():
            print(f"{val}: {current_vals[val]}")

        y_bound, x_bound = list(model.feature_dict.values())[0].data.shape

        while True:
            try:
                x_start = int(input("New x_start: "))
            except TypeError as e:
                print("Option was not an int")
                continue
            try:
                x_end = int(input("New x_end: "))
            except TypeError as e:
                print("Option was not an int")
                continue
            try:
                y_start = int(input("New y_start: "))
            except TypeError as e:
                print("Option was not an int")
                continue
            try:
                y_end = int(input("New y_end: "))
            except TypeError as e:
                print("Option was not an int")
                continue
            if x_start < x_end < x_bound and y_start < y_end < y_bound:
                model.x_start = x_start
                model.x_end = x_end
                model.y_start = y_start
                model.y_end = y_end
                break
            print("An entered value was invalid, please try again")
    elif option == 6:
        print("Select a data option below")
        data_opts = {1: "Load new animal data",
                     2: "Load existing animal data"}

        for i, data_option in zip(data_opts.keys(), data_opts.values()):
            print(f"{i}: {data_option}")

        data_option = None
        while data_option not in data_opts.keys():
            try:
                data_option = int(input("> "))
            except TypeError as e:
                print("Option was not an int")
                continue

        if data_option == 1:
            origin_file = input("Name of source file: ")
            # Read the vulture data
            df = move_data.read_file(origin_file)
            # Narrow down birds to those deemed acceeptable
            animals = move_data.get_data_by_name(df, move_data.get_names(df))
            filename = input("Name of new data file: ")
            file = open(filename, 'w')
            coords_list = list(
                [list(mod.get_coords(animal)) for animal in animals])
            file.write(str(coords_list))
            file.close()
            data_file = filename
        elif data_option == 2:
            data_file = input("Name of data file: ")

    elif option == 7:
        # Save and exit
        try:
            write_config(model, data_file, hyperparams, filename=config_file)
            quit()
        except IOError as e:
            print("Problem saving configuration. Will save to Autosave.cfg")
            try:
                write_config(model, data_file, hyperparams,
                             filename="Autosave.cfg")
            except IOError:
                print("Could not autosave, dumping to command line")
                # TODO: Dump dict to terminal
            raise e
