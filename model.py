#!/usr/bin/env python

# File for working with the feature set of the model

import numpy as np
import pandas as pd
from functools import reduce
import operator
import itertools
from return_pixel import return_pixel
import vultures
from collections import namedtuple
import random

Step = namedtuple('Step', 'cur_state action next_state reward done')


class Feature:
    """Class which represents a single feature of the data to be used
    for learning rewards"""

    def __init__(self, name, buckets,
                 max=None, min=None, custom_func=None, file=None):
        """Initialize feature with name and number of buckets.
        If provided, custom_func sets the function to be called to
        aquire a value for this feature at a given x, y coord, to be
        passed as args.
        Max and min are the maximum and minimum values respectively that
        this feature can have. custom_func(x,y) should return a numerical
        value.  Default is the feature class's get_value function..
        If provided, file will determine the filename of the
        data file from which a pixel array of values can be generated
        for this feature. Default assumes it is called name.txt where
        name is the parameter passed in.
        """

        self.name = name
        self.buckets = buckets

        if file is None:
            self.file = f"{self.name}.npy"
        else:
            self.file = file

        if custom_func is None:
            self.function = self.get_value
            try:
                self.data = np.load(self.file, allow_pickle=True)
            except IOError:
                print(f"There was a problem with the numpy file {self.file}")
                exit(1)
            self.max = self.data.max()
            self.min = self.data.min()
        else:
            self.function = custom_func
            self.max = max
            self.min = min

            if self.max is None or self.min is None:
                raise AttributeError(
                    "Min and Max must be passed for custom  function")

    def __repr__(self):
        """How the feature should represent itself"""

        return f"{self.name} feature with {self.buckets} buckets"

    def get_bucket(self, value):
        """Returns the bucket number that a given value should be placed in
        for a given feature, given numbers representing the max and min of
        values for this feature. Returns an int"""
        if value == -1:
            return -1

        normal = (value - self.min) / (self.max - self.min)
        return int(round((self.buckets - 1) * normal))

    def get_value(self, x, y):
        """Default data reading function. Should access an array read from
        the contents of the file named at self.file, then access
        this matrix at the given x, y, coordinate.
        """

        try:
            return self.data[y, x]
        except IndexError:
            print(f"Shape of data is {self.data.shape}")
            print(f"You tried to access ({x}, {y})")
            return -1


class Model:
    """Class which defines parameters of model and model features"""

    def __init__(self, x_start=0, x_end=None, y_start=0, y_end=None):
        """Initialize class with features and dimensions"""

        self.feature_dict = {
            "water": Feature("water", 2, file="ocean_or_land.npy"),
            "coast": Feature("coast", 10),
            "elevation": Feature("elevation", 8),
            "population": Feature("population", 10)
        }

        shape = Feature("coast", 10).data.shape

        self.x_start = x_start
        if x_end is None:
            self.x_end = shape[1] - 1
        else:
            self.x_end = x_end
        self.y_start = y_start
        if y_end is None:
            self.y_end = shape[0] - 1
        else:
            self.y_end = y_end

        self.shape = (self.y_end + 1 - self.y_start,
                      self.x_end + 1 - self.x_start)
        self.size = self.shape[0] * self.shape[1]

    def list_features(self):
        """Returns a list of feature strings"""

        return list(self.feature_dict.keys())

    def get_counts(self, feature=None):
        """Returns the number of buckets for the given feature string.
        If no feature is passed, returns the whole dictionary of strings
        and counts"""

        if feature is not None:
            return self.feature_dict[feature].buckets
        else:
            return self.feature_dict

    def get_bucket(self, feature, value):
        """Returns the bucket number that a given value should be placed in
        for a given feature. Returns an int"""

        try:
            return self.feature_dict[feature].get_bucket(value)
        except IndexError:
            print(f"{feature} is not a known feature of this model")

    def get_states(self):
        """Returns the necessary number of states to handle all combinations
        of the known features of the model, if a feature list is provided,
        it will only return the state number required for the features
        specified"""

        return ((self.x_end + 1 - self.x_start)
                * (self.y_end + 1 - self.y_start))

    def get_feature_matrix(self, features=[]):
        """Returns an n by d numpy array where n is the number of states,
        and d is the dimensionality of each of those states.
        If a list of features is passed, will return a matrix only
        considering those features."""

        if len(features) == 0:
            features = list(self.feature_dict.keys())

        try:
            f = np.zeros((self.size, len(features)), dtype=np.uint8)
            for i in range(len(features)):
                feature = self.feature_dict[features[i]]
                for y in range(self.shape[0]):
                    for x in range(self.shape[1]):
                        value = feature.get_value(x + self.x_start,
                                                  y + self.y_start)
                        bucket = feature.get_bucket(value)
                        f[y + x * self.shape[0], i] = bucket

            return f
        except IndexError as e:
            print("One of the features provided was unknown")
            print(f"x was {x} and y was {y}")
            raise e

    def get_state(self, x, y):
        """Returns the state at the given x, y coordinate
        """

        return (int(y - self.y_start), int(x - self.x_start))

    def get_episode(self, points) -> list:
        """Returns an list representing an episode of the given
        series of points."""

        episode = []

        last_x = None
        last_y = None
        for point in points:
            x = point[0]
            y = point[1]

            if last_x is None and last_y is None:
                last_x = x
                last_y = y

            if last_x < self.x_start or last_y < self.y_start:
                last_x = x
                last_y = y
                continue

            if last_x > self.x_end or last_y > self.y_end:
                return episode

            cur_state = self.get_state(last_x, last_y)
            cur_state = cur_state[0] + cur_state[1] * self.shape[0]
            next_state = self.get_state(x, y)
            next_state = next_state[0] + next_state[1] * self.shape[0]
            action = get_action(x - last_x, y - last_y)
            # print(f"Point ({x}, {y}) -> State {state}, Action {action}")
            episode.append(Step(cur_state=cur_state,
                                action=action,
                                next_state=next_state,
                                reward=0,
                                done=False))

            last_x = x
            last_y = y

        episode[-1] = Step(cur_state=episode[-1].cur_state,
                           action=episode[-1].action,
                           next_state=episode[-1].next_state,
                           reward=episode[-1].reward,
                           done=True)
        return episode

    def get_trajectories(self, list) -> list:
        """Returns a list of episodes. Takes in a list of dataframes or a
        list pf lists of tuples generated by coords
        """

        trajectories = []

        for item in list:
            if isinstance(item, pd.DataFrame):
                trajectory = self.get_episode(get_coords(item))
            else:
                trajectory = self.get_episode(item)
            if len(trajectory) > 0:
                trajectories.append(trajectory)

        return trajectories

    def pos2idx(self, pos):
        """
        input:
          column-major 2d position
        returns:
          1d index
        """
        return int(pos[0] + pos[1] * self.shape[0])

    def idx2pos(self, idx):
        """
        input:
          1d idx
        returns:
          2d column-major position
        """
        return (int(idx % self.shape[0]), int(idx / self.shape[0]))


def get_action(dx, dy):
    """Returns the number of the action taken given change in x and
    change in y"""

    try:
        assert abs(dx) + abs(dy) < 2
    except AssertionError:
        print("x and y changed by too much")
        print(f"dx is {dx} and dy is {dy}")

    if dx != 0:
        dx = dx / abs(dx)
        if dx < 0:
            return 1
        else:
            return 0
    if dy != 0:
        dy = dy / abs(dy)
        if dy < 0:
            return 3
        else:
            return 2
    return 4


def add_pixels(df) -> pd.DataFrame:
    """Uses the return_pixel function to add x and y pixel locations
    to the data frame"""

    if "x" in df and "y" in df:
        return

    df["x"] = df.apply(lambda row: return_pixel(row["location-lat"],
                                                row["location-long"])[0],
                       axis=1)
    df["y"] = df.apply(lambda row: return_pixel(row["location-lat"],
                                                row["location-long"])[1],
                       axis=1)


def interpolate(last_x, last_y, dest_x, dest_y) -> list:
    """Provides a list of x,y pairs in tuple form that bridge from last_x,
    lasy_y and include dest_x, dest_y. The path will move one step at a
    time in the dimension that changes the most, to guarantee continuity
    """

    steps = []
    while (last_x, last_y) != (dest_x, dest_y):
        dx = dest_x - last_x
        dy = dest_y - last_y
        if abs(dx) > abs(dy):
            last_x += dx / abs(dx) if dx != 0 else 0
        elif abs(dy) > abs(dx):
            last_y += dy / abs(dy) if dy != 0 else 0
        else:
            binary = random.randint(0, 1)
            last_x += (dx / abs(dx) if dx != 0 else 0) * binary
            last_y += (dy / abs(dy) if dy != 0 else 0) * (1 - binary)
        steps.append((last_x, last_y))
    return steps


def get_coords(df) -> list:
    """Takes a data frame representing the path of a single bird, and
    generates (x, y) coords that represent this bird's path.
    Makes use of the interpolate function to fill in gaps"""

    try:
        print(f"Finding coords of {df['individual-local-identifier'].iloc[0]}")
    except IndexError as e:
        print(df)
        raise e

    add_pixels(df)
    last_x = None
    last_y = None

    for row in df.itertuples(index=False):
        x = getattr(row, "x")
        y = getattr(row, "y")
        if last_x is None or last_y is None:
            last_x = x
            last_y = y
            yield (last_x, last_y)
            continue
        # print(f"Interpolating ({last_x}, {last_y})->({x}, {y})")
        for coord in interpolate(last_x, last_y, x, y):
            yield coord
        # print("Finished interpolation")
        last_x = x
        last_y = y


if __name__ == "__main__":
    test_hub = Model()
    print("***Features***")
    print(test_hub.list_features())
    print("***Full Feature Matrix***")
    print(test_hub.get_feature_matrix())
    print("***Feature Matrix with just water***")
    print(test_hub.get_feature_matrix(["water"]))
    print("***Reading vulture data***")
    df = vultures.read_file()
    my_df = vultures.get_data_by_name(df, vultures.get_west_names())[0]
    print("***Trajectories for all vultures on west***")
    west_birds = vultures.get_data_by_name(df, vultures.get_west_names())
    print(test_hub.get_trajectories(west_birds))
