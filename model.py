#!/usr/bin/env python

# File for working with the feature set of the model

import numpy as np
import pandas as pd
from functools import reduce
import operator
import itertools
from return_pixel import return_pixel
import vultures


class feature:
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
        return round((self.buckets - 1) * normal)

    def get_value(self, x, y):
        """Default data reading function. Should access an array read from
        the contents of the file named at self.file, then access
        this matrix at the given x, y, coordinate.
        """

        try:
            return self.data[y - 1, x - 1]
        except IndexError:
            print(f"Shape of data is {self.data.shape}")
            print(f"You tried to access ({x}, {y})")
            return -1


class model:
    """Class which defines parameters of model and model features"""

    def __init__(self):
        """Initialize class with features and dimensions"""

        self.feature_dict = {
            "water": feature("water", 2, file="ocean_or_land.npy"),
            # "coast": feature("coast", 10, 5000, 0),
            "elevation": feature("elevation", 8),
            # "population": feature("population", 10)
        }

        self.set_states()

    def set_states(self, features=[]):
        """Set the states list for the model given the features
        """

        if len(features) == 0:
            features = self.feature_dict.keys()

        self.states = np.zeros(tuple(
            self.feature_dict[f].buckets for f in features), dtype=np.int32)

        range_list = list(
            [range(0, self.feature_dict[f].buckets) for f in features])
        combinations = list(itertools.product(*range_list))
        for s, combo in enumerate(combinations):
            self.states[combo] = int(s)

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

    def get_states(self, features=[]):
        """Returns the necessary number of states to handle all combinations
        of the known features of the model, if a feature list is provided,
        it will only return the state number required for the features
        specified"""

        if len(features) == 0:
            features = self.feature_dict.keys()

        try:
            return reduce(operator.mul,
                          [self.feature_dict[f] for f in features],
                          1)
        except IndexError:
            print("One of the features passed in is not in this model")

    def get_feature_matrix(self, features=[]):
        """Returns an n by d numpy array where n is the number of states,
        and d is the dimensionality of each of those states.
        If a list of features is passed, will return a matrix only
        considering those features."""

        if len(features) == 0:
            features = self.feature_dict.keys()

        try:
            range_list = list(
                [range(
                    0, self.feature_dict[f].buckets
                ) for f in features]
            )
            combinations = list(itertools.product(*range_list))

            return np.array(combinations)
        except IndexError:
            print("One of the features provided was unknown")

    def get_state(self, x, y, features=[]):
        """Returns the state at the given x, y coordinate, assuming
        states are determined by all features of the model. If a list
        of features is passed, only those features are used to generate
        the state space
        """

        if len(features) == 0:
            features = self.feature_dict.keys()

        self.set_states(features)

        features = list(map(lambda name: self.feature_dict[name], features))

        indices = [f.get_bucket(f.function(x, y)) for f in features]
        index_tuple = tuple(indices)

        if -1 in index_tuple:
            return -1

        return self.states[index_tuple]

    def get_trajectory(self, points, features=[]) -> list:
        """Returns an Ix2 matrix representing the trajectory of the given
        series of points."""

        if len(features) == 0:
            features = self.feature_dict.keys()

        pairs = []

        last_x = None
        last_y = None
        for point in points:
            x = point[0]
            y = point[1]
            if last_x is None and last_y is None:
                last_x = x
                last_y = y

            state = self.get_state(x, y, features)
            if state == -1:
                break
            action = get_action(x - last_x, y-last_y)
            # print(f"Point ({x}, {y}) -> State {state}, Action {action}")
            pairs.append((state, action))

            last_x = x
            last_y = y

        return pairs

    def get_trajectories(self, list, features=[]) -> list:
        """Returns a list of trajectories. Splits trajectories
        into pieces to minimize lost data.
        """

        trajectories_list = []
        for df in list:
            trajectory = self.get_trajectory(get_coords(df), features)
            trajectories_list.append(trajectory)

        return trajectories_list


def get_action(dx, dy):
    """Returns the number of the action taken given change in x and
    change in y"""

    if dx != 0:
        dx = dx / abs(dx)
    if dy != 0:
        dy = dy / abs(dy)

    action_dict = {
        -1: {-1: 0, 0: 1, 1: 2},
        0: {-1: 3, 0: 4, 1: 5},
        1: {-1: 6, 0: 7, 1: 8}
    }

    return action_dict[dy][dx]


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

    dx = dest_x - last_x
    dy = dest_y - last_y

    if (dy == 0 and dx == 0):
        return []

    max_val = max(abs(dx), abs(dy))

    x_step = float(dx)/max_val
    y_step = float(dy)/max_val

    return list(
            [(round(last_x + x_step * (i + 1)),
              round(last_y + y_step * (i + 1))) for i in range(max_val)])


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
    test_hub = model()
    print("***Features***")
    print(test_hub.list_features())
    print("***Full Feature Matrix***")
    print(test_hub.get_feature_matrix())
    print("***Feature Matrix with just water***")
    print(test_hub.get_feature_matrix(["water"]))
    print("***State of 0, 0 using water feature***")
    print(test_hub.get_state(0, 0, ["water"]))
    print("***Reading vulture data***")
    df = vultures.read_file()
    my_df = vultures.get_data_by_name(df, vultures.get_west_names())[0]
    coords = get_coords(my_df)
    print("***Trajectory for first vulture on west***")
    print(test_hub.get_trajectory(coords))
    print("***Trajectories for all vultures on west***")
    west_birds = vultures.get_data_by_name(df, vultures.get_west_names())
    print(test_hub.get_trajectories(west_birds))
