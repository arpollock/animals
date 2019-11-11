#!/usr/bin/env python

# File for working with the feature set of the model

import numpy as np
from functools import reduce
import operator
import itertools
from water import get_if_water_xy


class feature:
    """Class which represents a single feature of the data to be used
    for learning rewards"""

    def __init__(self, name, buckets, max, min, custom_func=None, file=None):
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
        self.max = max
        self.min = min
        if custom_func is None:
            self.function = self.get_value
        else:
            self.function = custom_func

        if file is None:
            self.file = f"{self.name}.txt"
        else:
            self.file = file

    def __repr__(self):
        """How the feature should represent itself"""

        return f"{self.name} feature with {self.buckets} buckets"

    def get_bucket(self, value):
        """Returns the bucket number that a given value should be placed in
        for a given feature, given numbers representing the max and min of
        values for this feature. Returns an int"""

        normal = (value - self.min) / (self.max - self.min)
        return round(self.buckets * normal)

    def get_value(self):
        """Default data reading function. Should construct an array from
        reading the contents of the file named at self.file, then access
        this matrix at the given x, y, coordinate.
        """

        # TODO: Actually read data and return a real value
        pass


class model:
    """Class which defines parameters of model and model features"""

    def __init__(self):
        """Initialize class with features and dimensions"""

        self.feature_dict = {
            "water": feature("water", 2, 1, 0, get_if_water_xy),
            "coast": feature("coast", 10, 5000, 0),
            "elevation": feature("elevation", 10, 5000, 0)
        }

        self.set_states()

    def set_states(self, features=[]):
        """Set the states list for the model given the features
        """

        if len(features) == 0:
            features = self.feature_dict.keys()

        self.states = np.zeros(tuple(
            self.feature_dict[f].buckets for f in features))

        range_list = list(
            [range(0, self.feature_dict[f].buckets) for f in features])
        combinations = list(itertools.product(*range_list))
        for s, combo in enumerate(combinations):
            self.states[combo] = s

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
            return self.feature_dict[feature].get_bucket(value, max, min)
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

        return self.states[[f.get_bucket(f.function(x, y)) for f in features]]


def get_action(dx, dy):
    """Returns the number of the action taken given change in x and
    change in y"""

    dx = dx / abs(dx)
    dy = dy / abs(dy)

    action_dict = {
        -1: {-1: 0, 0: 1, 1: 2},
        0: {-1: 3, 0: 4, 1: 5},
        1: {-1: 6, 0: 7, 1: 8}
    }

    return action_dict[dy][dx]


if __name__ == "__main__":
    test_hub = model()
    print("***Features***")
    print(test_hub.list_features())
    print("***Full Feature Matrix***")
    print(test_hub.get_feature_matrix())
    print("***Feature Matrix with just water***")
    print(test_hub.get_feature_matrix(["water"]))

    print(test_hub.get_state(0, 0, ["water"]))
