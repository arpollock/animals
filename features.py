#!/usr/bin/env python

# File for working with the feature set of the model

import numpy as np
from functools import reduce
import operator
import itertools


class feature_hub:
    """Class which defines dimensionality of features"""

    def __init__(self):
        """Initialize class with features and dimensions"""

        self.feature_dict = {
            "water": 2,
            "coast": 10,
            "elevation": 10
        }

    def list_features(self):
        """Returns a list of feature strings"""

        return list(self.feature_dict.keys())

    def get_counts(self, feature=None):
        """Returns the number of buckets for the given feature string.
        If no feature is passed, returns the whole dictionary of strings
        and counts"""

        if feature is not None:
            return self.feature_dict[feature]
        else:
            return self.feature_dict

    def get_bucket(self, feature, value, max, min):
        """Returns the bucket number that a given value should be placed in
        for a given feature, given numbers representing the max and min of
        values for this feature. Returns an int"""

        try:
            normal = (value - min) / (max - min)
            return round(self.feature_dict[feature] * normal)
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
                [range(0, self.feature_dict[f]) for f in features]
            )
            combinations = list(itertools.product(*range_list))

            return np.array(combinations)
        except IndexError:
            print("One of the features provided was unknown")


if __name__ == "__main__":
    test_hub = feature_hub()
    print("***Features***")
    print(test_hub.list_features())
    print("***Full Feature Matrix***")
    print(test_hub.get_feature_matrix())
    print("***Feature Matrix with just water***")
    print(test_hub.get_feature_matrix(["water"]))
