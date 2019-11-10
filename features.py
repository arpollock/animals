#!/usr/bin/env python

# File for defining, dividing, and working with the feature set of the model

import pandas
import vultures
import numpy as np


class feature_hub:
    """Class which defines dimensionality of features"""

    def __init__(self):
        """Initialize class with features and dimensions"""

        self.feature_dict = {
            "water": 2,
            "coast": 10,
            "elevation": 10
        }

    def get_counts(self, feature=None):
        """Returns the number of buckets for the given feature string.
        If no feature is passed, returns the whole dictionary of strings
        and counts"""

        if feature is not None:
            return self.feature_dict[feature]
        else:
            return self.feature_dict
