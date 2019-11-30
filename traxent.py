# Implementation of command line interface for animal tracking 2

import numpy as np
import pandas as pd
import argparse
import model as mod
import json

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
        details = config[feature]
        obj = mod.Feature(*details.values())
        new_model.feature_dict[feature] = obj

    hyperparams = config["hyperparams"]
    if hyperparams is None:
        hyperparams = {}

    return new_model, hyperparams


def write_config(model, hyperparams={}, filename="traxent.cfg"):
    """Writes model with features and a dictionary of hyperparameters to
    a json dictionary file specified by filename, which defaults to traxent.cfg
    """

    try:
        existing = json.load(open(filename, "r"))
    except IOError:
        file = open(filename, "w")

    try:
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
        feature_config["max"] = obj.max
        feature_config["min"] = obj.min
        feature_config["custom_func"] = obj.custom_func
        feature_config["file"] = obj.file
        features_config[feature] = feature_config

    config["features"] = features_config
    json.dump(config, file)


# Start The CLI
config_file = "trackr.cfg" if args.config is None else args.config

try:
    model, hyperparams = read_config(config_file)
except IOError:
    print(f"Configuration file {config_file} not found. Using fresh model")
    model = mod.Model()
