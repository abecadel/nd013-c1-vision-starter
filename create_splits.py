import argparse
import glob
import os
import random
import collections
import logging
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import numpy as np
from object_detection.protos import input_reader_pb2
from object_detection.builders.dataset_builder import build as build_dataset


def get_dataset(tfrecord_path, label_map="label_map.pbtxt"):
    """
    Opens a tf record file and create tf dataset
    args:
      - tfrecord_path [str]: path to a tf record file
      - label_map [str]: path the label_map file
    returns:
      - dataset [tf.Dataset]: tensorflow dataset
    """
    input_config = input_reader_pb2.InputReader()
    input_config.label_map_path = label_map
    input_config.tf_record_input_reader.input_path[:] = [tfrecord_path]

    dataset = build_dataset(input_config)
    return dataset


def get_module_logger(mod_name):
    """simple logger"""
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def calculate_label_distro(dataset):
    co = collections.Counter()
    for row in dataset:
        co.update(row["groundtruth_classes"].numpy())
    return co


def get_distros(path):
    distros = {}
    for data_file in tqdm(glob.glob(path)):
        df = get_dataset(data_file).take(1000)
        co = calculate_label_distro(df)
        distros[data_file] = co

    return pd.DataFrame({x: dict(distros[x].items()) for x in distros}).T.fillna(0)


def add_has_value_column(df):
    ret = pd.DataFrame()
    for col in df.columns:
        ret[f"has_{col}"] = (df[col] > 0.0).astype("category")
    return ret


def merge_categorical_columns(df):
    return df.apply(lambda x: " ".join(x.astype(str)), axis=1).astype("category")


def prepare_split_files(path):
    df = get_distros(path)
    dd = merge_categorical_columns(add_has_value_column(df))
    dataset_train, dataset_test = train_test_split(
        dd.cat.codes, stratify=dd.cat.codes.values, test_size=0.1, random_state=1
    )
    dataset_train, dataset_val = train_test_split(
        dataset_train, stratify=dataset_train.values, test_size=0.11, random_state=1
    )
    return dataset_train, dataset_test, dataset_val


def move_files(files, destination):
    os.makedirs(destination, exist_ok=True)
    for file in tqdm(files):
        shutil.copy(file, destination)


def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    dataset_train, dataset_test, dataset_val = prepare_split_files(
        f"{source}/*.tfrecord"
    )

    move_files(dataset_train.index, f"{destination}/train")
    move_files(dataset_test.index, f"{destination}/test")
    move_files(dataset_val.index, f"{destination}/val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data into training / validation / testing"
    )
    parser.add_argument("--source", required=True, help="source data directory")
    parser.add_argument(
        "--destination", required=True, help="destination data directory"
    )
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info("Creating splits...")
    split(args.source, args.destination)
