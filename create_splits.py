import argparse
import glob
import os
import random
import collections
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import get_dataset
import shutil
import numpy as np
from utils import get_module_logger


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
    return df.apply(lambda x: " ".join(x.astype(str)), axis=1).astype('category')


def prepare_split_files(path):
    df = get_distros(path)
    dd = merge_categorical_columns(add_has_value_column(df))
    dataset_train, dataset_test = train_test_split(dd.cat.codes, stratify=dd.cat.codes.values, test_size=0.1, random_state=1)
    dataset_train, dataset_val = train_test_split(dataset_train, stratify=dataset_train.values, test_size=0.11, random_state=1)
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
    dataset_train, dataset_test, dataset_val = prepare_split_files(f"{source}/*.tfrecord")
    
    move_files(dataset_train.index, f"{destination}/train")
    move_files(dataset_test.index, f"{destination}/test")
    move_files(dataset_val.index, f"{destination}/val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)