"""
Utilties to create mock datasets for testing
"""
from collections.abc import Iterable
from random import choice, randint
from string import ascii_lowercase, digits
from typing import Tuple, Optional
from datasets import Dataset, DatasetDict

NUM_FEATURES = NUM_SUBSETS = 10
NAME_LENGTH = 10

def create_dataset_shell(features: Iterable[str] | None = None,
                         subsets: Iterable[str] | None = None) -> Dataset | DatasetDict:
    """Create an near "empty" dataset with only our splits and features
    as relevant information.

    Args:
        features (Iterable[str] | None, optional): An list of dataset columns. Defaults to None.
        subsets (Iterable[str] | None, optional): Slice dataset into each subset. Defaults to None.

    Returns:
        Dataset | DatasetDict: A Dataset when no subsets are given. Otherwise, return a DatasetDict.
    """
    if features is None:
        features = []
    shell = Dataset.from_dict({feature: [None] for feature in features})
    if subsets is not None:
        return DatasetDict({split: shell for split in subsets})
    return shell


def random_list(str_len: int, up_to: int) -> list[str]:
    """Generate a random list of strings

    Args:
        str_len (int): Specify how long each string should be
        up_to (int): Specify number of total elements

    Returns:
        list[str]: A random length list of random strings of length str_len
    """
    # Code adapted from
    # https://stackoverflow.com/questions/34484972/generate-a-list-of-random-string-of-fixed-length-in-python
    chars = ascii_lowercase + digits
    return [''.join((choice(chars)
                    for string in range(randint(1, str_len))))
            for num_strings in range(randint(1, up_to))]


def random_features() -> list[str]:
    """Create a random list of features"""
    return random_list(str_len=NAME_LENGTH, up_to=NUM_FEATURES)


def random_subsets() -> list[str]:
    """Create a random list of subsets"""
    return random_features()  # By design, features and subsets should be equally random


def random_dataset(include_features: bool = True,
                   include_subsets: bool = False) -> \
                   Tuple[Dataset | DatasetDict, Optional[list[str]], Optional[list[str]]]:
    """Generate a random near "empty" dataset with random list of features and subsets, where
    the number of features, subsets and name-length are specified by NUM_FEATURES, NUM_SUBSETS
    and NAME_LENGTH, respectively.

    Args:
        include_features (bool, optional): Should the dataset contain features? Defaults to True.
        include_subsets (bool, optional): Should the dataset be spliced? Defaults to True.

    Returns:
        Tuple[Dataset | DatasetDict, list[str], list[str]]:
        The dataset and the corresponding features and subsets.
        Return a Dataset if subsets are not included. Otherwise, return a DatasetDict.
    """
    features = random_features() if include_features else None
    subsets = random_subsets() if include_subsets else None
    return create_dataset_shell(features=features, subsets=subsets), features, subsets
