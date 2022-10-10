"""
Test custom validator logic for DatasetModel and DatasetDictModel
"""
from collections.abc import Iterable
from random import choice, randint
from string import ascii_lowercase, digits
from typing import Tuple, Optional
from datasets import Dataset, DatasetDict
import unittest
import verify



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


class TestDatasetModel(unittest.TestCase):
    """Test the validation logic of our DatasetModel
    """

    def test_none_features(self):
        """Features should not be verified if no features are given
        """
        dataset, _, _ = random_dataset()
        try:
            verify.DatasetModel(dataset=dataset)
            verify.DatasetModel(dataset=dataset, features=None)
        except KeyError:
            self.fail("Error raised despite feature error checking was disabled")

    def test_empty_features(self):
        """A dataset with no features should give an error on anything other than
        an empty featureset
        """
        featureless, _, _, = random_dataset(include_features=False)
        with self.assertRaises(KeyError):
            verify.DatasetModel(dataset=featureless,
                                features=random_features())

    def test_missing_features(self):
        """A dataset with explicitly incorrect features should always given an error
        """
        dataset, features, _ = random_dataset(include_features=True)
        missing_features = set(random_features()) - set(features)
        with self.assertRaises(KeyError):
            verify.DatasetModel(dataset=dataset, features=missing_features)

    def test_valid_features(self):
        """A dataset with exactly matching features should always go through
        """
        featureless, _, _, = random_dataset(include_features=False)
        dataset, features, _ = random_dataset(include_features=True)
        try:
            verify.DatasetModel(dataset=featureless, features=[])
            verify.DatasetModel(dataset=dataset, features=features)
        except KeyError:
            self.fail(
                "Mismatched features error raised despite inputting identical features")


class DatasetDictModel(unittest.TestCase):
    """Test the validation logic of our DatasetDict Model
    """

    def test_none_features_subsets(self):
        """Features or Subsets should not be verified if no features are given
        """
        dataset, _, _ = random_dataset(include_subsets=True)
        try:
            verify.DatasetDictModel(dataset=dataset)
            verify.DatasetDictModel(dataset=dataset, features=None)
            verify.DatasetDictModel(dataset=dataset, subsets=None)
            verify.DatasetDictModel(
                dataset=dataset, features=None, subsets=None)
        except KeyError:
            self.fail("Error raised despite feature error checking was disabled")

    def test_empty_features_subsets(self):
        """An empty dataset with no features or subsets should always error
        with any non-empty features/subsets
        """
        empty = create_dataset_shell(features=[], subsets=[])
        features, subsets = random_features(), random_subsets()
        with self.assertRaises(KeyError):
            verify.DatasetDictModel(dataset=empty, features=features)
            verify.DatasetDictModel(dataset=empty, subsets=subsets)
            verify.DatasetDictModel(
                dataset=empty, features=features, subsets=subsets)

    def test_missing_features_subsets(self):
        """Explicitly designed datasets with missing features/subsets should result
        in an error
        """
        dataset, features, subsets = random_dataset(
            include_features=True, include_subsets=True)
        missing_features = set(random_features()) - set(features)
        missing_subsets = set(random_subsets()) - set(subsets)
        with self.assertRaises(KeyError):
            verify.DatasetDictModel(dataset=dataset, features=missing_features)
            verify.DatasetDictModel(dataset=dataset, subsets=missing_subsets)
            verify.DatasetDictModel(dataset=dataset, features=missing_features,
                                    subsets=missing_subsets)

    def test_valid_features_subsets(self):
        """Datasets that have identical features/subsets as required in our model should not
        result in an error
        """
        empty = create_dataset_shell(features=[], subsets=[])
        dataset, features, subsets = random_dataset(
            include_features=True, include_subsets=True)
        try:
            verify.DatasetDictModel(dataset=empty, features=[])
            verify.DatasetDictModel(dataset=empty, subsets=[])
            verify.DatasetDictModel(dataset=empty, features=[], subsets=[])
            verify.DatasetDictModel(dataset=dataset, features=features)
            verify.DatasetDictModel(dataset=dataset, subsets=subsets)
            verify.DatasetDictModel(
                dataset=dataset, features=features, subsets=subsets)
        except KeyError:
            self.fail(
                "Mismatched features error raised despite inputting identical features")


if __name__ == '__main__':
    unittest.main()
