"""
Test custom validator logic for DatasetModel and DatasetDictModel
"""
import unittest
from generate import verify
from .test_dataset_utils import (
    random_features, random_subsets, create_dataset_shell, random_dataset
)

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
