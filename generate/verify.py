"""
Data validation for generation components
"""
from collections.abc import Iterable
import datasets
from pydantic import BaseModel, root_validator, validator


class DatasetModel(BaseModel):
    """A model to validate the features and subsets of a Huggingface Dataset.
    Currently only checks for the existence of certain splits

    Raises:
        KeyError: Whenever a explictely required feature/subset in our dataset does not exist
        AttributeError: Whenever the features/subsets of our dataset is needed but not accessible
    """
    dataset: datasets.Dataset
    features: Iterable[str] | None
    subsets: Iterable[str] | None

    class Config:
        arbitrary_types_allowed = True

    def _verify(self, expected_values: Iterable[str],
                actual_values: Iterable[str],
                error_msg: str):
        """Checks to see if there are any differences between our expected values
        and the actual values received.

        Args:
            expected_values (Iterable[str]): What is explictely required in our values
            actual_values (Iterable[str]): The values that were actually given
            error_msg (str): A f-string showing our error message.
            Let {missing} be what is missing between expected and actual values.

        Raises:
            KeyError: The expected values do not match our actual values
        """
        missing = set(expected_values) - set(actual_values)
        if missing:
            raise KeyError(error_msg.format(missing=missing))

    @root_validator(pre=True)
    def check_features_subsets(cls, values):
        """Whenever the features or splits must be valdiated, check to see if
        the features or splits can be accessible from our dataset.

        Raises:
            AttributeError: Features are required to be checked but not found in our dataset
            AttributeError: Splits are required to be checked but not found in our dataset
        """
        no_features = not values['dataset'].features
        no_subsets = not datasets.get_dataset_split_names(datasets)
        if no_features and values['features']:
            raise AttributeError("Features not accessible in our dataset")
        if no_subsets and values['subsets']:
            raise AttributeError("Subsets not accessible in our dataset")
        return values

    @validator("features")
    def verify_features(cls, value, values):
        """Verify that our explictly required features are in our dataset
        """
        features = values["dataset"].features.keys()
        self._verify(expected_values=value, actual_values=features,
                     error_msg="Features {missing} are missing")
        return value

    @validator('subsets')
    def verify_subsets(cls, value, values):
        """Verify that our split subset names are in our dataset
        """
        subsets = values["dataset"].keys()
        self._verify(expected_values=value, actual_values=subsets,
                     error_msg="Subsets {missing} are missing")
        return value
