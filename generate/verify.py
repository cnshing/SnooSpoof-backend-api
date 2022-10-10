"""
Data validation for generation components
"""
from collections.abc import Iterable
from typing import Any
import datasets
from pydantic import BaseModel, validator

def _verify(expected_values: Iterable[str],
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

def should_verify(field:str, values: dict[str, Any]) -> bool:
    """Check to see if verficiation should be done based off
    the initial input values of our model

    Args:
        field (str): A field in BaseModel
        values (dict[str, Any]): The values parameter from a @validator function

    Returns:
        bool: Whether or not a validator should verify
    """
    return field in values and values[field]


class DatasetModel(BaseModel):
    """A model to validate the features of a Huggingface Dataset.
    Currently only checks for the existence of certain splits

    Raises:
        KeyError: Whenever a explictely required feature in our dataset does not exist
    """
    features: Iterable[str] | None = None
    dataset: datasets.Dataset

    class Config:
        arbitrary_types_allowed = True

    @validator("dataset")
    def verify_features(cls, value, values):
        """Verify that our explictly required features are in our dataset"""
        if should_verify("features", values):
            features = value.features.keys()
            _verify(expected_values=values['features'], actual_values=features,
                        error_msg="Features {missing} are missing")
        return value

class DatasetDictModel(BaseModel):
    """A Model to validate not only the features of every dataset in our dictionary but the
    split names.

    Raises:
        KeyError: Whenever an explicitly required subset in our dataset does not exist
        KeyError: Whenever any explicitly required feature for any of our datasets does not exist
    """
    features: Iterable[str] | None = None
    subsets: Iterable[str] | None = None
    dataset: datasets.DatasetDict

    class Config:
        arbitrary_types_allowed = True

    @validator("dataset")
    def verify_subsets(cls, value, values):
        """Verify that our split subset names are in our dataset
        """
        if should_verify('subsets', values):
            subsets = value.keys()
            _verify(expected_values=values['subsets'], actual_values=subsets,
                    error_msg="Subsets {missing} are missing")
        return value

    @validator('dataset')
    def verify_features(cls, value, values):
        """Verify that each dataset in our DatasectDict split has the correct features"""
        if should_verify("features", values):
            for sub_dataset in value.values():
                DatasetModel(dataset=sub_dataset, features=values['features'])
        return value
