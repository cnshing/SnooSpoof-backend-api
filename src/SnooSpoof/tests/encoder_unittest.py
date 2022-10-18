"""
Ensure that generalized functions of encoder works
"""
from typing import Callable, Iterable
import unittest
from tests.test_dataset_utils import random_dataset, random_list
from generate.encoder import requires


class TestRequiresFunctionality(unittest.TestCase):
    """Test the @requires decorator to ensure any mapping function performs
    consistently regardless if it was decorated or not.
    """

    # Sample functions to test mappings
    def identity(self, elem):
        return elem

    def concat(self, elem: str):
        return elem+"concat"

    def decoratorFunction(self, func: Callable,
                          features: Iterable[str] | None = None) -> Callable:
        """
        Functionally create a requires decorator function without
        @requires(features=features)

        Args:
            func (Callable): Any function for mapping
            features (Iterable[str], optional): Required features. Defaults to None.

        Returns:
            Callable: A @requires(feautres=features) function
        """
        # Assume requires(features=features)(func) = @requires(features=features) def func...
        required_func = requires(features=features)(func)
        return required_func

    def test_param_conflicts(self):
        """
        When a function uses a parameter "verify_dataset", the decorator cannot distinguish
        between the intent of calling function(verify_dataset=dataset) directly or whether to
        simply verify the dataset. This test ensures that the "verify_dataset" parameter
        is routed to the correct destination.
        """
        dataset, features, _ = random_dataset(
            include_features=True, include_subsets=False)

        def conflictParams(elem, verify_dataset=None):
            """The values of "verify_dataset" should pass through
            succesfully to this function regardless if "verify_dataset"
            is also a parameter to verify a dataset.
            """
            self.assertIsNotNone(verify_dataset)
            return elem

        required_func = self.decoratorFunction(
            conflictParams, features=features)
        required_func("elem", verify_dataset="value should pass through")
        required_func("elem", verify_dataset=dataset)

    def runMappingTest(self, func):
        """Test to ensure decorated function does not inadvertely affect the function values.
        In other words, test that the equivalences
        map(function(verify_dataset=dataset))
        =
        map(function)
        hold true.
        """
        dataset, features, _ = random_dataset(
            include_features=True, include_subsets=False)

        required_func = self.decoratorFunction(func, features=features)
        example_elements = random_list(str_len=10, up_to=10)

        values = set(map(func, example_elements))
        decorated_values = set(
            map(required_func(verify_dataset=dataset), example_elements))
        self.assertEqual(values, decorated_values,
                         msg="map(function(verify_dataset=dataset)) should equal to map(function)")

    def test_identity(self):
        self.runMappingTest(self.identity)

    def test_concat(self):
        self.runMappingTest(self.concat)


if __name__ == '__main__':
    unittest.main()
