"""
Ensure that generalized functions of encoder works
"""
from typing import Callable, Iterable
import unittest
import pandas
from datasets import Dataset
from generate.encoder import (
    requires,
    assign_types, remove_permalinks, create_prompt, create_response, keep_nondeleted_posts
)
from .test_dataset_utils import random_dataset, random_list


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

    def runMappingInvariantTest(self, func):
        """Test to ensure decorated function does not inadvertely affect the function values.
        In other words, test that the invariant
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
        self.runMappingInvariantTest(self.identity)

    def test_concat(self):
        self.runMappingInvariantTest(self.concat)


class TestEncoderExampleFunctions(unittest.TestCase):
    """Test that the example functions map/filter out to the expected values
    """

    # A base dataset for the functions to map on
    userdata = [
        {'is_original_content': False,
         'is_self': False,
         'over_18': False,
         'selftext': '',
         'subreddit': 'SnooSpoof',
         'title': 'Scrapper UnitTest 4',
         'url': 'http://link.post'},
        {'is_original_content': False,
         'is_self': True,
         'over_18': False,
         'selftext': 'Sample data for Scrapper UnitTest 3\n\nText submission',
         'subreddit': 'SnooSpoof',
         'title': 'Scrapper UnitTest 3',
         'url': 'https://www.reddit.com/r/SnooSpoof/comments/xtjag2/scrapper_unittest_3/'},
        {'is_original_content': False,
         'is_self': True,
         'over_18': False,
         'selftext': ' \n\nSample data for Scrapper UnitTest 2\n\nText submission',
         'subreddit': 'SnooSpoof',
         'title': 'Scrapper UnitTest 2',
         'url': 'https://www.reddit.com/r/SnooSpoof/comments/xtj9oh/scrapper_unittest_2/'},
        {'is_original_content': False,
         'is_self': True,
         'over_18': False,
         'selftext': 'Sample data for Scrapper UnitTest 1\n\nText submission',
         'subreddit': 'SnooSpoof',
         'title': 'Scrapper UnitTest 1',
         'url': 'https://www.reddit.com/r/SnooSpoof/comments/xtivpb/scrapper_unittest_1/'},
        {'body': 'Super Nested Comment for Scrapper UnitTest 3\r'
                 '  \n'
                 '\r'
                 '  \n'
                 'PARENT -> "Nested Comment for Scrapper UnitTest 3"',
         'parent': {'selftext': 'Sample data for Scrapper UnitTest 3\n'
                                '\n'
                                'Text submission',
                    'title': 'Scrapper UnitTest 3'},
         'parent_id': 't1_iqq5weo',
         'subreddit': 'SnooSpoof'},
        {'body': 'Nested Comment for Scrapper UnitTest 3\r'
                 '  \n'
                 '\r'
                 '  \n'
                 'PARENT -> "Comment for Scrapper UnitTest 3"',
         'parent': {'selftext': ' \n'
                                '\n'
                                'Sample data for Scrapper UnitTest 2\n'
                                '\n'
                                'Text submission',
                    'title': 'Scrapper UnitTest 2'},
         'parent_id': 't1_iqq5uex',
         'subreddit': 'SnooSpoof'},
        {'body': 'Comment for Scrapper UnitTest 3\n'
                 '\n'
                 'PARENT -> "Top-level comment for Scrapper UnitTest 3"',
         'parent': {'selftext': 'Sample data for Scrapper UnitTest 1\n'
                                '\n'
                                'Text submission',
                    'title': 'Scrapper UnitTest 1'},
         'parent_id': 't1_iqq5u6c',
         'subreddit': 'SnooSpoof'},
        {'body': 'Top-level comment for Scrapper UnitTest 3',
         'parent': {'body': 'Nested Comment for Scrapper UnitTest 3\r'
                            '  \n'
                            '\r'
                            '  \n'
                            'PARENT -> "Comment for Scrapper UnitTest 3"'},
         'parent_id': 't3_xtjag2',
         'subreddit': 'SnooSpoof'},
        {'body': 'Super Nested Comment for Scrapper UnitTest 2\n'
                 '\n'
                 'PARENT -> "Nested Comment for Scrapper UnitTest 2"',
         'parent': {'body': 'Comment for Scrapper UnitTest 3\n'
                            '\n'
                            'PARENT -> "Top-level comment for Scrapper UnitTest 3"'},
         'parent_id': 't1_iqq5r2v',
         'subreddit': 'SnooSpoof'},
        {'body': 'Nested Comment for Scrapper UnitTest 2\n'
                 '\n'
                 'PARENT -> "Comment for Scrapper UnitTest 2"',
         'parent': {'body': 'Top-level comment for Scrapper UnitTest 3'},
         'parent_id': 't1_iqq5qxm',
         'subreddit': 'SnooSpoof'},
        {'body': 'Comment for Scrapper UnitTest 2\n'
                 '\n'
                 'PARENT -> "Top-level comment for Scrapper UnitTest 2"',
                 'parent': {'body': 'Nested Comment for Scrapper UnitTest 2\n'
                            '\n'
                            'PARENT -> "Comment for Scrapper UnitTest 2"'},
         'parent_id': 't1_iqq5qc1',
         'subreddit': 'SnooSpoof'},
        {'body': 'Top-level comment for Scrapper UnitTest 2',
         'parent': {'body': 'Comment for Scrapper UnitTest 2\n'
                            '\n'
                            'PARENT -> "Top-level comment for Scrapper UnitTest 2"'},
         'parent_id': 't3_xtj9oh',
         'subreddit': 'SnooSpoof'},
        {'body': 'Super Nested Comment for Scrapper UnitTest 1\n'
                 '\n'
                 'PARENT -> "Nested Comment for Scrapper UnitTest1"',
         'parent': {'body': 'Top-level comment for Scrapper UnitTest 2'},
         'parent_id': 't1_iqq4j7g',
         'subreddit': 'SnooSpoof'},
        {'body': 'Nested Comment for Scrapper UnitTest 1\n'
                 '\n'
                 'PARENT -> "Comment for Scrapper UnitTest1"',
         'parent': {'body': 'Nested Comment for Scrapper UnitTest 1\n'
                            '\n'
                            'PARENT -> "Comment for Scrapper UnitTest1"'},
         'parent_id': 't1_iqq3xx7',
         'subreddit': 'SnooSpoof'},
        {'body': 'Comment for Scrapper UnitTest1\n'
                 '\n'
                 'PARENT -> "Top-level comment for Scrapper UnitTest 1"',
         'parent': {'body': 'Comment for Scrapper UnitTest1\n'
                            '\n'
                            'PARENT -> "Top-level comment for Scrapper UnitTest 1"'},
         'parent_id': 't1_iqq3wg6',
                 'subreddit': 'SnooSpoof'},
        {'body': 'Top-level comment for Scrapper UnitTest 1',
         'parent': {'body': 'Top-level comment for Scrapper UnitTest 1'},
         'parent_id': 't3_xtivpb',
         'subreddit': 'SnooSpoof'}
    ]

    # Userdata that has been removed or deleted
    deleted_userdata = [
        {'is_original_content': False,
         'is_self': True,
         'over_18': False,
         'selftext': '[removed]',
         'subreddit': 'SnooSpoof',
         'title': 'Encoder Deleted Post UnitTest 1',
         'url': 'https://www.reddit.com/r/SnooSpoof/comments/yczloi/encoder_deleted_post_unittest_1/'},
        {'body': 'Original Submission Data:\n'
                 '\n'
                 '"Encoder Deleted Post UnitTest 1  \n'
                 'A post that has been deleted or removed should not be in our '
                 'dataset  \n'
                 'Text Submission\n'
                 '\n'
                 '"',
         'parent': {'selftext': '[removed]',
                    'title': 'Encoder Deleted Post UnitTest 1'},
         'parent_id': 't3_yczloi',
         'subreddit': 'SnooSpoof'},
        {'body': 'Comment for Scrapper UnitTest 3\n'
                 '\n'
                 'PARENT -> "Top-level comment for Scrapper UnitTest 3"',
         'parent': {'body': '[deleted]'},
         'parent_id': 't1_iqq5u6c',
         'subreddit': 'SnooSpoof'}
    ]

    # The mapped values of each function in the same order as userdata+deleted_userdata
    corresponding_encoding = [
        {'post': 'link',
         'prompt': 'Scrapper UnitTest 4',
         'response': '',
         'url': 'http://link.post'},
        {'post': 'submission',
         'prompt': 'Scrapper UnitTest 3',
         'response': 'Sample data for Scrapper UnitTest 3\n\nText submission',
         'url': None},
        {'post': 'submission',
         'prompt': 'Scrapper UnitTest 2',
         'response': ' \n\nSample data for Scrapper UnitTest 2\n\nText submission',
         'url': None},
        {'post': 'submission',
         'prompt': 'Scrapper UnitTest 1',
         'response': 'Sample data for Scrapper UnitTest 1\n\nText submission',
         'url': None},
        {'post': 'comment',
         'prompt': 'Scrapper UnitTest 3\n'
                   'Sample data for Scrapper UnitTest 3\n'
                   '\n'
                   'Text submission',
         'response': 'Super Nested Comment for Scrapper UnitTest 3\r'
                   '  \n'
                   '\r'
                   '  \n'
                   'PARENT -> "Nested Comment for Scrapper UnitTest 3"',
         'url': None},
        {'post': 'comment',
         'prompt': 'Scrapper UnitTest 2\n'
                   ' \n'
                   '\n'
                   'Sample data for Scrapper UnitTest 2\n'
                   '\n'
                   'Text submission',
         'response': 'Nested Comment for Scrapper UnitTest 3\r'
                   '  \n'
                   '\r'
                   '  \n'
                   'PARENT -> "Comment for Scrapper UnitTest 3"',
         'url': None},
        {'post': 'comment',
         'prompt': 'Scrapper UnitTest 1\n'
                   'Sample data for Scrapper UnitTest 1\n'
                   '\n'
                   'Text submission',
         'response': 'Comment for Scrapper UnitTest 3\n'
                   '\n'
         'PARENT -> "Top-level comment for Scrapper UnitTest 3"',
         'url': None},
        {'post': 'comment',
         'prompt': 'Nested Comment for Scrapper UnitTest 3\r'
                   '  \n'
                   '\r'
                   '  \n'
                   'PARENT -> "Comment for Scrapper UnitTest 3"',
         'response': 'Top-level comment for Scrapper UnitTest 3',
         'url': None},
        {'post': 'comment',
         'prompt': 'Comment for Scrapper UnitTest 3\n'
                   '\n'
                   'PARENT -> "Top-level comment for Scrapper UnitTest 3"',
         'response': 'Super Nested Comment for Scrapper UnitTest 2\n'
                   '\n'
                   'PARENT -> "Nested Comment for Scrapper UnitTest 2"',
         'url': None},
        {'post': 'comment',
         'prompt': 'Top-level comment for Scrapper UnitTest 3',
         'response': 'Nested Comment for Scrapper UnitTest 2\n'
                   '\n'
                   'PARENT -> "Comment for Scrapper UnitTest 2"',
         'url': None},
        {'post': 'comment',
         'prompt': 'Nested Comment for Scrapper UnitTest 2\n'
                   '\n'
                   'PARENT -> "Comment for Scrapper UnitTest 2"',
         'response': 'Comment for Scrapper UnitTest 2\n'
                   '\n'
                   'PARENT -> "Top-level comment for Scrapper UnitTest 2"',
         'url': None},
        {'post': 'comment',
         'prompt': 'Comment for Scrapper UnitTest 2\n'
                   '\n'
                   'PARENT -> "Top-level comment for Scrapper UnitTest 2"',
         'response': 'Top-level comment for Scrapper UnitTest 2',
         'url': None},
        {'post': 'comment',
         'prompt': 'Top-level comment for Scrapper UnitTest 2',
         'response': 'Super Nested Comment for Scrapper UnitTest 1\n'
                   '\n'
                   'PARENT -> "Nested Comment for Scrapper UnitTest1"',
         'url': None},
        {'post': 'comment',
         'prompt': 'Nested Comment for Scrapper UnitTest 1\n'
                   '\n'
                   'PARENT -> "Comment for Scrapper UnitTest1"',
         'response': 'Nested Comment for Scrapper UnitTest 1\n'
                   '\n'
                   'PARENT -> "Comment for Scrapper UnitTest1"',
         'url': None},
        {'post': 'comment',
         'prompt': 'Comment for Scrapper UnitTest1\n'
                   '\n'
                   'PARENT -> "Top-level comment for Scrapper UnitTest 1"',
         'response': 'Comment for Scrapper UnitTest1\n'
                   '\n'
                   'PARENT -> "Top-level comment for Scrapper UnitTest 1"',
         'url': None},
        {'post': 'comment',
         'prompt': 'Top-level comment for Scrapper UnitTest 1',
         'response': 'Top-level comment for Scrapper UnitTest 1',
         'url': None},
        {'post': 'submission',
         'prompt': 'Encoder Deleted Post UnitTest 1',
         'response': '[removed]',
         'url': None},
        {'post': 'comment',
         'prompt': 'Encoder Deleted Post UnitTest 1\n'
                   '[removed]',
         'response': 'Original Submission Data:\n'
                   '\n'
                   '"Encoder Deleted Post UnitTest 1  \n'
                   'A post that has been deleted or removed should not be in our '
                   'dataset  \n'
                   'Text Submission\n'
                   '\n'
                   '"',
         'url': None},
        {'post': 'comment',
         'prompt': '[deleted]',
         'response': 'Comment for Scrapper UnitTest 3\n'
                   '\n'
                   'PARENT -> "Top-level comment for Scrapper UnitTest 3"',
         'url': None}
    ]

    dataset = Dataset.from_pandas(pandas.DataFrame(userdata+deleted_userdata))

    def maps(self, dataset: Dataset, functions: Iterable[Callable]) -> Dataset:
        """Maps a dataset multiple times

        Args:
            dataset (Dataset): Any Huggingface Dataset
            functions (Iterable[Callable]): An iterable of Huggingface
            example functions

        Returns:
            (Dataset): A mapped Dataset
        """
        encoded_dataset = dataset
        for function in functions:
            encoded_dataset = encoded_dataset.map(function)
        return encoded_dataset

    def runMappingTest(self, functions: Iterable[Callable], features: Iterable[str]):
        """Runs a mapping on the test dataset and see if its values
        match to what is expected in corresponding_encodings

        Args:
            functions (Iterable[Callable]): An iterable of Huggingface
            example functions
            features (Iterable[str]): Features both in corresponding_encodings
            and the Dataset that will be evaluated
        """
        encoded_dataset = self.maps(self.dataset, functions)
        function_names = list(
            map(lambda function: function.__name__, functions))
        for data, encode in zip(encoded_dataset, self.corresponding_encoding):
            for feature in features:
                self.assertEqual(data[feature], encode[feature],
                                 msg=f"{function_names} for \'{feature}\'\
                                 created an invalid value.\n\
                                 The correct encoding should be: \n{encode}\n")

    def test_assign_types(self):
        self.runMappingTest(functions=[assign_types], features=['post'])

    def test_remove_permalinks(self):
        self.runMappingTest(
            functions=[assign_types, remove_permalinks], features=['url'])

    def test_create_prompt(self):
        self.runMappingTest(
            functions=[assign_types, create_prompt], features=['prompt'])

    def test_create_response(self):
        self.runMappingTest(
            functions=[assign_types, create_response], features=['response'])

    def test_keep_nondeleted_posts(self):
        """Test to ensure keep_nondeleted_posts correctly filters out the removed/deleted
        datapoints
        """
        functions = [assign_types, create_prompt, create_response]

        test_dataset = self.maps(self.dataset, functions)
        test_dataset = test_dataset.filter(keep_nondeleted_posts)

        invalid_dataset = Dataset.from_pandas(
            pandas.DataFrame(self.deleted_userdata))
        invalid_dataset = self.maps(invalid_dataset, functions)

        for invalid_data in invalid_dataset:
            self.assertNotIn(invalid_data, test_dataset,
                             msg="Filtered data is unexpectedly in the dataset")


if __name__ == '__main__':
    unittest.main()
