"""
Ensure that generalized functions of encoder works
"""
from typing import Callable, Iterable
import unittest
from transformers import AutoTokenizer
from generate.encoder import requires, text_infilling_func
from parse.convert import from_infill, dict2gentext
from .test_dataset_utils import random_dataset, random_list

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
IGNORE_TEXT = ""
IGNORE_TAGS = []


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


class TestInfillFunctionality(unittest.TestCase):
    """Test that text infilling encoder behaves as expected
    """

    def test_func_types(self):
        """Test that the text infilling function properly returns correct accessible values"""
        text_values = text_infilling_func(
            tokenizer, IGNORE_TAGS, return_input_ids=False)

        self.assertTrue('text' in text_values(
            IGNORE_TEXT), msg="text_infilling values on text should be accessible via 'key'")
        input_values = text_infilling_func(
            tokenizer, IGNORE_TAGS, return_input_ids=True)
        self.assertTrue('input_ids' in input_values(
            IGNORE_TEXT), msg="text_infilling values on ids should be accessible via 'input ids")

    def test_seperator_logic(self):
        """Test that an tokenizer's existing sep token
        should error out when ununiquely used as the text infilling sep token"""
        with self.assertRaises(ValueError):
            text_infilling_func(tokenizer, IGNORE_TAGS,
                                unique_sep_token='[SEP]')

    def test_empty_tags(self):
        """Test that text with empty tags should still conform to the infilled text definition:
        text = input [sep] target
        Even when the inputs and targets are empty
        """
        tags = ['tag1', 'tag2', 'tag3', 'tag4']
        example = {tag: '' for tag in tags}
        valid_text = """tag1: 
tag2: 
tag3: 
tag4: [sep]"""
        test_func = text_infilling_func(tokenizer, tags)
        test_text = test_func(example)['text']
        self.assertEqual(valid_text, test_text)

    def test_zero_infill_probability(self):
        """Test that infilled text still conform to the infilled text definition:
        text = input [sep] target
        even when no infill occurs, resulting in no target values
        """
        random_values = random_list(str_len=40, up_to=10)
        tags = random_list(str_len=5, up_to=10)
        example = {tag: value for tag, value in zip(tags, random_values)}
        valid_text = dict2gentext(**example)+"[sep]"
        test_func = text_infilling_func(
            tokenizer, example.keys(), infill_probability=0)
        test_text = test_func(example)['text']
        self.assertEqual(valid_text, test_text)

    def runInfillEncodingTest(self, example: dict[str, str], num_tests: int):
        """Test that infilled text can succesfully be parsed back to the original valid text
        using an real generated example from the Notebook

        This test case has a depedency on "from_infill" to succeed correctly. Therefore, failure
        for this test case may be the result of any combination(from_infill,
        text_infilling_func, or both) of these components not working in tandem.

        Args:
            example (dict[str, str]): An dictionary representation of any valid texts
            num_tests (int): The total number of tests to run
        """
        tags = example.keys()
        test_func = text_infilling_func(tokenizer, tags)
        valid_text = dict2gentext(**example)
        for _ in range(num_tests):
            test_text = test_func(example)['text']
            infill_text = from_infill(test_text)
            self.assertEqual(valid_text, infill_text)

    def test_real_text(self):
        """Test a real example from the notebook
        """
        example = {'is_original_content': 'False',
                   'spoiler': 'False',
                   'over_18': 'False',
                   'edited': 'False',
                   'post': 'comment',
                   'subreddit': 'webcomics',
                   'prompt': 'What would you do?',
                   'url': 'None',
                   'response': "Why would I think that's a good idea? It would be like I'd be happy on a robot to kill a super human with one simple button but with a gun, people would assume that it's actually a bad idea. Just google that and people wouldn't believe it. Its not true that guns are bad, all they do is send them away. It's only true since guns destroy their bodies and the gun isn't just a bullet. If you tried to design a computer without a button, its just one reason that we were talking about guns. My guess would have been to use the guns to destroy the computer completely. I guess even making a completely different computer would force someone to think about something.\n\nTo be honest you can think of this as a problem for real life, when it comes to robots and we are still in the early stages of automation. You might not think this is the same as what happens. But I don't think guns actually do a big job and they just do something for fun. This would end my life (or at least, just like with most problems) because you probably need to keep guns for yourself and someone else."}
        self.runInfillEncodingTest(example, num_tests=500)


if __name__ == '__main__':
    unittest.main()
