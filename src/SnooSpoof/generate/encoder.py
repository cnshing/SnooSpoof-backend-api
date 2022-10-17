"""
Encode the raw data of a username into a dataset suitable for training
"""
from inspect import signature
from typing import Callable
from collections.abc import Iterable
from random import random
from datasets import Dataset
from tokenizers import Tokenizer
from parse.util import line_delimited_text, special_tag_tokens
from generate import verify


def requires(features: Iterable[str]) -> Callable:
    """Decorator that verfies all features are
    within a Huggingface's Dataset before any
    mapping operations.

    Usage:
        Decorate any function used for Dataset.map with

        @requires(features)
        def function...

        and call the function with an additional
        "verify_dataset" parameter

        Dataset.map(function(verify_dataset=dataset))

        or implicitly via

        Dataset.map(function(dataset))

        to verify the dataset for valid features before mapping.

        Keep in mind that all existing functionality
        will still continue to proceed as normal, for example
        cases such as

        Dataset.map(function)

        will carry the map function without any verification.

    Args:
        features (Iterable[str]): Features expected in the Dataset
    """
    def decorator(function):
        def check_dataset(*args, verify_dataset=None, **kwargs):
            """When function(dataset) is called, the map operation
            is resolved as follows:

            Dataset.map(function(dataset)) => Dataset.map(function)

            with the side effect of verfiying our dataset.
            """
            dataset = verify_dataset
            valid_dataset = dataset is not None and isinstance(
                dataset, Dataset)
            verify_requested = args is None and kwargs is None

            if verify_requested and valid_dataset:
                verify.DatasetModel(dataset=dataset, features=features)
                return function

            # Functions with dataset as a keyword parameter must be reinserted
            param_conflict = signature(function).parameters['verify_dataset']
            if param_conflict:
                kwargs['verify_dataset'] = dataset

            return function(*args, **kwargs)
        return check_dataset
    return decorator


@requires(features=['body', 'is_self'])
def assign_types(example):
    """
    Create a 'post' column signfying the type of post the entry is.
    'link' - A link post
    'comment' - A comment
    'submission' - A submission
    """
    types = {lambda is_comment: example['body'] == "": 'comment',
             lambda is_submission: example['is_self'] is True: 'submission',
             lambda is_link: example['is_self'] is False: 'link'}
    for is_type, post in types.items():
        if is_type(example):
            return {'post': post}
    return example


@requires(features=['post'])
def remove_permalinks(example):
    """
    Remove the permalinks of the post. A permalink is an auto-generated url of
    linking to the reddit post. Unless the link contains relevant information that
    is semantically related to the text(a link post), permalinks are non relevant
    to text generation and should be removed.
    """
    if example['post'] != 'url':
        return {'url': None}
    return example


@requires(features=['post', 'parent'])
def create_prompt(example):
    """
    Create the prompt column.
    A comment's prompt should only contain relevant information of the parents.
    A submission's prompt should only be the title.
    """
    if example['post'] == 'comment':
        text = line_delimited_text(
            ["body", "title", "selftext"],
            lambda get: example['parent'][get],
            lambda valid: valid in example['parent'] and example['parent'][valid])
        return {'prompt': text}
    return {'prompt': example['title']}


@requires(features=['post', 'body', 'selftext'])
def create_response(example):
    """
    Create the response column.
    A response entry should only consist of the main text, comment or submission.
    """
    if example['post'] == 'comment':
        return {'response': example['body']}
    text = line_delimited_text(
        ["selftext"], lambda get: example[get], lambda valid: example[valid])
    return {'response': text}


@requires(features=['text'])
def keep_nondeleted_posts(example):
    """
    For any given entry, return whether or not any component of the text was
    removed or deleted in someway.
    Removed or deleted components pollute the quality of text generation.
    A majority deleted or removed posts in our dataset generates text that indicate
    the post was removed. Therefore removing these entries are advisable.
    """
    deleted_keywords = ["[deleted]", "[removed]"]
    for keyword in deleted_keywords:
        if keyword in example['text']:
            return False
    return True


def text_infilling_func(tokenizer: Tokenizer,
                        tags: Iterable[str],
                        unique_sep_token: str = '[sep]',
                        infill_probability: float = 0.15) -> Callable:
    """Create a text infilling function that randomly splices tokens
    and reconnects them with special answer and blank tokens.

    Given the following text:
    tag1: str1
    tag2: str2
    tag3: str3_1 str3_2 str3_3

    Create a function that manufactures a text infilling example:

    tag1: str1
    tag2: [blank tag2]
    tag3: str3_1 [blank tag3] str3_3
    [sep]
    str1 [answer tag1] str2 [answer tag2] str3_2 [answer tag3]

    where the str1, str2, and str3_2 are tokens determined by the tokenizer.

    Note that no maniuplation of the raw text data is done, this text infilling
    encoding/decoding issues(e.g: [blank tag] being decoded into subwords even though
    this tag is one single token). Lastly, the "tag:" metadata in our text will not
    be infilled.

    Args:
        tokenizer (Tokenizer): Any Huggingface Tokenizer that recognizes the each text
        as a single token:
            "tag: ", "[answer tag]", "[blank tag]", "\n"
        tags (Iterable[str]): Tags that correspond to a visual seperation of content in our text.
        unique_sep_token (str, optional): A seperation token used exclusively for text infilling.
        Defaults to '[sep]'.
        infill_probability (float, optional): Probability of applying our infill at each token.
        Defaults to 0.15.

    Raises:
        ValueError: When our seperation token is used in tokenizer
        TypeError: When any special tags(answer, blank, unqiue_sep_token, \n) is not a single token

    Returns:
        Callable: A Huggingface example mappable function
    """
    if tokenizer.sep_token == unique_sep_token:
        raise ValueError(f"Tokenizer seperation token should not be equal to \
        unique seperation token {unique_sep_token}")

    def token_to_id(token: str, as_list: bool = True) -> list[int] | int:
        """Retrieves the token id of a single token only.

        Args:
            token (str): A token_
            as_list (bool, optional): Should the id be encapsulated by a list? Defaults to True.

        Raises:
            TypeError: When the id is not one single token

        Returns:
            list[int] | int: A token id, optionally surrounded by a list.
        """
        single_id = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(single_id, int):
            raise TypeError(f"{single_id} should be a single token id, \
            suggesting {token} was not added to the tokenizer vocabulary")
        if as_list:
            return [single_id]
        return single_id

    # Add seperation token to our tokenizer
    sep = token_to_id(unique_sep_token)

    newline = token_to_id("\n")

    @requires(features=tags)
    def text_infilling(example):
        inputs = []
        target = []
        for tag, tag_format, answer_token, blank_token in special_tag_tokens(tags):

            # Initialize ids for answer, blank, tag, and text tokens
            answer_id, blank_id = token_to_id(
                answer_token), token_to_id(blank_token, as_list=False)
            tag_id = token_to_id(tag_format)
            tokens = tokenizer.tokenize(example[tag])
            token_ids = tokenizer.convert_tokens_to_ids(tokens)

            # Randomly mask out each id in our text
            for index, token_id in enumerate(token_ids):
                if random() < infill_probability:
                    # Move "token [answer tag]" to target
                    target += [token_id] + answer_id
                    # Replace current token with "[blank tag]"
                    token_ids[index] = blank_id
            inputs += tag_id+token_ids+newline  # Format text to tag: masked_str \n

        # Concatenate input [sep] target
        return {'input_ids': inputs+sep+target}
    return text_infilling


mappings = [remove_permalinks, create_prompt, create_response]
filters = [keep_nondeleted_posts]


def encode(dataset: Dataset,
           tokenizer: Tokenizer,
           tags: Iterable[str],
           test_size: float = 0.1) -> Dataset:
    """Encode a dataset for training.
    Specifically, build the final text and tokenize it for text-infilling.

    Args:
        dataset (Dataset): A Huggingface Dataset
        tokenizer (Tokenizer): Huggingface Tokenizer
        tags (Iterable[str]): Features of our dataset containing relevant
        information about the user's Dataset.
        test_size (float, optional): Porportion of train and test split. Defaults to 0.1.

    Returns:
        Dataset: A dataset ready for training.
    """
    encoded_dataset = dataset.map(assign_types)
    for apply in mappings:
        encoded_dataset = encoded_dataset.map(
            apply(verify_dataset=encoded_dataset))
    for filter_fn in filters:
        encoded_dataset = encoded_dataset.filter(
            filter_fn(verify_dataset=encoded_dataset))
    text_infilling = text_infilling_func(tokenizer, tags)
    encoded_dataset = encoded_dataset.map(
        text_infilling(verify_dataset=encoded_dataset))
    return encoded_dataset.train_test_split(test_size=test_size)
