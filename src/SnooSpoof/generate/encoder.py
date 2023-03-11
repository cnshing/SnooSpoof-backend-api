"""
Encode the raw data of a username into a dataset suitable for training
"""
from functools import wraps
from inspect import signature
from typing import Callable
from collections.abc import Iterable
from datasets import Dataset
from transformers import PreTrainedTokenizer
from parse.util import line_delimited_text, tag_format
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
        @wraps(function)
        def check_dataset(*args, verify_dataset=None, **kwargs):
            """When function(dataset) is called, the map operation
            is resolved as follows:

            Dataset.map(function(dataset)) => Dataset.map(function)

            with the side effect of verfiying our dataset.
            """
            dataset = verify_dataset
            valid_dataset = dataset is not None and isinstance(
                dataset, Dataset)
            verify_requested = not args and not kwargs
            if verify_requested and valid_dataset:
                verify.DatasetModel(dataset=dataset, features=features)
                return function

            # Functions with verify_dataset as a keyword parameter must be reinserted
            param_conflict = 'verify_dataset' in signature(function).parameters
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
    types = {lambda is_comment: example['body'] is not None: 'comment',
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
    if example['post'] != 'link':
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
    A response entry should only consist of the main "body" of the post.
    Link posts do not have a main body, therefore their responses
    will be empty.
    """
    responses = {'comment': example['body'],
                 'link': '', 'submission': example['selftext']}
    for response_type, response in responses.items():
        if example['post'] == response_type:
            return {'response': response}
    return example


@requires(features=['prompt', 'response'])
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
        prompt = example['prompt']
        response = example['response']
        if keyword in prompt or keyword in response:
            return False
    return True


def create_text_func(tags: Iterable[str]) -> Callable:
    """Create a functon that concatenates each tag in tags into
    the following text:

    "
    tag1: str1
    tag2: str2
    tag3: str3_1 str3_2 str3_3
    "
    where "str1", "str2", "str3_1 str3_2 str3_3" are retrivable
    via example['tag1'], example['tag2'], example['tag3']

    Args:
        tags (Iterable[str]): Tags that correspond to a visual seperation of content in our text.

    Returns:
        Callable: A Huggingface example mappable function
    """

    @requires(features=tags)
    def create_text(example):
        text = line_delimited_text(tags,
                                   lambda tag: tag_format(
                                       tag) + str(example[tag]),
                                   lambda default: True)
        return {'text': text}
    return create_text


def encode(dataset: Dataset,
           tokenizer: PreTrainedTokenizer,
           tags: Iterable[str],
           test_size: float = 0.1) -> Dataset:
    """Encode a dataset for training.
    Specifically, build the final text and tokenize it for text-infilling.

    Args:
        dataset (Dataset): A Huggingface Dataset
        tokenizer (PreTrainedTokenizer): Any Huggingface PreTrained Tokenizer
        tags (Iterable[str]): Features of our dataset containing relevant
        information about the user's Dataset.
        test_size (float, optional): Porportion of train and test split. Defaults to 0.1.

    Returns:
        Dataset: A dataset ready for training.
    """

    @requires(features=['text'])
    def tokenize(example):
        """Just tokenizes example['text']"""

        #Truncation must be enabled as examples exceeding the max length will fail to train.
        #Under expected behavior, a tokenizer pre-initialized with truncation enabled should
        #by default apply truncation without explicit specification. However, by design this
        #does not apply. See
        #https://github.com/huggingface/transformers/issues/14033
        #for more information.
        return tokenizer(example['text'], truncation=True)

    mappings = [remove_permalinks, create_prompt,
                create_response, create_text_func(tags=tags), tokenize]
    filters = [keep_nondeleted_posts]

    encoded_dataset = dataset.map(assign_types)
    for apply in mappings:
        encoded_dataset = encoded_dataset.map(
            apply(verify_dataset=encoded_dataset))
    for filter_fn in filters:
        encoded_dataset = encoded_dataset.filter(
            filter_fn(verify_dataset=encoded_dataset))
    return encoded_dataset.train_test_split(test_size=test_size)
