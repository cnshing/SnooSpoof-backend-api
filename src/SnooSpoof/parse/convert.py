"""
Converts any object to another equivalent form
"""
from json import dumps, loads
from collections.abc import Iterable
from re import finditer
from .util import line_delimited_text, answer_token, blank_token


def gentext2dict(text: str, tags: Iterable[str]) -> dict[str, str]:
    """Parses a generated text into a dictionary where the text is

    "tag1: str1
    tag2: str2
    tag3: str3
    "
    such that tag1, tag2, tag3 are elements of tags

    into the following key-value pairs

    {
        "tag1": "str1",
        "tag2": "str2",
        "tag3": "str3"
    }

    Throws an error if an tag does not exist or is not in correct order
    Args:
        text (str): Generated text delimited by some tags and a colin
        tags (list[str]: A ordered list of tags
    """
    convert = {}
    indices = list(map(text.find, tags))

    # The index positions just after the tag and semi colon
    starting = map(lambda index, tag: index+len(tag)+1, indices, tags)

    # The index positions just before the end of the text
    ending = indices[1:] + [len(text)]

    # The resulting variables are the starting and ending positions
    # of all the tags which can extracted into each key-value pair
    for tag, start, end in zip(tags, starting, ending):

        if start > end:
            raise IndexError(f"{tag} is not in the correct order")
        # Our text generation adds a semicolin and a newline that makes
        # the text more readable but isn't a true representation of our text
        skip_semicolon = trim_newline = 1
        convert[tag] = text[start+skip_semicolon:end-trim_newline]

    return convert


def dict2gentext(**kwargs: str) -> str:
    """Converts our dictionary of keyword arguments

    {
        "tag1": "str1",
        "tag2": "str2",
        "tag3": "str3"
    }

    such that tag1, tag2, tag3 are the arguments

    into the following text:
    "tag1: str1
    tag2: str2
    tag3: str3
    "
    """
    return line_delimited_text(kwargs,
                               lambda tag: f"{tag}: {kwargs[tag]}",
                               lambda default: True)


def json2gentext(json: str) -> str:
    """Converts a JSON string represenation of key-value pairs

    "{
        "tag1": "str1",
        "tag2": "str2",
        "tag3": "str3"
    }"

    such that tag1, tag2, tag3 are the arguments

    into the following text:
    "tag1: str1
    tag2: str2
    tag3: str3
    """
    return dict2gentext(**loads(json))


def from_infill(text: str, unqiue_sep_token: str = '[sep]') -> str:
    """Reconstructs a infilled text example.

    Given the following text:

    tag1: str1
    tag2: [blank tag2]
    tag3: str3_1[blank tag3] str3_3
    [sep]
    str2[answer tag2] str3_2[answer tag3]

    Connects back each answer token to the corresponding answer token:

    tag1: str1
    tag2: str2
    tag3: str3_1 str3_2 str3_3

    Args:
        text (str): Text that has been randomly infilled
        unqiue_sep_token (str, optional): A seperation token used exclusively for text infilling.
        Defaults to '[sep]'.

    Returns:
        str: Generated text delimited by some tags and a colin
    """

    # When the separator token is not unique(occurs more than once),
    # A ValueError occurs as there will be excessive values to unpack
    inputs, target = tuple(text.split(sep=unqiue_sep_token, maxsplit=2))

    extract_target = r'(?P<token>.*?)'+answer_token(as_regex=True)

    # Extract each token, [answer tag], and replace corresponding [blank tag] with token
    for match in finditer(extract_target, target):
        token = match['token']
        answer_tag = match['answer_tag']
        inputs = inputs.replace(blank_token(answer_tag), token, 1)
    return inputs
