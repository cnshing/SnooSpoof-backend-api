"""
Miscellaneous utilities for parsing
"""
from typing import Iterable, Tuple, Callable, Generator


def special_tag_tokens(tags: Iterable[str]) -> Generator[Tuple[str, str, str, str], None, None]:
    """Yields the relevant tag tokens for text infilling.

    "tag": The tag title
    "tag: ": An seperator of each tag with a semi-colin and space
    "[answer tag]": The answer token of tag
    "[blank tag]": The blank token of tag

    Args:
        tags (Iterable[str]): An Iterable of tags

    Yields:
        Generator[str, str, str, str]:  In the following order:
        "tag", "tag: ", "[answer tag]", "[blank tag]"
    """
    for tag in tags:
        tag_format = tag + ": "  # Text assumes each tag content is seperated by "tag: "
        answer_token = f"[answer {tag}]"
        blank_token = f"[blank {tag}]"
        yield tag, tag_format, answer_token, blank_token


def line_delimited_text(tags: Iterable[str],
                        get: Callable[[str], str],
                        valid: Callable[[str], bool]) -> str:
    """
    Returns the following text:
    "
    tag_1: get(tag_i)
    tag_3: get(tag_3)
    tag_n: get(tag_n)
    "
    for every tag in tags where get(tag) is the resulting string from calling tag on get,
    and only if the tag is considered "valid" via the valid function.

    Args:
        tags (Iterable[str]): An Interable of tags
        get (Callable[str, str]): Retrieves the tag information
        valid (Callable[str, bool]): Checks to see if tag should be part of our text

    Returns:
        str: A line delimited text of tags
    """

    return "\n".join([f"{get(tag)}" for tag in tags if valid(tag)])
