"""
Responsible for converting a text generated result into JSON
"""
from json import dumps
from collections.abc import Iterable

def convert(text: str, tags: Iterable[str]) -> str:
    """Parses a generated text into a JSON "object" where the text is

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
        text (str): Generated text delimited some sort of tags and a colon :
        tags (list[str]: A ordered list of tags
    """
    json = {}
    indices = list(map(text.find, tags))

    #The index positions just after the tag and semi colon
    starting = map(lambda index, tag: index+len(tag)+1, indices, tags)

    #The index positions just before the end of the text
    ending = indices[1:] + [len(text)]

    #The resulting variables are the starting and ending positions
    # of all the tags which can extracted into each key-value pair
    for tag, start, end in zip(tags, starting, ending):

        if start > end:
            raise IndexError(f"{tag} is not in the correct order")
        #Our text generation adds a semicolin and a newline that makes
        #the text more readable but isn't a true representation of our text
        skip_semicolon = trim_newline = 1
        json[tag] = text[start+skip_semicolon:end-trim_newline]

    return dumps(json)
