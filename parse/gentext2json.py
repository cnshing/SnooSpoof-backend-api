"""
Responsible for converting a text generated result into JSON
"""

def convert(text, tags):
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
        json[tag] = text[start:end]

    return json
