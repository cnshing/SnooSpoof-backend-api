"""
Parsing component responsible for converting text generated results into JSON
"""
import unittest
import json
from collections import OrderedDict

#Constant names
IS_OC = "is_original_content"
SPOILER = "spoiler"
NSFW = "over_18"
EDIT = "edited"
TYPE = "post"
SUBR = "subreddit"
PROMPT = "prompt"
URL = "url"
RESP = "response"

class ValidJSON:
    """
    An abstracted class to make testing code more readable
    """
    def __init__(self, *tags):
        """"
        We expictely require our initial tags to be a list of tuples
        because the order of tags in text generation is significant,
        therefore our data representation must also account for order.
        """
        if tags is None:
            self.dict = OrderedDict()
        else:
            self.dict = OrderedDict( (tag, "") for tag in tags)

    def __getitem__(self, index):
        return self.dict[index]

    def __setitem__(self, index, value):
        self.dict[index] = value

    def set_tags(self, value, *argv):
        """Quickly set multiple tags to the same value

        Args:
            value (str): A value all the tags will take
        """
        if argv is not None:
            for tag in argv:
                self.dict[tag] = value

    def json(self):
        """Returns a JSON version of itself

        Returns:
            _type_: A JSON version of itself
        """
        return json.dumps(self.dict)

    def __str__(self):
        return self.json()



IS_OC = "is_original_content"
SPOILER = "spoiler"
NSFW = "over_18"
EDIT = "edited"
TYPE = "post"
SUBR = "subreddit"
PROMPT = "prompt"
URL = "url"
RESP = "response"

class TestGenText2JsonMethods(unittest.TestCase):
    """Ensures validity of text generation so output does not affect our front-end service upstream
    """
    def test_convert(self):
        """For now, just the JSON classes have been created to create test cases quicker.
        """
        valid_text = ValidJSON(IS_OC, SPOILER, NSFW, EDIT, TYPE, SUBR, PROMPT, URL, RESP)
        valid_text.set_tags("true", SPOILER, EDIT)
        valid_text.set_tags("false", IS_OC, NSFW)
        valid_text[TYPE] = "comment"
        valid_text[SUBR] = "reddit"
        valid_text[PROMPT] = "This is a standard prompt"
        valid_text[URL] = "None"
        valid_text[RESP] = "This is a standard response"


if __name__ == '__main__':
    unittest.main()
