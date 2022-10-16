"""
Parsing component responsible for converting text generated results into JSON
"""
import unittest
import json
from collections import OrderedDict
from parse.convert import gentext2json


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

class TestGenText2JsonMethods(unittest.TestCase):
    """Ensures validity of text generation so output does not affect our front-end service upstream
    """
    def test_simple(self):
        """Simple test where the tested text is not too long and follows what the syntax we expect
        """
        test_text = """
is_original_content: false
spoiler: true
over_18: false
edited: true
post: comment
subreddit: reddit
prompt: This is a standard prompt
url: None
response: This is a standard response
"""
        tags = [IS_OC, SPOILER, NSFW, EDIT, TYPE, SUBR, PROMPT, URL, RESP]
        valid_text = ValidJSON(IS_OC, SPOILER, NSFW, EDIT, TYPE, SUBR, PROMPT, URL, RESP)
        valid_text.set_tags("true", SPOILER, EDIT)
        valid_text.set_tags("false", IS_OC, NSFW)
        valid_text[TYPE] = "comment"
        valid_text[SUBR] = "reddit"
        valid_text[PROMPT] = "This is a standard prompt"
        valid_text[URL] = "None"
        valid_text[RESP] = "This is a standard response"

        test_json = gentext2json(test_text, tags)
        self.assertEqual(test_json, valid_text.json(), "Simple Convert")

    def test_catch_unordered_exception(self):
        """Test scenario where just one of the tags is misordered
        """
        tags = [IS_OC, SPOILER, NSFW, EDIT, TYPE, SUBR, PROMPT, URL, RESP]
        test_text = """
is_original_content: false
over_18: false
spoiler: true
edited: true
post: comment
subreddit: reddit
prompt: This is a standard prompt
url: None
response: This is a standard response
"""
        with self.assertRaises(IndexError,
        msg="over_18 and spoiler is swapped, thus the order is mismatched"):
            gentext2json(test_text, tags)

    def test_real_text(self):
        """Test an real example text generated from the notebook
        """
        tags = [IS_OC, SPOILER, NSFW, EDIT, TYPE, SUBR, PROMPT, URL, RESP]
        test_text = """
is_original_content: False
spoiler: False
over_18: False
edited: False
post: comment
subreddit: webcomics
prompt: What would you do?
url: None
response: Why would I think that's a good idea? It would be like I'd be happy on a robot to kill a super human with one simple button but with a gun, people would assume that it's actually a bad idea. Just google that and people wouldn't believe it. Its not true that guns are bad, all they do is send them away. It's only true since guns destroy their bodies and the gun isn't just a bullet. If you tried to design a computer without a button, its just one reason that we were talking about guns. My guess would have been to use the guns to destroy the computer completely. I guess even making a completely different computer would force someone to think about something.

To be honest you can think of this as a problem for real life, when it comes to robots and we are still in the early stages of automation. You might not think this is the same as what happens. But I don't think guns actually do a big job and they just do something for fun. This would end my life (or at least, just like with most problems) because you probably need to keep guns for yourself and someone else.
"""
        valid_text = ValidJSON(IS_OC, SPOILER, NSFW, EDIT, TYPE, SUBR, PROMPT, URL, RESP)
        valid_text.set_tags("False", IS_OC, SPOILER, NSFW, EDIT)
        valid_text[TYPE] = "comment"
        valid_text[SUBR] = "webcomics"
        valid_text[PROMPT] = "What would you do?"
        valid_text[URL] = "None"
        valid_text[RESP] = """Why would I think that's a good idea? It would be like I'd be happy on a robot to kill a super human with one simple button but with a gun, people would assume that it's actually a bad idea. Just google that and people wouldn't believe it. Its not true that guns are bad, all they do is send them away. It's only true since guns destroy their bodies and the gun isn't just a bullet. If you tried to design a computer without a button, its just one reason that we were talking about guns. My guess would have been to use the guns to destroy the computer completely. I guess even making a completely different computer would force someone to think about something.

To be honest you can think of this as a problem for real life, when it comes to robots and we are still in the early stages of automation. You might not think this is the same as what happens. But I don't think guns actually do a big job and they just do something for fun. This would end my life (or at least, just like with most problems) because you probably need to keep guns for yourself and someone else."""
        test_json = gentext2json(test_text, tags)
        self.assertEqual(test_json, valid_text.json(), "Real text example")


if __name__ == '__main__':
    unittest.main()
