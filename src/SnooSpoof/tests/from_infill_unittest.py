"""Test to ensure that from_infill correctly parses the infilled text.
"""
import unittest
from parse.convert import from_infill


class TestFromInfill(unittest.TestCase):
    """
    Commonly used parameters:
    infill_text (str): Text whose tokens are randomly seperated into blank and answer tokens
    valid_text (str): An equivalent correct representation of infill_text
    unique_sep_token (str): A seperator token used to seperate the blank and answer tokens
    """

    def runInfillTest(self, infill_text: str, valid_text: str, unqiue_sep_token: str = '[sep]'):
        """Test to see if valid_text is equivalent to the reconstructed infill text
        """
        test_text = from_infill(infill_text, unqiue_sep_token=unqiue_sep_token)
        self.assertEqual(valid_text, test_text)

    def runSepTest(self, infill_text: str, valid_text: str):
        """Having different seperators should result in the same behavior with
        """
        standard_sep = '[sep]'
        alternative_seps = ['(sep)', '#sep#', '[@@@]', 'I(*&^#@)']
        for sep in alternative_seps:
            alternative_infill_text = infill_text.replace(standard_sep, sep)
            self.runInfillTest(alternative_infill_text,
                               valid_text, unqiue_sep_token=sep)

    def test_document_text(self):
        """Test the infilled text that is referenced from the documentation
        """
        infill_text = """
tag1: str1
tag2: [blank tag2]
tag3: str3_1[blank tag3] str3_3
[sep]
str2[answer tag2] str3_2[answer tag3]
"""
        valid_text = """
tag1: str1
tag2: str2
tag3: str3_1 str3_2 str3_3
"""
        self.runInfillTest(infill_text, valid_text)
        self.runSepTest(infill_text, valid_text)

    def test_real_text(self):
        """Test an real example text generated from the notebook
        """
        valid_text = """is_original_content: False
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
        infill_text = """is_original_content: False
spoiler: False
over_18: False
edited: [blank edited]
post: comment
subreddit: webcomics
prompt: What would you do?
url: [blank url]
response: Why would I think that[blank response]s a [blank response] [blank response]? It would be like I'd be happy on a robot to [blank response] a super human with [blank response] [blank response] button but with a gun[blank response] [blank response] would assume that it's actually a bad idea[blank response] Just google that and people wouldn't believe it. Its not true that guns are bad, all they do is send them away. It's only true since guns destroy their [blank response] [blank response] the gun isn't just [blank response] bullet. If you tried to design a computer without a button, its just one reason that we were talking [blank response] guns. My guess would have [blank response] to use the guns [blank response] destroy the [blank response] completely. I guess even making [blank response] [blank response] different [blank response] would force someone to [blank response] about something.

[blank response] be honest you can think of this as a problem for [blank response] life, when it comes to robots [blank response] [blank response] are still [blank response] the early stages of automation. You might [blank response] think this is the same as what happens. But I don[blank response]t think guns actually do [blank response] big job and they just do something for [blank response]. This would end my life (or [blank response] least[blank response] just like [blank response] [blank response] [blank response][blank response] [blank response] you probably need to keep guns [blank response] yourself and someone else.
[sep]False[answer edited]None[answer url]'[answer response]good[answer response]idea[answer response]kill[answer response]one[answer response]simple[answer response],[answer response]people[answer response].[answer response]bodies[answer response]and[answer response]a[answer response]about[answer response]been[answer response]to[answer response]computer[answer response]a[answer response]completely[answer response]computer[answer response]think[answer response]To[answer response]real[answer response]and[answer response]we[answer response]in[answer response]not[answer response]'[answer response]a[answer response]fun[answer response]at[answer response],[answer response]with[answer response]most[answer response]problems[answer response])[answer response]because[answer response]for[answer response]
"""
        self.runInfillTest(infill_text, valid_text)
        self.runSepTest(infill_text, valid_text)


if __name__ == '__main__':
    unittest.main()
