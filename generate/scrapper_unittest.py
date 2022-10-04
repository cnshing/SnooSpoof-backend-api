"""
The underlying implementation that fetches our data is PRAW, which we assume is already tested.
Therefore we will only test everything else, including the type of data and expected behavior
of our scrapper. 
"""
from types import SimpleNamespace
from collections.abc import Iterable
from random import sample, randrange
import unittest
import scrapper
import praw
reddit = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="",
)
reddit.read_only = True
import scrapper
submission_ids = ['t3_xtivpb', 't3_xtj9oh', 't3_xtjag2', 't3_xtjbwj']
comment_ids = ['t1_iqq5wy7', 't1_iqq5weo', 't1_iqq5uex', 't1_iqq5u6c', 't1_iqq5ree', 't1_iqq5r2v', 't1_iqq5qxm', 't1_iqq5qc1', 't1_iqq4mcw', 't1_iqq4j7g', 't1_iqq3xx7', 't1_iqq3wg6']
parent_ids = ['t1_iqq5weo', 't1_iqq5uex', 't1_iqq5u6c', 't3_xtjag2', 't1_iqq5r2v', 't1_iqq5qxm', 't1_iqq5qc1', 't3_xtj9oh', 't1_iqq4j7g', 't1_iqq3xx7', 't1_iqq3wg6', 't3_xtivpb']
username = 'SnooSpoof'
PRAW = scrapper.PRAWExtension(reddit)
search_submissions = PRAW.search_submissions
search_parents = PRAW.search_parents
search_comments = PRAW.search_comments

def randomSublist(lst: list):
    random_length = randrange(start=1, stop=len(lst))
    return sample(lst, random_length)


def selectRandomSubmissions():
    return randomSublist(submission_ids)

def selectRandomComments():
    return randomSublist(comment_ids)

def selectRandomParents():
    return randomSublist(parent_ids)

class TestAuthorIdsCheck(unittest.TestCase):

    def run_ValueErrorTest(self, search, **kwargs):
        with self.assertRaises(ValueError):
            search(**kwargs)

    

    def test_parent_BothAuthorIds(self):
        self.run_ValueErrorTest(search_parents, parent_ids=submission_ids, author=username)

    def test_submission_BothAuthorIds(self):
        self.run_ValueErrorTest(search_submissions, ids=submission_ids, author=username)

    def test_comment_BothAuthorIds(self):
        self.run_ValueErrorTest(search_comments, ids=submission_ids, author=username)

    def test_parent_NoneAuthorIds(self):
        self.run_ValueErrorTest(search_parents)

    def test_submission_NoneAuthorIds(self):
        self.run_ValueErrorTest(search_submissions)

    def test_comment_NoneAuthorIds(self):
        self.run_ValueErrorTest(search_comments)

class ValidFilter():
    def __init__(self, filter_fn, search_result: Iterable[dict[str,]]):
        self.original = search_result
        self.filter_fn = filter_fn
        #We convert our iterable into its own namespace as our functions in filter_fn
        #also use attributes to compare object data
        self.attributes = map(lambda unpack: SimpleNamespace(**unpack), search_result)

        #Even though the lambda function uses attributes, our end compared results are
        #dictionaries, and so a reconvert to its list-dictionary form is neccesary
        self.filtered = map(vars, filter(filter_fn, self.attributes))
        
    def __eq__(self, other):
        return all(valid == test for valid, test in zip(self.filtered, other))

    def __str__(self):
        return str(list(self.filtered))

class TestFilter(unittest.TestCase):

    
    def alwaysTrue(self, item):
        return True
    
    def alwaysFalse(self, item):
        return False
    
    def idsContainLetteri(self, item):
        return 'i' in item.name

    def beforeCertainTime(self, item):
        chosen_time = 1664698700
        return item.created < chosen_time

    def run_filterTest(self, filter_fn, **kawrgs):
        searches = [search_submissions, search_comments]
        ids = [selectRandomSubmissions(), selectRandomComments()]
        for search, id in zip(searches, ids): 
            posts = search(ids=id, **kawrgs)
            filtered = search(filter_fn=filter_fn, ids=id, **kawrgs)
            valid = ValidFilter(search_result=posts, filter_fn=filter_fn)
            self.assertEqual(valid, filtered)



    def test_alwaysTrue(self):
        self.run_filterTest(self.alwaysTrue)

    def test_alwaysFalse(self):
       self.run_filterTest(self.alwaysFalse)

    def test_idsContainLetteri(self):
        self.run_filterTest(self.idsContainLetteri)
    
    def test_beforeCertainTime(self):
        self.run_filterTest(self.beforeCertainTime)

if __name__ == '__main__':
    unittest.main()