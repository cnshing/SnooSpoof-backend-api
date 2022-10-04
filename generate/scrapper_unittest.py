"""
The underlying implementation that fetches our data is PRAW, which we assume is already tested.
Therefore we will only test everything else, including the type of data and expected behavior
of our scrapper. 
"""
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
    random_length = 0
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
    def __init__(self, filter_fn, iterable: Iterable):
        self.original = iterable
        self.filter_fn = filter_fn
        
    def __eq__(self, other):
        filtered = filter(self.filter_fn, self.original)
        return all(valid == test for valid, test in zip(filtered, other))

    def __str__(self):
        return str(list(filter(self.filter_fn, self.original)))

class TestFilter(unittest.TestCase):

    
    def alwaysTrue(self, item):
        return True
    
    def alwaysFalse(self, item):
        return False
    
    def run_filterTest(self, filter_fn, **kawrgs):
        searches = [search_submissions, search_comments]
        ids = [selectRandomSubmissions(), selectRandomComments()]
        for search, id in zip(searches, ids):  
            print(id)
            posts = search(ids=id, **kawrgs)
            filtered = search(filter_fn=filter_fn, ids=id, **kawrgs)
            valid = ValidFilter(filter_fn=filter_fn, iterable=posts)
            self.assertEqual(valid, filtered)
            print(valid)
            print(filtered)


    def test_alwaysTrue(self):
        self.run_filterTest(self.alwaysTrue, tags=['fullname'])

if __name__ == '__main__':
    unittest.main()