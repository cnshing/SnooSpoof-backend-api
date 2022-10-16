"""
The underlying implementation that fetches our data is PRAW, which we assume is already tested.
Therefore we will only test everything else, including the type of data and expected behavior
of our scrapper.
"""
from types import SimpleNamespace
from collections.abc import Iterable
from random import sample, randrange
import unittest
from generate import scrapper
import praw

reddit = praw.Reddit('SnooSpoof')
reddit.read_only = True

submission_ids = ['t3_xtivpb', 't3_xtj9oh', 't3_xtjag2', 't3_xtjbwj']

comment_ids = ['t1_iqq5wy7', 't1_iqq5weo', 't1_iqq5uex', 't1_iqq5u6c',
               't1_iqq5ree', 't1_iqq5r2v', 't1_iqq5qxm', 't1_iqq5qc1',
               't1_iqq4mcw', 't1_iqq4j7g', 't1_iqq3xx7', 't1_iqq3wg6']

# Each parent id corresponds to a comment id in the same order of both lists
parent_ids = ['t1_iqq5weo', 't1_iqq5uex', 't1_iqq5u6c', 't3_xtjag2',
              't1_iqq5r2v', 't1_iqq5qxm', 't1_iqq5qc1', 't3_xtj9oh',
              't1_iqq4j7g', 't1_iqq3xx7', 't1_iqq3wg6', 't3_xtivpb']

USERNAME = 'SnooSpoof'
PRAW = scrapper.PRAWExtension(reddit)
search_submissions = PRAW.search_submissions
search_parents = PRAW.search_parents
search_comments = PRAW.search_comments


def random_sublist(lst: list) -> list:
    """Create a pseudo-random sublist of any list

    Args:
        lst (list): Any list

    Returns:
        (list): A sublist of lst
    """
    random_length = randrange(start=1, stop=len(lst))
    return sample(lst, random_length)


def random_submissions():
    """Select pseudo-randomly from a list of pre-selected submission ids

    Returns:
       (list): List of ids
    """
    return random_sublist(submission_ids)


def random_comments():
    """Select pseudo-randomly from a list of pre-selected comments ids

    Returns:
       (list): List of ids
    """
    return random_sublist(comment_ids)


def random_parents():
    """Select pseudo-randomly from a list of pre-selected parent ids

    Returns:
       (list): List of ids
    """
    return random_sublist(parent_ids)


class TestAuthorIdsCheck(unittest.TestCase):
    """Prevent confusing searches with authors or ids or when no information/reference
    is given to search from
    """

    def run_ValueErrorTest(self, search, **kwargs):
        """Attempt a search request excepting a ValueError"""
        with self.assertRaises(ValueError):
            search(**kwargs)

    """Test case when both authors and ids are supplied"""

    def test_parent_BothAuthorIds(self):
        self.run_ValueErrorTest(
            search_parents, parent_ids=submission_ids, author=USERNAME)

    def test_submission_BothAuthorIds(self):
        self.run_ValueErrorTest(
            search_submissions, ids=submission_ids, author=USERNAME)

    def test_comment_BothAuthorIds(self):
        self.run_ValueErrorTest(
            search_comments, ids=submission_ids, author=USERNAME)

    """Test case when no authors or ids are supplied"""

    def test_parent_NoneAuthorIds(self):
        self.run_ValueErrorTest(search_parents)

    def test_submission_NoneAuthorIds(self):
        self.run_ValueErrorTest(search_submissions)

    def test_comment_NoneAuthorIds(self):
        self.run_ValueErrorTest(search_comments)


class ValidFilter():
    """Allows us to compare our own filter with Python's own default and tested filter
    """

    def __init__(self, filter_fn, search_result: Iterable[dict[str, ]]):
        self.original = search_result
        self.filter_fn = filter_fn
        # Our function filter_fn utilizes the attribute namespace to access data,
        # but our search result utilizes the dictionary namespace. A conversion
        # is required.
        self.attributes = map(
            lambda unpack: SimpleNamespace(**unpack), search_result)

        # However, comparing the end filtered results are done in list-dictionary form,
        # so again a reconversion is neccesary.
        self.filtered = map(vars, filter(filter_fn, self.attributes))

    def __eq__(self, other):
        return all(valid == test for valid, test in zip(self.filtered, other))

    def __str__(self):
        """Aids in debugging. Not ideal to place debugging in production code,
        but since this function is very small and isolated, it shouldn't be
        a problem.
        """
        return str(list(self.filtered))


class TestFilter(unittest.TestCase):

    """Some functions to apply our filter"""

    def always_true(self, item):
        return True

    def always_false(self, item):
        return False

    def ids_containi(self, item):
        return 'i' in item.name

    def compare_timestamp(self, item):
        """Check to see if a creation date of an item is past a certain time-stamp"""
        chosen_time = 1664698700
        return item.created < chosen_time

    def run_FilterTest(self, filter_fn, **kawrgs):
        """Compares the filtered result from an search request agaisnt
        a manually filtered result

        Args:
            filter_fn (Callable[[], bool]): A function to filter
        """
        searches = [search_submissions, search_comments]
        random_ids = [random_submissions(), random_comments()]
        for search, ids in zip(searches, random_ids):
            # When our initial data is also fetched from the same search request,
            # any biases or errors from the request will carry over to both
            # filtered objects equally, resulting in a relatively unbiased comparsion
            posts = search(ids=ids, **kawrgs)
            filtered = search(filter_fn=filter_fn, ids=ids, **kawrgs)
            valid = ValidFilter(search_result=posts, filter_fn=filter_fn)
            self.assertEqual(valid, filtered)

    def test_always_true(self):
        self.run_FilterTest(self.always_true)

    def test_always_false(self):
        self.run_FilterTest(self.always_false)

    def test_ids_containi(self):
        self.run_FilterTest(self.ids_containi)

    def test_compare_timestamp(self):
        self.run_FilterTest(self.compare_timestamp)


if __name__ == '__main__':
    unittest.main()
