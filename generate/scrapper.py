"""Allows collection of dataset to happen
"""
from collections.abc import Iterable, Callable
from operator import attrgetter
import praw


class PRAWExtension():
    """Abstracts the Python Reddit API Wrapper to allow for declarative
    searches for comments, submissions, and parents.
    """

    def __init__(self, reddit: praw.Reddit):
        self.reddit = reddit

    def _get(self, listing, attr: Iterable):
        """
        Get a a listing's attributes as a dictionary.
        """
        values = attrgetter(*attr)(listing)
        return dict(zip(attr, values))

    def _search(self,
                request: praw.models.ListingGenerator,
                attr: Iterable,
                filter_fn: Callable[[], bool]=lambda default: True):
        """Queries a listing for certain attributes. Elements of the listing that do not
        pass filter will not be saved.

        Args:
            request (praw.models.ListingGenerator): Any instances from (Ex. user.submissions.new)
            https://praw.readthedocs.io/en/latest/search.html?q=ListingGenerator
            attr (_type_): A iterable of attributes
            filter (func): Boolean function with an element of the listing as the single parameter
        """
        return [self._get(listing, attr) for listing in request if filter_fn(listing)]

    def _check_author_ids(self,
                        author: str = None,
                        ids: Iterable[str] = None):
        if author == ids == None:
            raise ValueError('Either an reddit username or ids be specified')
        elif author != ids != None:
            raise ValueError('Cannot search for both authors and ids at the same time')

    def is_comment(self, fullname):
        """Given a post's fullname, Return True if a post is a comment"""
        return fullname.startswith("t1")

    def is_submission(self, fullname):
        """Given a post's fullname, Return True if a post is a submission"""
        return fullname.startswith("t3")

    def search_submissions(self,
                        sort: str = 'new',
                        tags: Iterable[str] = None,
                        author: str = None,
                        ids: Iterable[str] = None,
                        filter_fn: Callable[[], bool]=lambda default: True,
                        **kawrgs):
        """Searches for submissions by IDs or author, returning the metadata in tags.
        If a filter_fn is specified, only submissions that pass the filter are saved.

        Args:
            sort (str, optional): Sorts by "hot", "new", etc for any author. Defaults to 'new'.
            tags (Iterable[str], optional): An iterable of metadata tags. Defaults to None.
            author (str, optional): A redditor's username. Defaults to None.
            ids (Iterable[str], optional): An iterable of submission ids. Defaults to None.
            filter (func): Boolean function with an element of the listing as the single parameter

        Returns:
            List[dict]: A list of submissions
        """
        self._check_author_ids(author, ids)

        if ids:
            submissions = self._search(
                self.reddit.info(fullnames=ids), attr=tags, filter_fn=filter_fn)
        else:
            user = self.reddit.redditor(author)
            listing = getattr(user.submissions, sort)
            submissions = self._search(listing(**kawrgs), attr=tags, filter_fn=filter_fn)
        return submissions

    def search_comments(self,
                    sort: str = 'new',
                    tags: Iterable[str] = None,
                    author: str = None,
                    ids: Iterable[str] = None,
                    filter_fn: Callable[[], bool]=lambda default: True,
                    **kawrgs):
        """Searches for comments by IDs or author. returning the metadata in tags.
        If a filter_fn is specified, only comments that pass the filter are saved.

        Args:
            sort (str, optional): Sorts by "hot", "new", etc for any author. Defaults to 'new'.
            tags (Iterable[str], optional): An iterable of metadata tags. Defaults to None.
            author (str, optional): A redditor's username. Defaults to None.
            ids (Iterable[str], optional): An iterable of submission ids. Defaults to None.
            filter (func): Boolean function with an element of the listing as the single parameter

        Returns:
            List[dict]: A list of comments
        """
        self._check_author_ids(author, ids)

        if ids:
            comments = self._search(
                self.reddit.info(fullnames=ids), attr=tags, filter_fn=filter_fn)
        else:
            user = self.reddit.redditor(author)
            listing = getattr(user.comments, sort)
            comments = self._search(listing(**kawrgs), attr=tags, filter_fn=filter_fn)
        return comments

    def seperate_posts(self, ids: Iterable[str]):
        """Seperate a list of ids into comment ids and submission ids

        Args:
            ids (Iterable[str]): A list of fullnames, characterized by a t3_ or t_1 prefix
        """
        def seperate(restriction):
            return list(filter(restriction, ids))
        return seperate(self.is_submission), seperate(self.is_comment)

    def search_parents(self,
                    author: str = None,
                    parent_ids: Iterable[str] = None,
                    submission_tags: Iterable[str] = None,
                    comment_tags:  Iterable[str] = None):
        """Search the parents of a comment by author or ID. When no IDs are given,
        search by the author's comments instead, following the same parameters
        as search_comments().

        Args:
            author (str, optional): A redditor's username. Defaults to None.
            parent_ids (Iterable[str], optional): An iterable of parent ids. Defaults to None.
            submission_tags (Iterable[str], optional): Metadata for submissions. Defaults to None.
            comment_tags (Iterable[str], optional): Metadata for comments. Defaults to None.

        Returns:
            dict: A dictionary of parents keyed by their unique parent id
        """
        self._check_author_ids(author, parent_ids)

        if parent_ids is None:
            parent_ids = [comment['parent_id']
                          for comment in self.search_comments(author, ['parent_id'])]

        submission_ids, comment_ids = self.seperate_posts(parent_ids)
        submissions = self.search_submissions(
            ids=submission_ids, tags=submission_tags)
        comments = self.search_comments(ids=comment_ids, tags=comment_tags)
        parents = {id: parent for id, parent in zip(
            submission_ids+comment_ids, submissions+comments)}
        return parents