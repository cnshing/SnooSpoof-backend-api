"""
Responsible for combining every isolated component into one coherent piece for text generation
"""
from typing import Any
from copy import deepcopy
import praw
from transformers import PreTrainedTokenizer, PreTrainedModel
from datasets import Dataset, DatasetDict
import pandas
from parse.convert import dict2gentext, gentext2dict
from generate.scrapper import PRAWExtension
from generate.verify import DatasetDictModel
from generate.trainer import TrainerExtension
from generate.encoder import encode


class Generator:
    """Srapes, Encodes, and Trains the Dataset for text generation
    """

    def __init__(self, reddit: praw.reddit, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.parent_tags = {'submission': [
            'title', 'selftext'], 'comment': ['body']}
        self.submission_tags = ["subreddit", "is_self",
                                "is_original_content", "over_18", "title", "url", "selftext"]
        self.comment_tags = ["subreddit", "body"]
        self.scrapper = PRAWExtension(reddit=reddit)
        # Create a copy of the model because the model is mutated as it is trained
        self.trainer = TrainerExtension(
            model=deepcopy(model), tokenizer=tokenizer)

    def _fetch_data(self, username: str) -> list[dict[str, Any]]:
        """Fetches the raw data of a user.

        Args:
            username (str): A reddit username

        Returns:
            list[dict[str, Any]]: A list of posts accessible via key-value pairs
        """
        submissions = self.scrapper.search_submissions(
            author=username, tags=self.submission_tags)
        comments = self.scrapper.search_comments(
            author=username, tags=self.comment_tags+['parent_id'])
        parent_ids = [comment['parent_id'] for comment in comments]
        parents = self.scrapper.search_parents(
            parent_ids=parent_ids,
            submission_tags=self.parent_tags['submission'],
            comment_tags=self.parent_tags['comment'])
        for comment in comments:
            comment['parent'] = parents[comment['parent_id']]
        return submissions+comments

    def _create_dataset(self, data: list[dict[str, Any]]) -> Dataset:
        """Converts userdata into a Huggingface Dataset

        Args:
            data (list[dict[str, Any]]): Userdata from fetch_data()

        Returns:
            Dataset: A Dataset representation of data
        """
        dataframe = pandas.DataFrame(data)
        dataframe[["subreddit"]] = dataframe[["subreddit"]].astype(str)
        return Dataset.from_pandas(dataframe)

    def _check_dataset_trainable(self, dataset: DatasetDict):
        """Ensures the dataset has the right splits before it is passed
        for training
        """
        required_splits = ['train', 'test']
        DatasetDictModel(dataset=dataset, subsets=required_splits)

    def generate(self, username: str,
                 comments_only: bool,
                 is_original_content: bool,
                 over_18: bool, subreddit: str | None = None,
                 prompt: str | None = None) -> dict:
        """Based off the initial confirguation, generate the reddit impression.

        Args:
            username (str): A Reddit Username
            comments_only (bool): Should the text be a reply to a post?
            is_original_content (bool): Should the generator attempt original content posts?
            over_18 (bool): Should the generator attempt NSFW posts?
            subreddit (str | None, optional): An initial subreddit to guide generation.
            Defaults to None.
            prompt (str | None, optional): An initial text to guide generation. Defaults to None.

        Returns:
            dict: A dictionary containing the missing key-values:
            "subreddit", "prompt", and "response".
        """
        args = locals()
        features = ['is_original_content', 'over_18',
                    'post', 'subreddit', 'prompt', 'response']

        if not subreddit and prompt:
            # Swap'subreddit' and 'prompt' for unidirecitonal language models
            features[3], features[4] = features[4], features[3]

        args['post'] = 'comments' if comments_only else 'submission'

        missing_features = list(filter(lambda feature:
                                       feature not in args
                                       or
                                       args[feature] is None,
                                       features))

        raw_userdata = self._fetch_data(username=username)
        dataset = self._create_dataset(data=raw_userdata)
        encoded_dataset = encode(
            dataset=dataset, tags=features, tokenizer=self.trainer.tokenizer)
        self._check_dataset_trainable(dataset=encoded_dataset)
        self.trainer.train(encoded_dataset=encoded_dataset)
        input_text = dict2gentext(**{feature: args[feature] for feature in features
                                     if feature not in missing_features})
        generated_text = self.trainer.text(initial_text=input_text)
        results = gentext2dict(text=generated_text, tags=features)

        #Prompts that have been autofilled needs to also be sent back
        autofill_prompt = [feature for feature in ['prompt'] 
                                     if args[feature] != results[feature]]
        return {feature: results[feature] for feature in missing_features + autofill_prompt}
