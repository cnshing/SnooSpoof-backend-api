"""Train a dataset and output text
"""
from typing import Any, Optional
from collections.abc import Iterable
from transformers import (
    PreTrainedModel, PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer,
    pipeline
)
from datasets import DatasetDict


class TrainerExtension():
    """A very small wrapper of the Huggingface API to make training datasets more digestable
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast):
        self.model = model
        self.tokenizer = tokenizer

    def train(self, encoded_dataset: DatasetDict):
        """Trains the dataset.

        Since training a dataset for SnooSpoof does not require dynamic parameters,
        the Huggingface's train function is abstracted to simplify the training
        process.

        Args:
            encoded_dataset (DatasetDict): An encoded dataset
        """
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        training_args = TrainingArguments(
            output_dir="temp",
            evaluation_strategy="epoch",
            # Limit the total training iterations as much as possible
            # One epoch is enough for users with large post/comment history
            # Multiple epochs for users with very little post/comments will incur overfitting
            num_train_epochs=1,
            per_device_train_batch_size=1
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["test"],
            data_collator=data_collator
        )
        trainer.train()

    def _check_args(self, invalid_args: Iterable[str], kwargs: dict[str, Any]):
        """
        Args:
            invalid_args (Iterable[str]): An iterable of arguments
            kwargs (dict[str, Any]): A key word argument mapping

        Raises:
            ValueError: When an key word argument is invalid
        """
        for arg in invalid_args:
            if arg in kwargs:
                raise ValueError(
                    f"{arg} is an default argument and should not be overrided")

    def text(self,
             initial_text: str,
             pipeline_args: Optional[dict[str, Any]] = None,
             generator_args: Optional[dict[str, Any]] = None) -> str | Iterable[str]:
        """Generate text based off an initial text, and additional pipeline or
        generator args.

        The arguments ['model', 'task', 'tokenizer'], and ['inputs'] should not
        be used in pipeline_args or generator_args respectively as they are already
        defined during class initilization.

        An ValueError will occur if these arguments are ever referenced in pipeline_args
        and generator_args.

        Args:
            initial_text (str): Gives context on how the text should be filled.
            pipeline_args (Optional[dict[str,Any]], optional): Additional arguments
            for the pipeline object. Defaults to None.
            generator_args (Optional[dict[str,Any]], optional): Additional arguments
            for the generator object. Defaults to None.

        Returns:
            str | Iterable[str]: Batches of text if the arguments request multiple text,
            otherwise just a single text.
        """
        if pipeline_args is None:
            pipeline_args = {}

        if generator_args is None:
            generator_args = {}

        # Pipeline/Generator arguments should not override default parameters
        self._check_args(['model', 'task', 'tokenizer'], pipeline_args)
        self._check_args(['inputs'], generator_args)

        generator = pipeline(task='text-generation',
                             model=self.model,
                             tokenizer=self.tokenizer,
                             **pipeline_args)

        generated_text = generator(inputs=initial_text, **generator_args)
        return generated_text
