"""Train a dataset and output text
"""
from copy import deepcopy
from transformers import (
    PreTrainedModel, PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer,
    TextGenerationPipeline
)
from datasets import DatasetDict

class TrainerExtension():
    """A very small wrapper of the Huggingface API to make training datasets more digestable
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast):
        self.model = model
        self._weights = deepcopy(model.state_dict())
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
            per_device_train_batch_size=1,

            # These are the optimization steps neccesary to succesfully run training
            # on my own 4 GB VRAM machine. In production, remove or adjust these arguments
            # as needed.
            gradient_accumulation_steps=4,
            gradient_checkpointing=True, #This sets use_cache=False for now
            optim='adafactor',
            per_device_eval_batch_size=1,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["test"],
            data_collator=data_collator
        )
        trainer.train()

    def reset_model(self):
        """Reset the model weights back to it's pretrained values.
        """
        self.model.load_state_dict(self._weights)

    def text(self,
             initial_text: str) -> str | list[str]:
        """Generate text based off an initial text.

        Adjusting the text parameters is done by following the configuration of
        self.model.

        Args:
            initial_text (str): Gives context on how the text should be filled.

        Returns:
            str | list[str]: Batches of text if the arguments request multiple text,
            otherwise just a single text.
        """
        generator = TextGenerationPipeline(model=self.model,
                                           tokenizer=self.tokenizer,
                                           device='cuda:0')

        results = generator(initial_text)

        generated_text = [result['generated_text'] for result in results]

        # To prevent previous finetuning from affecting future text generation,
        # reset the model weights.
        self.reset_model()

        if len(generated_text) == 1:
            return generated_text[0]
        return generated_text
