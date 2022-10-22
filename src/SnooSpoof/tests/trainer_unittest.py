"""
Test the trainer component logically behaves as expected
"""
import unittest
from transformers import BertTokenizerFast, AutoModel
from generate.trainer import TrainerExtension


class TestTrainerExtension(unittest.TestCase):
    """Most of the Trainer functionality is already provided out-of-box with Huggingface.
    This unit test is mainly used to check for any other manual implementations not
    in the well-tested library.
    """

    def test_text_invalid_args(self):
        """Test that any invalid arguments in pipeline_arg and generator_arg
        are correctly detected
        """
        ignore_tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-uncased")
        ignore_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        trainer = TrainerExtension(
            model=ignore_model, tokenizer=ignore_tokenizer)

        ignore_value = None
        ignore_text = ""
        invalid_pipeline_args = {
            'model': ignore_value,
            'tokenizer': ignore_value,
            'task': ignore_value
        }
        invalid_generator_args = {
            'inputs': ignore_value
        }

        with self.assertRaises(ValueError):
            # 'model', 'tokenizer', and 'task' should not be in pipeline_args
            trainer.text(initial_text=ignore_text,
                         pipeline_args=invalid_pipeline_args)

            # 'inputs' should not be in generator_args
            trainer.text(initial_text=ignore_text,
                         generator_args=invalid_generator_args)


if __name__ == '__main__':
    unittest.main()
