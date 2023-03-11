"""Passes requests for text generation and responds with the corresponding text
"""
from json import dumps
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # pytype: disable=import-error
import praw
from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from .generate import Generator

tokenizer = GPT2Tokenizer.from_pretrained(
    'gpt2', padding=False, truncation=True)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2',
                                        # Configure num_beams for proper memory allocation
                                        num_beams=1,
                                        max_length=1024,
                                        no_repeat_ngram_size=2)
reddit = praw.Reddit("SnooSpoof")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

class GenerationConfig(BaseModel):
    """Model containing information for how text should be generated
    """
    username: str
    subreddit: str | None = None
    prompt: str | None = None
    comments_only: bool
    is_original_content: bool
    over_18: bool


@app.get("/generate/")
def generate(config: GenerationConfig = Depends()):
    """Fill in the missing 'subreddit', 'prompt' and 'response'
    values of any generation request.

    Returns:
        str: An JSON object with any of the following keys:
        'subreddit', 'prompt', and 'response.
    """
    gen = Generator(reddit=reddit, model=model, tokenizer=tokenizer)
    return dumps(gen.generate(**config.dict()))
