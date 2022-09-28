"""Passes requests for text generation and responds with the corresponding text
"""
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

class GenerationConfig(BaseModel):
    """Model containing information for how text should be generated
    """
    username: str
    subreddit: str | None = None
    prompt: str | None = None
    comments_only: bool
    original_content: bool
    nsfw: bool
