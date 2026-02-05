"""
General utility functions for PDF summarization across different AI providers.
Provides unified interface that calls provider-specific low-level functions.
"""

import os
from dotenv import load_dotenv
from google import genai

def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text content from a txt file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Reading a txt file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    return text


def load_gemini_client() -> genai.Client:
    """
    Load Gemini client with API key from environment variables.

    Returns
    -------
    genai.Client
        Initialized Gemini client

    Raises
    ------
    ValueError
        If GEMINI_API_KEY is not found in environment variables
    """
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment variables. "
            "Please create a .env file with your API key."
        )

    client = genai.Client(api_key=api_key)
    return client

 