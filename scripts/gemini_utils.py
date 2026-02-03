"""
Utility functions for connecting to Google Gemini API and processing PDF files.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


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


def summarize_file_gemini(
    client: genai.Client,
    pdf_path: str,
    prompt: str = "Please summarize this document.",
    model: str = "gemini-2.5-flash-lite",
    max_tokens: int = 50000,
    temperature: float = 1.0,
) -> str:
    """
    Summarize a PDF file using Gemini API.

    Parameters
    ----------
    client : genai.Client
        Initialized Gemini client
    pdf_path : str
        Path to the PDF file
    prompt : str, optional
        Prompt to guide the summarization (default: "Please summarize this document.")
    model : str, optional
        Model to use for summarization (default: "gemini-2.5-flash-lite")
    max_tokens : int, optional
        Maximum tokens in response (default: 50000)
    temperature : float, optional
        Sampling temperature (default: 1.0)

    Returns
    -------
    str
        Generated summary text
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    with open(pdf_path, "rb") as pdf_file:
        pdf_file_bytes = pdf_file.read()

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=pdf_file_bytes, mime_type="application/pdf"),
            prompt,
        ],
        config={
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        },
    )

    return response.text
