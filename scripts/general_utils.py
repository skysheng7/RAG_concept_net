"""
General utility functions for PDF summarization across different AI providers.
Provides unified interface that calls provider-specific low-level functions.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Callable, Any

from anthropic_utils import (
    load_anthropic_client,
    upload_file_anthropic,
    summarize_file_anthropic,
)
from openai_utils import (
    load_openai_client,
    upload_file_openai,
    summarize_file_openai,
)
from ollama_utils import (
    load_ollama_client,
    summarize_file_ollama,
)

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

def summarize_pdfs_in_folder(
    provider: str,
    folder_path: str = "input_docs",
    prompt: str = "Please summarize this document.",
    model: Optional[str] = None,
    max_tokens: int = 50000,
    temperature: float = 1.0,
    output_folder: Optional[str] = "results",
    delay_seconds: int = 30,
) -> Dict[str, str]:
    """
    Iterate through all PDF files in a folder and summarize them using specified provider.

    Parameters
    ----------
    provider : str
        AI provider to use ('openai', 'anthropic', 'ollama', or 'google')
    folder_path : str, optional
        Path to folder containing PDF files (default: "input_docs")
    prompt : str, optional
        Prompt to guide the summarization (default: "Please summarize this document.")
    model : str or None, optional
        Model to use for summarization. If None, uses provider's default model
    max_tokens : int, optional
        Maximum tokens in response (default: 50000)
    temperature : float, optional
        Sampling temperature (default: 1.0)
    output_folder : str or None, optional
        Folder to save summary text files (default: "results").
        If None, summaries are not saved to disk.
    delay_seconds : int, optional
        Number of seconds to wait after each API call to avoid rate limits (default: 30)

    Returns
    -------
    Dict[str, str]
        Dictionary mapping PDF filenames to their summaries

    Raises
    ------
    ValueError
        If provider is not supported
    FileNotFoundError
        If the folder does not exist
    """
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Get provider-specific functions and default model
    provider = provider.lower()
    if provider == "openai":
        client = load_openai_client()
        upload_func = upload_file_openai
        summarize_func = summarize_file_openai
        default_model = "gpt-5.2-2025-12-11"
    elif provider == "anthropic":
        client = load_anthropic_client()
        upload_func = upload_file_anthropic
        summarize_func = summarize_file_anthropic
        default_model = "claude-sonnet-4-5-20250929"
    elif provider == "ollama":
        client = load_ollama_client()
        upload_func = None  # OLLAMA doesn't use file upload
        summarize_func = summarize_file_ollama
        default_model = "llama3"
    elif provider == "google":
        raise NotImplementedError("Google provider is not yet implemented")
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            "Supported providers are: 'openai', 'anthropic', 'ollama', 'google'"
        )

    # Use default model if not specified
    if model is None:
        model = default_model

    # Get all PDF files in folder
    pdf_files = list(Path(folder_path).glob("*.pdf"))

    if len(pdf_files) == 0:
        print(f"No PDF files found in {folder_path}")
        return {}

    # Create output folder if specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # Process each PDF
    summaries = {}

    for idx, pdf_path in enumerate(pdf_files):
        pdf_name = pdf_path.name

        try:
            # For OLLAMA, skip file upload and pass file path directly
            if provider == "ollama":
                summary = summarize_func(
                    client=client,
                    file_path=str(pdf_path),
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                )
            else:
                # Upload file
                file_id = upload_func(client, str(pdf_path))

                # Generate summary
                summary = summarize_func(
                    client=client,
                    file_id=file_id,
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            # Store summary
            summaries[pdf_name] = summary

            # Save to file if output folder specified
            if output_folder:
                output_path = os.path.join(
                    output_folder, pdf_name.replace(".pdf", "_summary.txt")
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(summary)

            # Wait before next API call (except for the last file)
            if delay_seconds > 0 and idx < len(pdf_files) - 1:
                time.sleep(delay_seconds)

        except Exception as e:
            summaries[pdf_name] = f"Error: {e}"
            # Still wait after an error to avoid rate limiting
            if delay_seconds > 0 and idx < len(pdf_files) - 1:
                time.sleep(delay_seconds)

    return summaries


def summarize_pdfs_by_index(
    provider: str,
    folder_path: str = "input_docs",
    start_index: int = 0,
    end_index: Optional[int] = None,
    prompt: str = "Please summarize this document.",
    model: Optional[str] = None,
    max_tokens: int = 50000,
    temperature: float = 1.0,
    output_folder: Optional[str] = "results",
    delay_seconds: int = 30,
) -> Dict[str, str]:
    """
    Summarize PDF files in a folder by index range using specified provider.

    Parameters
    ----------
    provider : str
        AI provider to use ('openai', 'anthropic', 'ollama', or 'google')
    folder_path : str, optional
        Path to folder containing PDF files (default: "input_docs")
    start_index : int, optional
        Starting index (0-based) of files to process (default: 0)
    end_index : int or None, optional
        Ending index (exclusive) of files to process.
        If None, processes to the end (default: None)
    prompt : str, optional
        Prompt to guide the summarization (default: "Please summarize this document.")
    model : str or None, optional
        Model to use for summarization. If None, uses provider's default model
    max_tokens : int, optional
        Maximum tokens in response (default: 50000)
    temperature : float, optional
        Sampling temperature (default: 1.0)
    output_folder : str or None, optional
        Folder to save summary text files (default: "results").
        If None, summaries are not saved to disk.
    delay_seconds : int, optional
        Number of seconds to wait after each API call to avoid rate limits (default: 30)

    Returns
    -------
    Dict[str, str]
        Dictionary mapping PDF filenames to their summaries

    Raises
    ------
    ValueError
        If provider is not supported or indices are invalid
    FileNotFoundError
        If the folder does not exist
    """
    # Check if folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Get provider-specific functions and default model
    provider = provider.lower()
    if provider == "openai":
        client = load_openai_client()
        upload_func = upload_file_openai
        summarize_func = summarize_file_openai
        default_model = "gpt-5.2-2025-12-11"
    elif provider == "anthropic":
        client = load_anthropic_client()
        upload_func = upload_file_anthropic
        summarize_func = summarize_file_anthropic
        default_model = "claude-sonnet-4-5-20250929"
    elif provider == "ollama":
        client = load_ollama_client()
        upload_func = None  # OLLAMA doesn't use file upload
        summarize_func = summarize_file_ollama
        default_model = "llama3"
    elif provider == "google":
        raise NotImplementedError("Google provider is not yet implemented")
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            "Supported providers are: 'openai', 'anthropic', 'ollama', 'google'"
        )

    # Use default model if not specified
    if model is None:
        model = default_model

    # Get all PDF files in folder
    pdf_files = sorted(list(Path(folder_path).glob("*.pdf")))

    if len(pdf_files) == 0:
        print(f"No PDF files found in {folder_path}")
        return {}

    # Validate indices
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    if end_index is not None and end_index <= start_index:
        raise ValueError("end_index must be greater than start_index")

    if start_index >= len(pdf_files):
        raise ValueError(
            f"start_index {start_index} is out of range (total files: {len(pdf_files)})"
        )

    # Select files by index
    if end_index is None:
        selected_files = pdf_files[start_index:]
    else:
        selected_files = pdf_files[start_index:end_index]

    # Create output folder if specified
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    # Process selected PDFs
    summaries = {}

    for idx, pdf_path in enumerate(selected_files):
        pdf_name = pdf_path.name

        try:
            # For OLLAMA, skip file upload and pass file path directly
            if provider == "ollama":
                summary = summarize_func(
                    client=client,
                    file_path=str(pdf_path),
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                )
            else:
                # Upload file
                file_id = upload_func(client, str(pdf_path))

                # Generate summary
                summary = summarize_func(
                    client=client,
                    file_id=file_id,
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            # Store summary
            summaries[pdf_name] = summary

            # Save to file if output folder specified
            if output_folder:
                output_path = os.path.join(
                    output_folder, pdf_name.replace(".pdf", "_summary.txt")
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(summary)

            # Wait before next API call (except for the last file)
            if delay_seconds > 0 and idx < len(selected_files) - 1:
                time.sleep(delay_seconds)

        except Exception as e:
            summaries[pdf_name] = f"Error: {e}"
            # Still wait after an error to avoid rate limiting
            if delay_seconds > 0 and idx < len(selected_files) - 1:
                time.sleep(delay_seconds)

    return summaries
