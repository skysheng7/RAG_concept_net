import os
import getpass
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def load_txt_files(file_path):
    """
    Loads a txt file into a list of documents.
    Args:
        file_path (str): The path to the txt file.
    Returns:
        list: A list of documents.
    """
    loader = TextLoader(file_path)

    docs = loader.load()
    return docs


def split_text(docs, chunk_size=2000, chunk_overlap=200, add_start_index=True):
    """
    Splits a list of documents into chunks.
    Args:
        docs (list): A list of documents.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.
        add_start_index (bool): Whether to add the start index to the metadata.
    Returns:
        list: A list of chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits


def create_vector_store(all_splits, batch_size=300, delay_seconds=2):
    """
    Create a vector store with rate limiting to avoid quota exhaustion.
    Args:
        all_splits (list): List of document splits to embed
        batch_size (int): Number of documents to process per batch
        delay_seconds (float): Delay between batches in seconds
    Returns:
        tuple: (vector_store, ids)
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        api_key = getpass.getpass("Enter API key for Google Gemini: ")

    # create embeddings with API key
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=api_key
    )
    # instantiate vector store using the embeddings
    vector_store = InMemoryVectorStore(embeddings)

    # Process documents in batches to respect rate limits
    all_ids = []
    total_batches = (len(all_splits) + batch_size - 1) // batch_size

    print(f"Processing {len(all_splits)} documents in {total_batches} batches...")

    for i in range(0, len(all_splits), batch_size):
        batch = all_splits[i : i + batch_size]
        batch_num = i // batch_size + 1

        print(
            f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)..."
        )

        try:
            ids = vector_store.add_documents(documents=batch)
            all_ids.extend(ids)
            print(f"✓ Batch {batch_num} completed successfully")

            # Add delay between batches to avoid rate limiting (except for the last batch)
            if i + batch_size < len(all_splits):
                print(f"Waiting {delay_seconds}s before next batch...")
                time.sleep(delay_seconds)

        except Exception as e:
            print(f"✗ Error in batch {batch_num}: {e}")
            # If rate limited, wait longer and retry
            if "429" in str(e) or "quota" in str(e).lower():
                print("Rate limit hit. Waiting 20 seconds before retrying...")
                time.sleep(20)
                # Retry this batch
                try:
                    ids = vector_store.add_documents(documents=batch)
                    all_ids.extend(ids)
                    print(f"✓ Batch {batch_num} completed on retry")
                except Exception as retry_error:
                    print(f"✗ Retry failed: {retry_error}")
                    raise
            else:
                raise

    return vector_store, all_ids


def build_retriver(vector_store, queries):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )

    retrived_docs = retriever.batch(
        queries,
    )

    return retrived_docs


if __name__ == "__main__":
    file_path = "input_data/concept_net.txt"
    docs = load_txt_files(file_path)

    all_splits = split_text(docs)
    print(f"Split into {len(all_splits)} chunks")

    vector_store, ids = create_vector_store(all_splits)
    print(f"\nSuccessfully created vector store with {len(ids)} documents")
