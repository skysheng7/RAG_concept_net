import os
import getpass
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma


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


def split_text_by_character(docs, chunk_size=2000, chunk_overlap=200, add_start_index=True):
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

def split_text_by_newline(docs, chunk_size=1000, chunk_overlap=0, add_start_index=True):
    """
    Splits a list of documents into chunks by newlines.
    Perfect for line-based data like ConceptNet where each line is a fact.
    Args:
        docs (list): A list of documents.
        chunk_size (int): The size of each chunk in characters.
        chunk_overlap (int): The overlap between chunks.
        add_start_index (bool): Whether to add the start index to the metadata.
    Returns:
        list: A list of chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
        separators=["\n\n", "\n"],  # Prioritize splitting by newlines
        is_separator_regex=False,
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def split_text_by_sentences(docs, chunk_size=1000):
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
    # Ensure NLTK punkt_tab data is available
    try:
        import nltk
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading required NLTK data (punkt_tab)...")
        import nltk
        nltk.download('punkt_tab')
    
    text_splitter = NLTKTextSplitter(chunk_size=chunk_size)
    all_splits = text_splitter.split_documents(docs)
    return all_splits


def create_vector_store(all_splits, batch_size=300):
    """
    Create a vector store with rate limiting to avoid quota exhaustion.
    Args:
        all_splits (list): List of document splits to embed
        batch_size (int): Number of documents to process per batch
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
        model="gemini-embedding-001", google_api_key=api_key
    )
    # instantiate vector store using the embeddings
    #vector_store = InMemoryVectorStore(embeddings)
    vector_store = Chroma(
        collection_name="concept_net",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )
    
    # count the total number of splits
    total_splits = len(all_splits)
    cur_split = 0
    ids = []
    print(f"Total splits: {total_splits}")
    
    while cur_split < total_splits:
        if (cur_split + batch_size) >= total_splits:
            batch = all_splits[cur_split:]
            print(f"Last batch!")
        else:
            batch = all_splits[cur_split: cur_split+batch_size]
            print(f"Processing {cur_split} to {cur_split+batch_size}")
        
        try: 
            cur_ids = vector_store.add_documents(documents=batch)
            ids.extend(cur_ids)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print("Rate limit hit, waiting 20 seconds before retrying...")
                time.sleep(20)
                
                cur_ids = vector_store.add_documents(documents=batch)
                ids.extend(cur_ids)
            
        cur_split = cur_split + batch_size
    
    return vector_store
    
    
def build_retriver(vector_store, queries, k = 1):
    """
    Build a retriever for the vector store.

    Args:
        vector_store (InMemoryVectorStore): The vector store to build the retriever from
        queries (list): The queries to build the retriever from
        k (int, optional): The number of most similar documents to return. Defaults to 1.

    Returns:
        list: The retrieved documents
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}, # only include the top 1 most similar document
    )

    retrived_docs = retriever.batch(
        queries,
    )

    return retrived_docs

def rag_pipeline(file_path, split_by="newline", chunk_size=10, chunk_overlap=200, add_start_index=True, batch_processing_size=300):
    """
    Build a RAG pipeline to load, split, and vectorize documents.
    
    Args:
        file_path (str): Path to the text file
        split_by (str): How to split the text. Options: "newline", "character", "sentence"
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks (not used for sentence splitting)
        add_start_index (bool): Whether to add start index to metadata
        batch_processing_size (int): Batch size for processing embeddings
    
    Returns:
        vector_store: The created vector store
    """
    docs = load_txt_files(file_path)
    
    if split_by == "newline":
        all_splits = split_text_by_newline(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index)
    elif split_by == "sentence":
        all_splits = split_text_by_sentences(docs, chunk_size=chunk_size)
    else:  # character
        all_splits = split_text_by_character(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index)
    
    vector_store = create_vector_store(all_splits, batch_size=batch_processing_size)
    
    return vector_store

    
