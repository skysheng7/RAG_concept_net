from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from general_utils import load_gemini_client

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


def split_text(docs, chunk_size=1000, chunk_overlap=200, add_start_index=True):
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
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=add_start_index
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def create_vector_store(all_splits):
    # load gemini API client
    load_gemini_client()
    # create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    # instantiate vector store using the embeddings
    vector_store = InMemoryVectorStore(embeddings)
    # index the documents
    ids = vector_store.add_documents(documents=all_splits)

if __name__ == "__main__":
    file_path = "input_data/concept_net.txt"
    docs = load_txt_files(file_path)
    all_splits = split_text(docs)
    print(len(all_splits))