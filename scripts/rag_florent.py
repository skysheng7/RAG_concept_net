import time
import json
import os
import getpass
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


## Embedding models
def setup_gemini_embeddings(model_name="models/embedding-001"):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter API key for Google Gemini: ")
    print(f"Loading Gemini embeddings model: {model_name}...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=api_key,
    )
    print("✓ Embeddings model loaded successfully")
    return embeddings

def setup_minilm_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print(f"Loading HuggingFace embeddings model: {model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("✓ Embeddings model loaded successfully")
    return embeddings

def get_embeddings(embedding_model="gemini", model_name=None):
    if embedding_model == "gemini":
        return setup_gemini_embeddings(model_name or "models/embedding-001")
    if embedding_model == "minilm":
        return setup_minilm_embeddings(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError("Unsupported embedding_model. Use 'gemini' or 'minilm'.")


## Load text file
def load_text_file(file_path: str):
    # Load text file into documents
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Text file not found: {abs_path}")
    try:
        loader = TextLoader(abs_path, encoding="utf-8", autodetect_encoding=True)
        documents = loader.load()
        return documents
    except Exception as exc:
        raise RuntimeError(f"Error loading text file: {abs_path}") from exc


## Split documents into chunks
def split_doc(documents, chunk_size=1000, chunk_overlap=200):
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    docs = text_splitter.split_documents(documents)
    return docs


## Embed documents into vector store
def embed_doc(
    docs,
    embeddings,
    batch_size=700,
    delay_seconds=1.0,
    persist_path="./vector_store",
    skip_delay=False,
):
    # Create initial vector store with first batch
    print(f"Processing {len(docs)} documents in batches of {batch_size}...")
    
    first_batch = docs[:batch_size]
    print(f"Creating vector store with first batch ({len(first_batch)} documents)...")
    
    try:
        vector_store = FAISS.from_documents(first_batch, embeddings)
        print(f"✓ Initial batch completed successfully")
    except Exception as e:
        print(f"✗ Error creating initial vector store: {e}")
        if "429" in str(e) or "quota" in str(e).lower():
            print("Rate limit hit. Waiting 20 seconds before retrying...")
            time.sleep(20)
            vector_store = FAISS.from_documents(first_batch, embeddings)
            print(f"✓ Initial batch completed on retry")
        else:
            raise
    
    # Process remaining batches
    remaining_docs = docs[batch_size:]
    total_batches = (len(remaining_docs) + batch_size - 1) // batch_size + 1
    
    for i in range(0, len(remaining_docs), batch_size):
        batch = remaining_docs[i : i + batch_size]
        batch_num = (i // batch_size) + 2  # +2 because first batch already processed
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        
        try:
            vector_store.add_documents(documents=batch)
            print(f"✓ Batch {batch_num} completed successfully")
            
            # Add delay between batches to avoid rate limiting
            if i + batch_size < len(remaining_docs) and not skip_delay and delay_seconds > 0:
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
                    vector_store.add_documents(documents=batch)
                    print(f"✓ Batch {batch_num} completed on retry")
                except Exception as retry_error:
                    print(f"✗ Retry failed: {retry_error}")
                    raise
            else:
                raise
    
    # Save vector store to disk
    print(f"\nSaving vector store to {persist_path}...")
    vector_store.save_local(persist_path)
    print(f"✓ Successfully created and saved vector store with {len(docs)} documents")
    
    return vector_store


## Main function creating vector store
def create_vector_store(
    file_path,
    chunk_size=1000,
    chunk_overlap=200,
    batch_size=700,
    delay_seconds=1.0,
    persist_path="./vector_store",
    embedding_model="gemini",
    model_name=None,
):
    embeddings = get_embeddings(embedding_model=embedding_model, model_name=model_name)
    if os.path.exists(persist_path):
        print(f"Loading existing vector store from {persist_path}...")
        vector_store = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
        print("✓ Vector store loaded successfully")
        return vector_store

    documents = load_text_file(file_path)
    docs = split_doc(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    skip_delay = embedding_model == "minilm"
    vector_store = embed_doc(
        docs,
        embeddings,
        batch_size=batch_size,
        delay_seconds=delay_seconds,
        persist_path=persist_path,
        skip_delay=skip_delay,
    )
    return vector_store


## Taking a vector store path or object, create a retriever
def create_retriever(persist_path="./vector_store", vector_store=None, k=1, embedding_model="gemini", model_name=None):
    # create embedding model
    embeddings = get_embeddings(embedding_model=embedding_model, model_name=model_name)
    
    if vector_store is None:
        vector_store = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever






def generate_answer_rag_flo(persist_path, question, choices, client, k=3, embedding_model="gemini", model_name=None):
    # Create retriever from saved vector store
    retriever = create_retriever(
        persist_path=persist_path,
        k=k,
        embedding_model=embedding_model,
        model_name=model_name,
    )
    
    # Format choices clearly for the model
    formatted_choices = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])

    # Get context and format for model
    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n{d.page_content}"
        for d in docs
    )

    prompt = (
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{formatted_choices}\n\n"
        "Provide your final answer as one of the letters A, B, C, or D."
    )

    # Use JSON mode to guarantee a parseable structure
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "answer": {"type": "string", "enum": ["A", "B", "C", "D"]}
                },
                "required": ["reasoning", "answer"]
            },
            "max_output_tokens": 5096,
            "temperature": 0.1, # Low temperature for factual MMLU tasks
        }
    )
    data = json.loads(response.text)
    return data["answer"], data["reasoning"]
    
    

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "input_data", "concept_net.txt")
    persist_path = "./vector_store_faiss"
    
    # Create vector store
    create_vector_store(
        file_path=file_path,
        chunk_size=2000,
        chunk_overlap=200,
        batch_size=1500,
        delay_seconds=1.0,
        persist_path=persist_path,
        embedding_model="gemini",
    )