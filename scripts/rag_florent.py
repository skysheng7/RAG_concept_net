import time
import json
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def setup_huggingface_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    print(f"Loading HuggingFace embeddings model: {model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print("✓ Embeddings model loaded successfully")
    return embeddings

def load_text_file(file_path: str):
    # Load text file into documents
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

def split_doc(documents, chunk_size=1000, chunk_overlap=200):
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )
    docs = text_splitter.split_documents(documents)
    return docs


def embed_doc(docs, embeddings, batch_size=300, delay_seconds=2, persist_path="./vector_store"):
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
            if i + batch_size < len(remaining_docs):
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

def create_vector_store(file_path, embeddings, chunk_size=1000, chunk_overlap=200, batch_size=300, delay_seconds=2, persist_path="./vector_store"):
    documents = load_text_file(file_path)
    docs = split_doc(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vector_store = embed_doc(docs, embeddings, batch_size=batch_size, delay_seconds=delay_seconds, persist_path=persist_path)
    return vector_store

def create_retriever(embeddings=None, persist_path="./vector_store", vector_store=None, stype="similarity", k=1):
    # If vector_store not provided, try to load from disk
    if vector_store is None:
        if os.path.exists(persist_path):
            print(f"Loading existing vector store from {persist_path}...")
            vector_store = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
            print("✓ Vector store loaded successfully")
        else:
            raise ValueError(f"No vector store found at {persist_path}. Please create one first using embed_doc().")
    
    retriever = vector_store.as_retriever(
        search_type=stype,
        search_kwargs={"k": k},
    )
    return retriever

def generate_answer_rag(embeddings, persist_path, question, choices, client, k=3, stype="similarity"):
    """
    Generate answer using RAG by loading vector store and retrieving relevant context.
    
    Parameters
    ----------
    embeddings : Embeddings
        Embedding model to use for loading the vector store
    persist_path : str
        Path to the saved vector store
    question : str
        Question to answer
    choices : list
        List of answer choices
    client : genai.Client
        Gemini client for generation
    k : int, optional
        Number of documents to retrieve (default: 3)
    stype : str, optional
        Search type for retriever (default: "similarity")
    
    Returns
    -------
    tuple
        (answer, reasoning)
    """
    # Create retriever from saved vector store
    retriever = create_retriever(embeddings=embeddings, persist_path=persist_path, stype=stype, k=k)
    
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
    
    