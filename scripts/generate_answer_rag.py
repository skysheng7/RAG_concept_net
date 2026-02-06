import json
from rag import build_retriver


def generate_answer_rag(question, choices, client, vector_store, k=1, prompt=None):
    """
    Generates an answer using RAG.
    Args:
        question (str): The question to answer.
        choices (list): A list of answer choices.
        client (Client): The Gemini client.
        vector_store (InMemoryVectorStore): The vector store.
        k (int): The number of most similar documents to return.
    Returns:
        str: The selected answer choice letter.
        str: The full response from the model for debugging.
    """
    # build the retriever 
    retrived_docs = build_retriver(vector_store, [question], k = k)
    # batch() returns a list of lists, so we need to get the first element
    retrived_docs = retrived_docs[0] if retrived_docs else []
    content_text = [doc.page_content for doc in retrived_docs]
    content = "\n\n".join(content_text)
    
    # Format choices clearly for the model
    formatted_choices = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
    
    prompt = (
        f"Context: {content}\n\n"
        f"Question: {question}\n\n"
        f"Choices:\n{formatted_choices}\n\n"
        f"{prompt}"
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
    