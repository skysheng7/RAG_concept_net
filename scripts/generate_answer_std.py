import json

def generate_answer_std(question, choices, client):
    """
    Generates an answer using standard prompting.
    Args:
        question (str): The question to answer.
        choices (list): A list of answer choices.
    Returns:
        str: The selected answer choice letter.
        str: The full response from the model for debugging.
    """
    # Format choices clearly for the model
    formatted_choices = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
    
    prompt = (
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
    

