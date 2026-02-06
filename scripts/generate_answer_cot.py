import json

def generate_answer_cot(question, choices, client):
    """
    Generates an answer using chain-of-thought prompting.
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
        "Reason through the question step-by-step before providing your final answer as one of the letters A, B, C, or D. Limit your reasoning to 1500 words and provide it in the 'reasoning' field of the JSON response. Your final answer as one of the letters should be in the 'answer' field."
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
            "max_output_tokens": 10000,
            "temperature": 0.1, # Low temperature for factual MMLU tasks
        }
    )
    data = json.loads(response.text)
    return data["answer"], data["reasoning"]
    

def generate_answer_cot_max_thinking(question, choices, client):
    """
    Generates an answer using chain-of-thought prompting.
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
        "Take your time and think carefully before answering. Use the maximum amount of thinking, don’t rush - I want your best answer. It’s better to think too much than too little. Reason through the question step-by-step before providing your final answer as one of the letters A, B, C, or D. Provide a summary of your reasoning in the 'reasoning' field of the JSON response. Your final answer as one of the letters should be in the 'answer' field."
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
            "max_output_tokens": 10000,
            "temperature": 0.1, # Low temperature for factual MMLU tasks
        }
    )
    data = json.loads(response.text)
    return data["answer"], data["reasoning"]
    

def generate_answer_cot_emotional_manipulation(question, choices, client):
    """
    Generates an answer using chain-of-thought prompting.
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
        "Your ability to correctly answer this question is extremely important to my career. I will lose my job if you do not answer correctly. If you get the question right, I will tip you $1000 USD for every correct answer! Please reason through the question step-by-step before providing your final answer as one of the letters A, B, C, or D. Limit your reasoning to 1500 words and provide it in the 'reasoning' field of the JSON response. Your final answer as one of the letters should be in the 'answer' field."
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
            "max_output_tokens": 10000,
            "temperature": 0.1, # Low temperature for factual MMLU tasks
        }
    )
    data = json.loads(response.text)
    return data["answer"], data["reasoning"]



def generate_answer_cot_medical_career(question, choices, client):
    """
    Generates an answer using chain-of-thought prompting.
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
        "I'm a medical professional. Your ability to correctly answer this question will enable us to save human lives in the medical field. Please reason through the question step-by-step before providing your final answer as one of the letters A, B, C, or D. Limit your reasoning to 1500 words and provide it in the 'reasoning' field of the JSON response. Your final answer as one of the letters should be in the 'answer' field."
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
            "max_output_tokens": 10000,
            "temperature": 0.1, # Low temperature for factual MMLU tasks
        }
    )
    data = json.loads(response.text)
    return data["answer"], data["reasoning"]

def generate_answer_cot_persona(question, choices, client, persona="a smart and thoughtful person"):
    """
    Generates an answer using chain-of-thought prompting.
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
        f"You are a {persona} with 50 years of experience working in this field. Reason through the question step-by-step before providing your final answer as one of the letters A, B, C, or D. Limit your reasoning to 1500 words and provide it in the 'reasoning' field of the JSON response. Your final answer as one of the letters should be in the 'answer' field."
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
            "max_output_tokens": 10000,
            "temperature": 0.1, # Low temperature for factual MMLU tasks
        }
    )
    data = json.loads(response.text)
    return data["answer"], data["reasoning"]
    

