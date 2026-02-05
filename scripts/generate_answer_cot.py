import pandas as pd
import os
from dotenv import load_dotenv
import json
from google import genai
import pandas as pd

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


def generate_answer_cot(question, choices, client=client):
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
        "Reason through the question step-by-step limiting yourself to 2000 words, then provide your final answer as one of the letters A, B, C, or D."
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
    

#Evaluate the model on one of the datasets. This code can be modified to evaluate other models as well.
def evaluate_model(df, debugging=False):
    """
    Evaluates the model on a given DataFrame of questions.
    Args:
        df (pd.DataFrame): DataFrame containing 'question', 'choices', and 'answer' columns.
        debugging (bool): If True, prints detailed information for each question.
    Returns:
        float: Accuracy of the model.
        pd.DataFrame: DataFrame containing results for each question.
    """
    correct_answers = 0
    total_questions = len(df)
    results_df = pd.DataFrame(columns=['question', 'model_answer', 'correct_answer'])
    for index, row in df.iterrows():
        question = row['question']
        choices = [row['choices'][0], row['choices'][1], row['choices'][2], row['choices'][3]]
        correct_choice = row['answer']
        converted_choices = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        correct_choice = converted_choices[correct_choice]
        try:
            model_answer = generate_answer_cot(question, choices)

        except:
            #try again in case of an error
            try:
                model_answer = generate_answer_cot(question, choices)
            except:
                model_answer = ("Error", "Error")

        if model_answer[0] == correct_choice:
            correct_answers += 1

        results_df = pd.concat([results_df, pd.DataFrame({
            'question': [question],
            'model_answer': [model_answer[0]],
            'correct_answer': [correct_choice]
        })], ignore_index=True)
        if debugging:
            print(f"Q{index}: {question}")
            print(f"Model Answer: {model_answer[0]}, Correct Answer: {correct_choice}\n")

    accuracy = correct_answers / total_questions
    return accuracy, results_df