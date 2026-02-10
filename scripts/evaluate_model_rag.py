import pandas as pd
import json
import time
from rag_florent import create_retriever

def generate_answer(context, question, choices, client):
    # format choices
    formatted_choices = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
    
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


def evaluate_model(df, vector_store, answer_generation_function, embedding_model, client, k=1, debugging=False):
    retriever = create_retriever(vector_store, k=k, embedding_model=embedding_model)

    # get context for each question
    contexts = retriever.batch(df['question'].tolist())

    # create results dataframe
    results_df = pd.DataFrame(columns=['question', 'model_answer', 'correct_answer'])

    for index, row in df.iterrows():
        # get answer
        question = row['question']
        choices = row['choices']
        context = "\n\n".join({doc.page_content for doc in contexts[index]})

        try:
            answer, reasoning = answer_generation_function(context, question, choices, client)
        except Exception as e: 
            print(f"Error: {e}, trying again...")
            
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(20)
            #try again in case of an error
            try:
                model_answer = answer_generation_function(question, choices, **kwargs)
            except Exception as e:
                print(f"Error: {e}")
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(20)
                error_answers += 1
                model_answer = ("Error", "Error")

        # store answer in results_df
        results_df.iloc[index] = {
            'question': question,
            'model_answer':model_answer,
            'correct_answer': row['answer']
            }
        
    # add subject column
    results_df['subject'] = df['subject']
        
    # compare answers
    results_df['correct'] = results_df['model_answer'][0] == results_df['correct_answer']

    # compute accuracy
    accuracy = results_df['correct'].astype(int).mean()

    return accuracy, results_df