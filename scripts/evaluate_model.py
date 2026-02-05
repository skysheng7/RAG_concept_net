import pandas as pd

def evaluate_model(df, answer_generation_function, client, debugging=False):
    """
    Evaluates the model on a given DataFrame of questions.
    Args:
        df (pd.DataFrame): DataFrame containing 'question', 'choices', and 'answer' columns.
        answer_generation_function (function): Function to generate answers given a question and choices.
                                        The function should take in (question: str, choices: List[str], client) 
                                        and return a tuple (answer_choice: str, explanation: str).
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
        correct = False
        subject = row['subject']
        try:
            model_answer = answer_generation_function(question, choices, client=client)

        except:
            #try again in case of an error
            try:
                model_answer = answer_generation_function(question, choices, client=client)
            except:
                model_answer = ("Error", "Error")

        if model_answer[0] == correct_choice:
            correct_answers += 1
            correct = True

        results_df = pd.concat([results_df, pd.DataFrame({
            'question': [question],
            'model_answer': [model_answer[0]],
            'correct_answer': [correct_choice],
            'correct': [correct],
            'subject': [subject]
        })], ignore_index=True)
        if debugging:
            print(f"Q{index}: {question}")
            print(f"Model Answer: {model_answer[0]}, Correct Answer: {correct_choice}\n")

    accuracy = correct_answers / total_questions
    return accuracy, results_df