import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def process_row(row, answer_generation_function, client, vector_store, k):
    """Helper to process a single row for parallel execution."""
    question = row['question']
    choices = row['choices'][:4]  # Simplified slicing
    correct_choice = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}[row['answer']]
    subject = row.get('subject', 'N/A')
    
    kwargs = {"client": client}
    if "rag" in answer_generation_function.__name__.lower():
        kwargs.update({"vector_store": vector_store, "k": k})

    # Retry logic
    for i in range(2):
        try:
            model_answer = answer_generation_function(question, choices, **kwargs)
            break # Success! Exit the loop
        except Exception as e:
            # Captures the specific error message as the reasoning
            error_msg = f"Attempt {i+1} failed: {str(e)}"
            model_answer = (f"Error", error_msg)
            
    return {
        'question': question,
        'model_answer': model_answer[0],
        'model_reasoning': model_answer[1],
        'correct_answer': correct_choice,
        'correct': model_answer[0] == correct_choice,
        'subject': subject
    }

def evaluate_model_fast(df, answer_generation_function, client, debugging=False, vector_store=None, k=1, max_workers=10):
    # Convert DF to list of dicts for faster iteration
    rows = df.to_dict('records')
    
    # Use ThreadPoolExecutor to call the API in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda r: process_row(r, answer_generation_function, client, vector_store, k), rows))

    results_df = pd.DataFrame(results)

    # --- New Filtering Logic ---
    # Filter for rows where the model actually gave a choice (A, B, C, or D)
    # and didn't return "Error"
    valid_results = results_df[results_df['model_answer'] != "Error"]
    
    if not valid_results.empty:
        accuracy = valid_results['correct'].mean()
    else:
        accuracy = 0.0  # Handle case where everything failed
    # ---------------------------
    
    if debugging:
        error_count = len(results_df) - len(valid_results)
        print(f"Evaluation complete.")
        print(f"Total Rows: {len(results_df)} | Errors Excluded: {error_count}")
        print(f"Adjusted Accuracy: {accuracy:.2%}")
        
    return accuracy, results_df
        

    return accuracy, results_df
