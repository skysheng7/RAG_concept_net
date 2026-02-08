import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import import_datasets
import general_utils
from generate_answer_std import generate_answer_std
from generate_answer_cot import generate_answer_cot
from generate_answer_rag import generate_answer_rag


def process_row(
    row, answer_generation_function, client, vector_store, k, prompt, prompt_type, shot_n=None
):
    """Helper to process a single row for parallel execution."""
    question = row["question"]
    choices = row["choices"][:4]  # Simplified slicing
    correct_choice = {0: "A", 1: "B", 2: "C", 3: "D"}[row["answer"]]
    subject = row.get("subject", "N/A")
    k_to_store = None

    kwargs = {"client": client}
    if "rag" in answer_generation_function.__name__.lower():
        kwargs.update({"vector_store": vector_store, "k": k})
        k_to_store = k

    # Retry logic
    for i in range(10):
        try:
            model_answer = answer_generation_function(question, choices, **kwargs)
            break  # Success! Exit the loop
        except Exception as e:
            # Captures the specific error message as the reasoning
            error_msg = f"Attempt {i+1} failed: {str(e)}"
            model_answer = (f"Error", error_msg)

    # if this is standard prompt, there is no reasoning
    if isinstance(model_answer, (list, tuple)):
        return_model_answer = model_answer[0]

        if len(model_answer) > 1:
            return_model_resoning = model_answer[1]
    else:
        return_model_answer = model_answer
        return_model_resoning = None

    return {
        "question": question,
        "model_answer": return_model_answer,
        "model_reasoning": return_model_resoning,
        "correct_answer": correct_choice,
        "correct": return_model_answer.upper() == correct_choice.upper(),
        "subject": subject,
        "k": k_to_store,
        "prompt_type": prompt_type,
        "prompt": prompt,
        "shot_n": shot_n
    }


def evaluate_model_fast(
    df,
    answer_generation_function,
    client,
    debugging=False,
    vector_store=None,
    k=1,
    max_workers=10,
    prompt=None,
    prompt_type=None,
    shot_n=None,
):
    # Convert DF to list of dicts for faster iteration
    rows = df.to_dict("records")

    # Use ThreadPoolExecutor to call the API in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            executor.map(
                lambda r: process_row(
                    r,
                    answer_generation_function,
                    client,
                    vector_store,
                    k,
                    prompt,
                    prompt_type,
                    shot_n,
                ),
                rows,
            )
        )

    results_df = pd.DataFrame(results)

    # --- New Filtering Logic ---
    # Filter for rows where the model actually gave a choice (A, B, C, or D)
    # and didn't return "Error"
    valid_results = results_df[results_df["model_answer"] != "Error"]

    if not valid_results.empty:
        accuracy = valid_results["correct"].mean()
    else:
        accuracy = 0.0  # Handle case where everything failed
    # ---------------------------

    if debugging:
        error_count = len(results_df) - len(valid_results)
        print(f"Evaluation complete.")
        print(f"Total Rows: {len(results_df)} | Errors Excluded: {error_count}")
        print(f"Adjusted Accuracy: {accuracy:.2%}")

    return accuracy, results_df


def batch_eval(
    dataset_names,
    prompt_db,
    client,
    k_range,
    debugging=False,
    vector_store=None,
    max_workers=10,
    folder_path="../results",
    few_shot_range=range(1, 6, 1)
):
    # load datasets (100 samples from each dataset)
    # Generate 100 samples from the mmlu datasets
    datasets = import_datasets.import_datasets(dataset_names)

    # Calculate total number of evaluations for progress bar
    total_tasks = 0
    for prompt_type in prompt_db.keys():
        if "rag" in prompt_type:
            total_tasks += len(datasets) * len(k_range)
        else:
            total_tasks += len(datasets)
    
    # create a master dataframe to store all the results
    master_results_df = pd.DataFrame()
    pbar = tqdm(total=total_tasks, desc="Batch Evaluation Progress")
    
    for subject, df in datasets.items():
        for prompt_type, prompt in prompt_db.items():
            # standard prompt
            if prompt_type == "standard":
                accuracy, results_df = evaluate_model_fast(
                    df,
                    generate_answer_std,
                    client,
                    debugging=debugging,
                    max_workers=max_workers,
                    prompt=prompt,
                    prompt_type=prompt_type,
                )
                master_results_df = pd.concat(
                    [master_results_df, results_df], ignore_index=True
                )
                general_utils.save_df_to_csv(
                    results_df,
                    f"results_df_{subject}_{prompt_type}.csv",
                    folder_path=folder_path,
                )
                pbar.update(1)
            elif "few_shot" in prompt_type: # few shot prompting
                # iterate through if we want to do 1-shot, 2-shot, 3-shot, 4-shot, 5-shot learning with different few shot examples
                for shot_n in few_shot_range: 
                    example_string = import_datasets.create_few_shot_examples(subject, shot_n)
                    prompt = prompt + example_string
                    accuracy, results_df = evaluate_model_fast(
                        df,
                        generate_answer_cot,
                        client,
                        debugging=debugging,
                        max_workers=max_workers,
                        prompt=prompt,
                        prompt_type=prompt_type,
                        shot_n=shot_n
                    )
                    master_results_df = pd.concat(
                        [master_results_df, results_df], ignore_index=True
                    )
                    general_utils.save_df_to_csv(
                        results_df,
                        f"results_df_{subject}_{prompt_type}_{shot_n}shot_learning.csv",
                        folder_path=folder_path,
                    )
                    pbar.update(1)
            elif "cot" in prompt_type:  # chain-of-thought prompting with variations
                if "persona" in prompt_type:
                    subject_name = subject.replace("_", " ")
                    prompt = prompt.replace("\{domain\}", subject_name)
                    
                accuracy, results_df = evaluate_model_fast(
                    df,
                    generate_answer_cot,
                    client,
                    debugging=debugging,
                    max_workers=max_workers,
                    prompt=prompt,
                    prompt_type=prompt_type,
                )
                master_results_df = pd.concat(
                    [master_results_df, results_df], ignore_index=True
                )
                general_utils.save_df_to_csv(
                    results_df,
                    f"results_df_{subject}_{prompt_type}.csv",
                    folder_path=folder_path,
                )
                pbar.update(1)
            elif "rag" in prompt_type:  # RAG prompting
                for k in k_range:
                    accuracy, results_df = evaluate_model_fast(
                        df,
                        generate_answer_rag,
                        client,
                        debugging=debugging,
                        vector_store=vector_store,
                        k=k,
                        max_workers=max_workers,
                        prompt=prompt,
                        prompt_type=prompt_type,
                    )
                    master_results_df = pd.concat(
                        [master_results_df, results_df], ignore_index=True
                    )
                    general_utils.save_df_to_csv(
                        results_df,
                        f"results_df_{subject}_{prompt_type}_k{k}.csv",
                        folder_path=folder_path,
                    )
                    pbar.update(1)

    pbar.close()
    
    # save the master results dataframe
    general_utils.save_df_to_csv(
        master_results_df, f"master_results_df.csv", folder_path=folder_path
    )

    return master_results_df
