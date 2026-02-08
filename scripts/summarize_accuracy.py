import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def summarize_accuracy(input_file='../results/master_results_df.csv', output_file='../results/accuracy_by_prompt_subject.csv', group_by=['prompt_type', 'subject']):
    """
    This function reads a CSV file containing model results, filters out rows where the model output was an "error", 
    calculates the accuracy of the model's answers for each combination of prompt type and subject, and then reshapes 
    the results into a table format. Finally, it prints the accuracy table and saves it to a new CSV file.
        Parameters: csv_file (str): The path to the CSV file containing the model results. Default is 'master_results_df.csv'.  
        Returns:    A DataFrame containing the accuracy of the model's answers for each combination of prompt type and subject.
    """
    df = pd.read_csv(input_file)
    # 1. Filter out rows where the model output was an "error"
    filtered_df = df[df['model_answer'] != 'Error']
    print(f"Filtered out {len(df) - len(filtered_df)} rows where the model output was an 'Error'")

    # 2. Calculate accuracy on the filtered data
    accuracy_by_prompt_subject = filtered_df.groupby(group_by)['correct'].mean()

    # 3. Reshape for your table
    accuracy_by_prompt_subject = accuracy_by_prompt_subject.unstack(level='subject')

    print(accuracy_by_prompt_subject)
    accuracy_by_prompt_subject.to_csv(output_file, index=False)
    return accuracy_by_prompt_subject

