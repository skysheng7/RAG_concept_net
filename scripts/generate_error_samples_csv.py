import pandas as pd

def generate_error_samples_csv(df, outputname='../results/round1/model_error_samples.csv'):
    """
    Generates a dataframe containing up to 20 random samples of errors for each prompt type and subject. Exports this to a csv. 
    Args:
        (df): dataframe containing the 
    Returns:
        dataframe containing up to 20 random samples of incorrect answers from the dataframe
    
    """
    
    # Filter for all errors first
    all_errors = df[df['correct'] == False]

    # Group by prompt and sample 20 from each group
    # Add group_keys=False to the groupby call
    error_samples_df = all_errors.groupby(
        ['prompt_type', 'subject', 'shot_n', 'k'], 
        group_keys=False,
        dropna=False
    ).apply(
        lambda x: x.sample(n=min(20, len(x)), random_state=42)
    ).reset_index(drop=True) # drop=True if you don't need the original row numbers

    error_samples_df = error_samples_df.sort_values(by=['prompt_type', 'subject', 'shot_n', 'k'])

    error_samples_df.to_csv(outputname, index=False)

    return error_samples_df