from datasets import load_dataset, Dataset
import pandas as pd

def import_datasets(names):
    datasets = {}
    for name in names:                                                                                             
        datasets[name] = load_dataset('cais/mmlu', name, split='test[:100]').to_pandas()
    return datasets

if __name__ == "__main__":
    dataset_names = ['anatomy', 'formal_logic', 'moral_scenarios']
    datasets = import_datasets(dataset_names)   