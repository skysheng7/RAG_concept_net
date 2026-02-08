from datasets import load_dataset, Dataset
import pandas as pd


def import_datasets(names):
    datasets = {}
    for name in names:
        datasets[name] = load_dataset("cais/mmlu", name, split="test[:100]").to_pandas()
    return datasets


def import_dev_set(subject):
    dev_dataset = load_dataset("cais/mmlu", subject, split="dev[:]").to_pandas()
    return dev_dataset


def create_few_shot_examples(subject, shot_n=1):
    # get the dev dataset as few shot examples.
    dev_dataset = import_dev_set(subject)
    if shot_n <= len(dev_dataset):
        cur_dataset = dev_dataset[:shot_n]
    else:
        cur_dataset = dev_dataset.copy()

    # create an empty string to return
    example_string = ""
    for index, row in cur_dataset.iterrows():
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]
        answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
        example_string += f"Example {index+1}: Question is: {question}. \n Choices are: {choices}. \n Correct answer is: {answer_map[answer]}. \n The selected answer is: {list(choices)[answer]}. \n\n"

    return example_string


if __name__ == "__main__":
    dataset_names = ["anatomy", "formal_logic", "moral_scenarios"]
    example_string = create_few_shot_examples("moral_scenarios", shot_n=5)
    print(example_string)
