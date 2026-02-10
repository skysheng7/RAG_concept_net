import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyse_errors(input_path, output_path):
    error_codes = {
        0: "Wrong answer in database",
        1: "Wrong fact",
        2: "Half correct half wrong",
        3: "No information in context",
        4: "Misunderstanding of the question",
        5: "Logic error",
        6: "LLM error",
    }

    error_analysis = pd.read_excel(input_path, sheet_name="selected_errors")

    error_counts = (
        error_analysis
        .groupby(["subject", "prompt_type"])["Error Category"]
        .value_counts()
    )

    subjects = error_analysis["subject"].unique()
    prompt_type_order = ["standard", "cot", "rag"]
    prompt_types = [pt for pt in prompt_type_order if pt in error_analysis["prompt_type"].unique()]
    categories = sorted(set(error_analysis["Error Category"].unique()).union({5}))

    full_index = pd.MultiIndex.from_product(
        [subjects, prompt_types, categories],
        names=["subject", "prompt_type", "Error Category"]
    )

    error_grouped = (
        error_counts
        .rename("count")
        .reindex(full_index, fill_value=0)
        .to_frame()
    )

    error_grouped["percentage"] = (
        error_grouped
        .groupby(level=[0, 1])["count"]
        .transform(lambda x: round(100 * x / x.sum(), 2))
    )

    error_pivot = error_grouped.pivot_table(
        index=["subject", "prompt_type"],
        columns="Error Category",
        values="percentage",
        fill_value=0
    )

    prompt_type_order = ["standard", "cot", "rag"]
    subjects = error_pivot.index.get_level_values("subject").unique()
    prompt_types = [
        pt for pt in prompt_type_order
        if pt in error_pivot.index.get_level_values("prompt_type").unique()
    ]

    error_pivot = error_pivot.reindex(
        pd.MultiIndex.from_product(
            [subjects, prompt_types],
            names=["subject", "prompt_type"]
        )
    )

    error_pivot.to_excel(output_path)
    return error_pivot



# # Ensure column order is sorted numerically (error categories)
# error_pivot = error_pivot.reindex(sorted(error_pivot.columns), axis=1)

# # Build row labels like "subject | prompt_type"
# row_labels = [
#     f"{subj} | {pt}"
#     for subj, pt in error_pivot.index.to_list()
# ]

# plt.figure(figsize=(1.2 * len(error_pivot.columns), 0.6 * len(error_pivot.index)))
# ax = sns.heatmap(
#     error_pivot.values,
#     annot=True,
#     fmt=".2f",
#     cmap="Reds",
#     cbar=False,
#     linewidths=0.5,
#     linecolor="black"
# )

# ax.set_xticklabels(error_pivot.columns, rotation=0)
# ax.set_yticklabels(row_labels, rotation=0)
# ax.set_xlabel("Error Category")
# ax.set_ylabel("Subject | Prompt Type")
# ax.set_title("Error Percentage by Category")

# plt.tight_layout()
# plt.show()

if __name__ == "__main__":
    input_filepath = "../results/round1/model_error_analysis.xlsx"
    output_filepath = "../results/round1/error_percentages.xlsx"
    analyse_errors(input_filepath, output_filepath)