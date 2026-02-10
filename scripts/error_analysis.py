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
        7: "Reasoning doesn't find an answer",
        8: "Chose answer not consistent with reasoning"
    }

    error_analysis = pd.read_excel(input_path, sheet_name="selected_errors")
    error_analysis = error_analysis.fillna(0)


    error_counts = (
        error_analysis
        .groupby(["prompt_type", "shot_n", "k"])["Error Category"]
        .value_counts()
    ).to_frame(name='count')

    error_counts["percentage"] = (
        error_counts
        .groupby(level=[0, 1, 2])["count"]
        .transform(lambda x: round(100 * x / x.sum(), 2))
    )

    error_pivot = error_counts.pivot_table(
        index=["prompt_type", "shot_n", "k"],
        columns="Error Category",
        values="percentage",
        fill_value=0
    )

    prompt_type_order = ["standard", "cot", "cot_few_shot", "rag", "rag_optional"]
    error_pivot = error_pivot.reindex(prompt_type_order, level=0)

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
    input_path = "../results/round1/model_error_analysis.xlsx"
    output_path = "../results/round1/error_percentages.xlsx"
    e = analyse_errors(input_path, output_path)