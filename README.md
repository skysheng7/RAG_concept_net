# RAG Concept Net

Authors: Sky (Kehan) Sheng, Bradley Beecher, Florent Samson, Jeenat Mehareen

## About

Compare the performance of the model in completing MMLU benchmark tasks when we use [1] standard prompting, [2] Chain-of-Thought prompting, [3] RAG prompting. The MMLU benchmark tasks we use are: moral scenario, formal logic, anatomy.

## Repository Structure

```
RAG_concept_net/
├── scripts/
│   ├── implementation.ipynb              # Main analysis notebook
│   ├── error_analysis_2.ipynb            # Error analysis notebook
│   ├── rag.py                            # Core RAG implementation
│   ├── generate_answer_std.py            # Standard prompting
│   ├── generate_answer_cot.py            # Chain-of-Thought prompting
│   ├── generate_answer_rag.py            # RAG prompting
│   ├── evaluate_model_fast.py            # Fast evaluation script
│   ├── evaluate_model_rag.py             # RAG evaluation script
│   ├── error_analysis.py                 # Error analysis utilities
│   ├── visualize_results.py              # Visualization utilities
│   ├── summarize_accuracy.py             # Accuracy metrics
│   ├── import_datasets.py                # Dataset loading
│   ├── generate_error_samples_csv.py     # Error export to CSV
│   ├── gemini_utils.py                   # Google Gemini API utilities
│   ├── general_utils.py                  # General utilities
│   ├── implementation_flo.ipynb          # Experimental (MiniLM, not used)
│   └── rag_florent.py                    # Experimental (MiniLM, not used)
├── input_data/                           # Concept net text file (not tracked in git)
├── plots/                                # Generated visualization plots
│   ├── accuracy_comparison_all_prompt_type.png
│   ├── accuracy_comparison_few_shot.png
│   ├── accuracy_comparison_rag_k.png
│   └── accuracy_comparison_rag_optional_k.png
├── results/                              # Experimental results (CSV and XLSX files)
├── environment.yml                       # Conda environment specification
├── conda-lock.yml                        # Conda-lock file for reproducible environments
├── .env                                  # API keys (not tracked in git)
├── .gitignore                            # Git ignore rules
├── LICENSE                               # Project license
└── README.md                             # This file
```

### Scripts and Notebooks

Note that all the scripts and notebooks are in the `scripts` folder.

#### Main Analysis Files

##### Jupyter Notebooks

- **`implementation.ipynb`** - **Main analysis notebook**. Contains the complete workflow for evaluating LLM performance on MMLU benchmarks across various prompting strategies (standard, Chain-of-Thought, and RAG). This notebook prompts LLMs to generate answers, process the data, and produces the final results.

- **`error_analysis_2.ipynb`** - Error analysis notebook for examining model failures and patterns in incorrect predictions across different prompting methods.

##### Python Scripts

- **`rag.py`** - RAG (Retrieval-Augmented Generation) implementation. Handles document loading, text chunking, embeddings generation using Google's Gemini model, and vector store creation with Chroma for retrieval.

- **`generate_answer_std.py`** - Generates answers using standard prompting (no additional context or reasoning steps).

- **`generate_answer_cot.py`** - Generates answers using Chain-of-Thought prompting, encouraging the model to explain its reasoning.

- **`generate_answer_rag.py`** - Generates answers using RAG prompting, incorporating relevant context retrieved from the vector store.

- **`evaluate_model_fast.py`** - Fast evaluation script for quickly testing model performance on MMLU tasks.

- **`evaluate_model_rag.py`** - Evaluation script specifically for RAG-based answer generation, integrating the retriever with the answer generation pipeline.

- **`error_analysis.py`** - Python script for programmatic error analysis and statistics.

- **`visualize_results.py`** - Visualization utilities for creating accuracy plots, boxplots, and other figures from experimental results.

- **`summarize_accuracy.py`** - Computes and summarizes accuracy metrics across different prompting methods and datasets.

- **`import_datasets.py`** - Utilities for loading and preprocessing MMLU datasets.

- **`generate_error_samples_csv.py`** - Exports error samples to CSV format for detailed inspection.

- **`gemini_utils.py`** - Google Gemini API utilities and helper functions.

- **`general_utils.py`** - General utilities and provider abstractions used across multiple scripts.

#### Experimental Files (Not Used in Final Analysis)

The following files represent Florent's experimental attempts using MiniLM for RAG and embeddings. These were exploratory but not included in the final analysis:

- **`implementation_flo.ipynb`** - experimental notebook testing MiniLM-based embeddings.
- **`rag_florent.py`** - RAG implementation using MiniLM embeddings instead of Google's Generative AI embeddings.


## Dependencies

<details>
<summary>Click to expand</summary>

### Environment Management

- Conda (recommended)
- conda-lock (for reproducible environments)

</details>

## Installation

1. Clone this repository:

```bash
git clone https://github.com/skysheng7/RAG_concept_net.git
cd RAG_concept_net
```

### Option 1: Using `environment.yml` (Quick Setup)

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate rag
```

**Note**: The environment includes:
- `nltk` for natural language processing and sentence tokenization
- `langchain_chroma` (via pip) for vector storage in the RAG pipeline

### Option 2: Using Conda-Lock (Reproducible Setup)

For exact reproducibility across different operating systems:

```bash
conda-lock install --name rag conda-lock.yml
conda activate rag
```

## Setup

### API Key Configuration

1. Create a `.env` file in the project root directory using command line:

```bash
touch .env
```

1. Add your API keys to the `.env` file:

```
GEMINI_API_KEY=<your_gemini_api_key_here>
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY=<your_langsmith_api_key_here>
```

1. Obtain API keys:
   - **Gemini**: Visit [Google AI Studio](https://aistudio.google.com/), sign in with your Google account, and generate an API key
