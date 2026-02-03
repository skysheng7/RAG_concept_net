# RAG Concept Net

Authors (listed in alphabetical order): Sky (Kehan) Sheng, Bradley Beecher, Florent Samson, Jeenat Mehareen

## About

Compare the performance of the model in completing MMLU benchmark tasks when we use [1] standard prompting, [2] Chain-of-Thought prompting, [3] RAG prompting. The MMLU benchmark tasks we use are: moral scenario, formal logic, anatomy

## Repository Structure

<details>
<summary>Click to expand</summary>

```
RAG_concept_net/
├── scripts/
│   ├── gemini_utils.py      # Google Gemini API utilities
│   └── general_utils.py     # General utilities and provider abstractions
├── environment.yml           # Conda environment specification
├── .env                      # API keys (not tracked in git)
├── .gitignore               # Git ignore rules
├── LICENSE                  # Project license
└── README.md                # This file
```

</details>

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
```

1. Obtain API keys:
   - **Gemini**: Visit [Google AI Studio](https://aistudio.google.com/), sign in with your Google account, and generate an API key
