import pandas as pd
from rag_florent import create_retriever


def _normalize_answer(answer):
    if isinstance(answer, (int, float)):
        answer_int = int(answer)
        if answer_int in (0, 1, 2, 3):
            return "ABCD"[answer_int]
        if answer_int in (1, 2, 3, 4):
            return "ABCD"[answer_int - 1]
    if isinstance(answer, str):
        cleaned = answer.strip().upper()
        if cleaned in ("A", "B", "C", "D"):
            return cleaned
        if cleaned.isdigit():
            answer_int = int(cleaned)
            if answer_int in (0, 1, 2, 3):
                return "ABCD"[answer_int]
            if answer_int in (1, 2, 3, 4):
                return "ABCD"[answer_int - 1]
    return str(answer)


def _docs_for_index(retrieved_docs, index, total_questions):
    if total_questions == 1 and retrieved_docs:
        if isinstance(retrieved_docs[0], list):
            return retrieved_docs[0]
        return retrieved_docs
    if index < len(retrieved_docs):
        docs = retrieved_docs[index]
        if isinstance(docs, list):
            return docs
        return [docs]
    return []


def _build_context(docs_list):
    context_parts = []
    for doc in docs_list:
        source = getattr(doc, "metadata", {}).get("source", "unknown")
        content = getattr(doc, "page_content", str(doc))
        context_parts.append(f"Source: {source}\n{content}")
    return "\n\n".join(context_parts)


def _call_answer_function(answer_generation_function, question, choices, client, context, docs_list):
    try_kwargs = [
        {"client": client, "context": context},
        {"client": client, "retrieved_docs": docs_list},
        {"client": client},
    ]
    for kwargs in try_kwargs:
        try:
            return answer_generation_function(question, choices, **kwargs)
        except TypeError:
            continue
    return ("Error", "Error")


def evaluate_model_rag(df, answer_generation_function, client, vector_store, k=1, debugging=False):
    """
    Evaluates the model on a given DataFrame of questions using batched retrieval.
    Args:
        df (pd.DataFrame): DataFrame containing 'question', 'choices', and 'answer' columns.
        answer_generation_function (function): Function to generate answers given a question and choices.
            Expected signature: (question: str, choices: List[str], client, context: str)
            Return: (answer_choice: str, explanation: str).
        client (Client): The Gemini client.
        vector_store (FAISS): The vector store to retrieve from.
        k (int): The number of most similar documents to return per question.
        debugging (bool): If True, prints detailed information for each question.
    Returns:
        float: Accuracy of the model.
        pd.DataFrame: DataFrame containing results for each question.
    """
    questions = df["question"].tolist()
    retriever = create_retriever(vector_store=vector_store, k=k)
    retrieved_docs = retriever.batch(questions)

    correct_answers = 0
    total_questions = len(df)
    results_df = pd.DataFrame(columns=["question", "model_answer", "correct_answer", "correct"])

    for index, row in df.iterrows():
        question = row["question"]
        choices = list(row["choices"])
        correct_choice = _normalize_answer(row["answer"])

        docs_list = _docs_for_index(retrieved_docs, index, total_questions)
        context = _build_context(docs_list)

        try:
            model_answer = _call_answer_function(
                answer_generation_function,
                question,
                choices,
                client,
                context,
                docs_list,
            )
        except Exception:
            try:
                model_answer = _call_answer_function(
                    answer_generation_function,
                    question,
                    choices,
                    client,
                    context,
                    docs_list,
                )
            except Exception:
                model_answer = ("Error", "Error")

        correct = model_answer[0] == correct_choice
        if correct:
            correct_answers += 1

        result_row = {
            "question": question,
            "model_answer": model_answer[0],
            "correct_answer": correct_choice,
            "correct": correct,
        }
        if "subject" in row:
            result_row["subject"] = row["subject"]
            if "subject" not in results_df.columns:
                results_df["subject"] = pd.Series(dtype="object")

        results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)

        if debugging:
            print(f"Q{index}: {question}")
            print(f"Model Answer: {model_answer[0]}, Correct Answer: {correct_choice}\n")

    accuracy = correct_answers / total_questions if total_questions else 0.0
    return accuracy, results_df
