"""
Step 12: Evaluation harness
-----------------------------
Runs a CSV of test questions (single-document, cross-document and
deliberately unanswerable) through the pipeline and reports:

  * Retrieval accuracy   - automatic: does the top retrieved chunk come
                            from one of the expected source document(s)?
  * Grounding correctness - automatic: for questions expected to be
                            unanswerable, did the system correctly refuse
                            instead of hallucinating?
  * answer_correct / citation_correct - left BLANK for a human grader to
    fill in, because judging free-text answer correctness reliably needs
    human (or careful LLM-judge) review, not a hard-coded rule.

This mirrors exactly the evaluation the problem statement asks for:
"measure grounded answers versus hallucinated answers" and "evaluate
citation accuracy" — the script gives you the scaffolding and the
automatically-checkable half; you fill in the judged half after reading
the transcript.
"""
import csv
from typing import List

from .rag import DocuMindRAG, NOT_FOUND_MESSAGE


def run_evaluation(rag: DocuMindRAG, csv_path: str, out_csv_path: str = "eval_results.csv"):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        questions = list(reader)

    n_retrieval_correct = 0
    n_grounding_correct = 0
    n_answerable = 0
    n_unanswerable = 0

    for row in questions:
        qid = row["id"]
        category = row["category"]
        question = row["question"]
        expected_sources = [s.strip() for s in row["expected_source"].split(";") if s.strip()]
        expect_answerable = expected_sources != ["NONE"]

        result = rag.answer(question)
        top_doc = result.retrieved[0].doc_name if result.retrieved else None

        retrieval_correct = ""
        if expect_answerable:
            n_answerable += 1
            retrieval_correct = bool(top_doc and top_doc in expected_sources)
            n_retrieval_correct += int(retrieval_correct)
        else:
            n_unanswerable += 1
            # for an unanswerable question, "correct" behaviour is refusing
            grounding_ok = not result.grounded
            n_grounding_correct += int(grounding_ok)

        rows.append({
            "id": qid,
            "category": category,
            "question": question,
            "expected_source": row["expected_source"],
            "top_retrieved_doc": top_doc or "",
            "top_score": round(result.scores[0], 3) if result.scores else "",
            "retrieval_correct(auto)": retrieval_correct,
            "system_answer": result.answer,
            "grounded": result.grounded,
            "citation": "; ".join(result.citations),
            "answer_correct(fill in manually)": "",
            "citation_correct(fill in manually)": "",
        })

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote detailed results to: {out_csv_path}\n")
    if n_answerable:
        print(f"Retrieval accuracy (answerable questions): "
              f"{n_retrieval_correct}/{n_answerable} = {n_retrieval_correct / n_answerable:.0%}")
    if n_unanswerable:
        print(f"Correct refusal rate (unanswerable questions): "
              f"{n_grounding_correct}/{n_unanswerable} = {n_grounding_correct / n_unanswerable:.0%}")
    print("\nanswer_correct / citation_correct columns are left blank for manual grading —")
    print("open eval_results.csv, read each system_answer against the source PDF, and mark Y/N.")
    return rows
