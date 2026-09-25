import os
import json
import csv
import time
from pathlib import Path

from dotenv import load_dotenv
from rag import RAGSystem


load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path("evaluation/golden_questions.csv")
PROMPT_VERSION = os.getenv("SAARAI_PROMPT_VERSION", "v1")

RESULTS_PATH = Path(f"evaluation/results_{PROMPT_VERSION}.json")
SUMMARY_PATH = Path(f"evaluation/summary_{PROMPT_VERSION}.json")


MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

# LLM judge can be disabled if needed
USE_LLM_JUDGE = True


# ============================================================
# LLM JUDGE
# ============================================================

def judge(question, reference, answer):

    if not USE_LLM_JUDGE:
        return {
            "faithfulness": None,
            "relevance": None,
            "completeness": None,
            "judge_reason": "LLM judge disabled"
        }

    try:

        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        prompt = f"""
You are an evaluation judge for a Retrieval-Augmented
Generation (RAG) system.

Evaluate the candidate answer against the reference answer.

Give each criterion a score from 0 to 2:

Faithfulness:
0 = contradicts or invents information
1 = partially faithful
2 = fully faithful

Relevance:
0 = does not answer the question
1 = partially answers the question
2 = directly answers the question

Completeness:
0 = major information missing
1 = partially complete
2 = sufficiently complete

Return ONLY valid JSON in this format:

{{
  "faithfulness": 0,
  "relevance": 0,
  "completeness": 0,
  "reason": "brief explanation"
}}

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

CANDIDATE ANSWER:
{answer}
"""

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=250
            )
        )

        text = (
            response.text
            if response.text
            else ""
        )

        # Remove possible markdown code fences
        text = (
            text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        result = json.loads(text)

        return {
            "faithfulness": result.get("faithfulness"),
            "relevance": result.get("relevance"),
            "completeness": result.get("completeness"),
            "judge_reason": result.get(
                "reason",
                ""
            )
        }

    except Exception as e:

        return {
            "faithfulness": None,
            "relevance": None,
            "completeness": None,
            "judge_reason": (
                f"judge unavailable: {str(e)}"
            )
        }


# ============================================================
# SAFE AVERAGE
# ============================================================

def average(values):

    valid = [
        float(v)
        for v in values
        if v is not None
    ]

    if not valid:
        return None

    return round(
        sum(valid) / len(valid),
        3
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 60)
    print(f"SaarAI — RAG Evaluation Harness ({PROMPT_VERSION})")
    print("=" * 60)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_PATH.exists():

        print(
            f"ERROR: Dataset not found: "
            f"{DATASET_PATH}"
        )

        return

    # --------------------------------------------------------
    # Load RAG system
    # --------------------------------------------------------

    print("\nLoading RAG system...")

    rag = RAGSystem()

    # --------------------------------------------------------
    # Load golden dataset
    # --------------------------------------------------------

    with DATASET_PATH.open(
        encoding="utf-8"
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    total_questions = len(rows)

    print(
        f"Golden questions loaded: "
        f"{total_questions}"
    )

    print(
        f"Evaluation model: "
        f"{MODEL_NAME}"
    )

    print("\nStarting evaluation...\n")

    # --------------------------------------------------------
    # Evaluation containers
    # --------------------------------------------------------

    results = []

    total_tokens = 0
    total_cost = 0
    latencies = []

    retrieval_success = 0
    answer_success = 0

    faithfulness_scores = []
    relevance_scores = []
    completeness_scores = []

    # --------------------------------------------------------
    # Run every question
    # --------------------------------------------------------

    for index, row in enumerate(
        rows,
        start=1
    ):

        question = (
            row.get("question", "")
            .strip()
        )

        reference = (
            row.get("reference_answer", "")
            .strip()
        )

        question_id = (
            row.get(
                "id",
                str(index)
            )
        )

        # Skip empty questions
        if not question:

            continue

        print(
            f"[{index}/{total_questions}] "
            f"{question}"
        )

        # ----------------------------------------------------
        # Run RAG
        # ----------------------------------------------------

        start = time.time()

        try:

            output = rag.answer(
                question,
                k=5
            )

        except Exception as e:

            output = {
                "answer": (
                    f"RAG execution failed: "
                    f"{str(e)}"
                ),
                "sources": [],
                "latency": round(
                    time.time() - start,
                    2
                ),
                "tokens": 0,
                "cost": 0
            }

        # ----------------------------------------------------
        # Extract metrics
        # ----------------------------------------------------

        candidate_answer = output.get(
            "answer",
            ""
        )

        sources = output.get(
            "sources",
            []
        )

        latency = float(
            output.get(
                "latency",
                0
            ) or 0
        )

        tokens = int(
            output.get(
                "tokens",
                0
            ) or 0
        )

        cost = float(
            output.get(
                "cost",
                0
            ) or 0
        )

        # ----------------------------------------------------
        # Retrieval success
        # ----------------------------------------------------

        retrieved = len(
            sources
        ) > 0

        if retrieved:

            retrieval_success += 1

        # ----------------------------------------------------
        # Answer success
        # ----------------------------------------------------

        failed_phrases = [
            "could not find",
            "could not generate",
            "execution failed",
            "outside the scope"
        ]

        answer_ok = (
            bool(candidate_answer)
            and not any(
                phrase in candidate_answer.lower()
                for phrase in failed_phrases
            )
        )

        if answer_ok:

            answer_success += 1

        # ----------------------------------------------------
        # LLM evaluation
        # ----------------------------------------------------

        evaluation = judge(
            question,
            reference,
            candidate_answer
        )

        faithfulness = evaluation.get(
            "faithfulness"
        )

        relevance = evaluation.get(
            "relevance"
        )

        completeness = evaluation.get(
            "completeness"
        )

        if faithfulness is not None:
            faithfulness_scores.append(
                faithfulness
            )

        if relevance is not None:
            relevance_scores.append(
                relevance
            )

        if completeness is not None:
            completeness_scores.append(
                completeness
            )

        # ----------------------------------------------------
        # Aggregate metrics
        # ----------------------------------------------------

        total_tokens += tokens
        total_cost += cost

        if latency > 0:
            latencies.append(latency)

        # ----------------------------------------------------
        # Store complete result
        # ----------------------------------------------------

        result = {
            "id": question_id,
            "prompt_version": PROMPT_VERSION,
            "question": question,
            "reference_answer": reference,
            "candidate_answer": candidate_answer,

            "retrieval": {
                "success": retrieved,
                "source_count": len(sources),
                "sources": sources
            },

            "performance": {
                "latency_seconds": latency,
                "tokens": tokens,
                "estimated_cost_usd": cost
            },

            "evaluation": {
                "faithfulness": faithfulness,
                "relevance": relevance,
                "completeness": completeness,
                "reason": evaluation.get(
                    "judge_reason",
                    ""
                )
            }
        }

        results.append(result)

        print(
            f"   Retrieval: "
            f"{'PASS' if retrieved else 'FAIL'}"
        )

        print(
            f"   Latency: {latency}s"
        )

        print(
            f"   Tokens: {tokens}"
        )

        print()

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    evaluated = len(results)

    avg_latency = average([
        r["performance"]["latency_seconds"]
        for r in results
        if r["performance"]["latency_seconds"] > 0
    ])

    avg_tokens = average([
        r["performance"]["tokens"]
        for r in results
        if r["performance"]["tokens"] > 0
    ])

    # Maximum possible score = 2
    avg_faithfulness = average(
        faithfulness_scores
    )

    avg_relevance = average(
        relevance_scores
    )

    avg_completeness = average(
        completeness_scores
    )

    # Convert 0–2 scores to percentage
    def percentage(score):

        if score is None:
            return None

        return round(
            (score / 2) * 100,
            2
        )

    summary = {

        "evaluation": {
            "dataset": str(
                DATASET_PATH
            ),
            "questions_in_dataset":
                total_questions,
            "questions_evaluated":
                evaluated,
            "model":
                MODEL_NAME,
            "prompt_version":
                PROMPT_VERSION
        },

        "retrieval": {
            "successful_queries":
                retrieval_success,
            "retrieval_success_rate":
                round(
                    (
                        retrieval_success
                        / evaluated
                        * 100
                    )
                    if evaluated
                    else 0,
                    2
                )
        },

        "answer_generation": {
            "successful_answers":
                answer_success,
            "answer_success_rate":
                round(
                    (
                        answer_success
                        / evaluated
                        * 100
                    )
                    if evaluated
                    else 0,
                    2
                )
        },

        "quality": {
            "faithfulness_score":
                avg_faithfulness,
            "faithfulness_percentage":
                percentage(
                    avg_faithfulness
                ),

            "relevance_score":
                avg_relevance,
            "relevance_percentage":
                percentage(
                    avg_relevance
                ),

            "completeness_score":
                avg_completeness,
            "completeness_percentage":
                percentage(
                    avg_completeness
                )
        },

        "performance": {
            "average_latency_seconds":
                avg_latency,
            "average_tokens":
                avg_tokens,
            "total_tokens":
                total_tokens,
            "total_estimated_cost_usd":
                round(
                    total_cost,
                    8
                )
        },

        "timestamp":
            time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )
    }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_PATH.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # OPTIONAL PROMPT COMPARISON
    # --------------------------------------------------------
    # If both prompt-version summaries exist, create a compact
    # comparison file automatically.
    comparison_path = Path("evaluation/prompt_comparison.json")
    v1_summary_path = Path("evaluation/summary_v1.json")
    v2_summary_path = Path("evaluation/summary_v2.json")

    if v1_summary_path.exists() and v2_summary_path.exists():
        try:
            v1 = json.loads(
                v1_summary_path.read_text(encoding="utf-8")
            )
            v2 = json.loads(
                v2_summary_path.read_text(encoding="utf-8")
            )

            def metric(summary_obj, *keys):
                value = summary_obj
                for key in keys:
                    value = value.get(key, {}) if isinstance(value, dict) else {}
                return value

            v1_metrics = {
                "retrieval_success_rate": metric(v1, "retrieval", "retrieval_success_rate"),
                "answer_success_rate": metric(v1, "answer_generation", "answer_success_rate"),
                "faithfulness_percentage": metric(v1, "quality", "faithfulness_percentage"),
                "relevance_percentage": metric(v1, "quality", "relevance_percentage"),
                "completeness_percentage": metric(v1, "quality", "completeness_percentage"),
                "average_latency_seconds": metric(v1, "performance", "average_latency_seconds"),
                "average_tokens": metric(v1, "performance", "average_tokens"),
                "total_tokens": metric(v1, "performance", "total_tokens"),
                "total_estimated_cost_usd": metric(v1, "performance", "total_estimated_cost_usd")
            }

            v2_metrics = {
                "retrieval_success_rate": metric(v2, "retrieval", "retrieval_success_rate"),
                "answer_success_rate": metric(v2, "answer_generation", "answer_success_rate"),
                "faithfulness_percentage": metric(v2, "quality", "faithfulness_percentage"),
                "relevance_percentage": metric(v2, "quality", "relevance_percentage"),
                "completeness_percentage": metric(v2, "quality", "completeness_percentage"),
                "average_latency_seconds": metric(v2, "performance", "average_latency_seconds"),
                "average_tokens": metric(v2, "performance", "average_tokens"),
                "total_tokens": metric(v2, "performance", "total_tokens"),
                "total_estimated_cost_usd": metric(v2, "performance", "total_estimated_cost_usd")
            }

            comparison = {
                "dataset": str(DATASET_PATH),
                "questions": max(
                    v1.get("evaluation", {}).get("questions_evaluated", 0),
                    v2.get("evaluation", {}).get("questions_evaluated", 0)
                ),
                "versions": {
                    "v1": v1_metrics,
                    "v2": v2_metrics
                },
                "delta_v2_minus_v1": {
                    key: (
                        round(v2_metrics[key] - v1_metrics[key], 4)
                        if isinstance(v2_metrics[key], (int, float))
                        and isinstance(v1_metrics[key], (int, float))
                        else None
                    )
                    for key in v1_metrics
                },
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            comparison_path.write_text(
                json.dumps(
                    comparison,
                    indent=2,
                    ensure_ascii=False
                ),
                encoding="utf-8"
            )

            print(f"  {comparison_path}")

        except Exception as e:
            print(f"  Prompt comparison not generated: {e}")

    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Questions evaluated: "
        f"{evaluated}"
    )

    print(
        f"Retrieval success: "
        f"{summary['retrieval']['retrieval_success_rate']}%"
    )

    print(
        f"Answer success: "
        f"{summary['answer_generation']['answer_success_rate']}%"
    )

    print(
        f"Faithfulness: "
        f"{summary['quality']['faithfulness_percentage']}%"
    )

    print(
        f"Relevance: "
        f"{summary['quality']['relevance_percentage']}%"
    )

    print(
        f"Completeness: "
        f"{summary['quality']['completeness_percentage']}%"
    )

    print(
        f"Average latency: "
        f"{avg_latency}s"
    )

    print(
        f"Total tokens: "
        f"{total_tokens}"
    )

    print(
        f"Estimated cost: "
        f"${total_cost:.8f}"
    )

    print("\nSaved:")
    print(
        f"  {RESULTS_PATH}"
    )
    print(
        f"  {SUMMARY_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()