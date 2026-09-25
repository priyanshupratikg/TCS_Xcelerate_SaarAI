# Evaluation
`golden_questions.csv` contains 100 representative questions for the demo corpus.
`evaluate.py` runs each question through the RAG system and optionally uses Gemini as an LLM judge.
The resulting `results.json` can be summarized into faithfulness, relevance and completeness metrics.
For the final submission, replace the demo questions with 100 questions written against your actual official institutional PDFs.
