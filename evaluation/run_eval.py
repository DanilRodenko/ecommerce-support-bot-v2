"""Offline evaluation of the RAG bot.

Metrics:
- retrieval hit@3: for in-scope questions, is the chunk with the answer among the top-3 retrieved?
- answer accuracy: an LLM judge compares the bot's answer with the expected answer.
- refusal accuracy: for out-of-scope questions, does the bot correctly say it has no information?

Run from the project root:
    python -m evaluation.run_eval
Needs GROQ_API_KEY in .env. Not part of CI (costs API calls, answers are non-deterministic).
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings

from src.chain import DEFAULT_MODEL, ask
from src.ingest import process_file
from src.retriever import retrieve

load_dotenv()

EVAL_DIR = Path(__file__).parent
KB_FILES = {
    "clothing": "data/uploads/clothing_faq.txt",
    "electronics": "data/uploads/electronics_faq.txt",
}
MODEL = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", DEFAULT_MODEL)
PAUSE_SECONDS = 2  # stay under the free-tier rate limit

JUDGE_PROMPT = """You are grading a customer support bot.

Question: {question}
Expected answer: {expected}
Bot answer: {answer}

If the expected answer says the bot should not have the information, the bot answer is
CORRECT only if it clearly says it doesn't have that information and does not make anything up.
Otherwise the bot answer is CORRECT if it contains the key facts of the expected answer
and does not contradict it. Extra polite wording is fine.

Reply with exactly one word: CORRECT or INCORRECT."""


def judge(client, question, expected, answer):
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, answer=answer)
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = response.choices[0].message.content.strip().upper()
    return verdict.startswith("CORRECT")


def pct(part, total):
    return f"{part}/{total} ({part / total:.0%})" if total else "n/a"


def main():
    items = json.loads((EVAL_DIR / "eval_set.json").read_text(encoding="utf-8"))
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    stores = {kb: process_file(path, embeddings) for kb, path in KB_FILES.items()}

    results = []
    for i, item in enumerate(items, 1):
        store = stores[item["kb"]]

        hit = None
        if item["in_scope"]:
            chunks = retrieve(store, item["question"])
            hit = any(item["keyword"].lower() in c.page_content.lower() for c in chunks)

        answer, _ = ask(item["question"], store, client, MODEL)
        time.sleep(PAUSE_SECONDS)
        correct = judge(client, item["question"], item["expected"], answer)
        time.sleep(PAUSE_SECONDS)

        results.append({**item, "retrieval_hit": hit, "answer": answer, "correct": correct})
        mark = "OK " if correct else "BAD"
        print(f"[{i:02d}/{len(items)}] {mark} {item['kb']:<11} {item['question']}")

    in_scope = [r for r in results if r["in_scope"]]
    out_scope = [r for r in results if not r["in_scope"]]
    hits = sum(r["retrieval_hit"] for r in in_scope)
    correct_in = sum(r["correct"] for r in in_scope)
    refusals = sum(r["correct"] for r in out_scope)
    correct_all = sum(r["correct"] for r in results)

    print("\n| Metric | Result |")
    print("|---|---|")
    print(f"| Retrieval hit@3 (in-scope) | {pct(hits, len(in_scope))} |")
    print(f"| Answer accuracy (in-scope) | {pct(correct_in, len(in_scope))} |")
    print(f"| Correct refusals (out-of-scope) | {pct(refusals, len(out_scope))} |")
    print(f"| Overall accuracy | {pct(correct_all, len(results))} |")
    print(f"\nModel: {MODEL}, judge: {JUDGE_MODEL}, questions: {len(results)}")

    out_path = EVAL_DIR / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Per-question results saved to {out_path}")


if __name__ == "__main__":
    main()