"""Variant of run_dataset_submission.py using the Figure 6 setting from the paper:
sample 20 alternative answers per triple and pass them all to check_factual_multi_score,
which asks whether any evidence contradicts the original statement.
"""
import argparse
import concurrent.futures
import json
import pathlib
import sys
import threading

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatprotect.consistency import check_factual_multi_score
from chatprotect.sentences import (
    extract_triples_compact_ie,
    generate_statement_missing_object_free,
)
from chatprotect.util import fetch_model, split_sentences


MODEL_SPECS = [
    ("Llama-3.1-8B-Instruct", "llama3.1-8b-instruct", {"temperature": 0.6, "top_p": 0.9, "max_tokens": 1000}),
    ("Gemma-3-12B-Instruct", "gemma3-12b-instruct", {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "max_tokens": 1000}),
]

NUM_ALTS = 20


def load_dataset(path: pathlib.Path):
    with path.open() as f:
        dataset = json.load(f)
    return [entry["question"] for entry in dataset]


def configure_batch_bot(bot, model_name: str, sampling_params: dict = None):
    if "openrouter/" in getattr(bot, "model", ""):
        bot.default_provider = {
            "order": ["deepinfra"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    if sampling_params:
        if "top_k" in sampling_params:
            bot.default_top_k = sampling_params["top_k"]
        if "top_p" in sampling_params:
            bot.default_top_p = sampling_params["top_p"]
        if "max_tokens" in sampling_params:
            bot.default_max_tokens = sampling_params["max_tokens"]
    return bot


def ask_question(question: str, model_name: str) -> str:
    bot = configure_batch_bot(fetch_model(model_name), model_name)
    with bot as bot_session:
        bot_session.set_deterministic(True)
        bot_session.set_num_answers(1)
        return bot_session.ask(question)[0]


def analyze_response(
    question: str,
    response: str,
    glm_name: str,
    alm_name: str,
    sampling_params: dict = None,
    num_alts: int = NUM_ALTS,
):
    sampling_params = sampling_params or {}
    temperature = sampling_params.get("temperature", 0.5)
    generator_bot = configure_batch_bot(fetch_model(glm_name), glm_name, sampling_params)
    analyzer_bot = configure_batch_bot(fetch_model(alm_name), alm_name)
    sentence_results = []
    prefix = ""
    for sentence in split_sentences(response):
        triples = extract_triples_compact_ie(sentence)
        for triple in triples:
            alternatives = []
            for _ in range(num_alts):
                alt = generate_statement_missing_object_free(
                    generator_bot,
                    triple[0],
                    triple[1],
                    question,
                    prefix,
                    override_temperature=temperature,
                    alts=1,
                )
                a = alt[0]
                alternatives.append(a[:500] + "..." if len(a) > 500 else a)
            score = check_factual_multi_score(
                analyzer_bot, sentence, alternatives, question, prefix
            )
            sentence_results.append(
                {
                    "sentence": sentence,
                    "triple": list(triple),
                    "alternatives": alternatives,
                    "score": score,
                    "hallucination": score > 0.5,
                }
            )
        prefix = f"{prefix} {sentence}".strip()
    if sentence_results:
        possibility = max(item["score"] for item in sentence_results)
        hallucination = any(item["hallucination"] for item in sentence_results)
    else:
        possibility = None
        hallucination = False
    return {
        "hallucination": hallucination,
        "hallucination_possibility(%)": possibility,
        "sentence_results": sentence_results,
    }


def load_cache(cache_file: pathlib.Path) -> dict:
    if not cache_file.exists():
        return {}
    with cache_file.open() as f:
        return json.load(f)


def save_cache(cache_file: pathlib.Path, cache_data: dict):
    with cache_file.open("w") as f:
        json.dump(cache_data, f, indent=2)


def migrate_cache(old_cache_dir: pathlib.Path, cache_file: pathlib.Path):
    """Migrate per-file cache directory to single JSON cache keyed by question."""
    cache_data = {}
    for path in sorted(old_cache_dir.glob("*.json")):
        with path.open() as f:
            entry = json.load(f)
        question = entry.get("question")
        model = entry.get("model")
        if question and model:
            cache_data.setdefault(question, {})[model] = entry
    save_cache(cache_file, cache_data)
    n = sum(len(v) for v in cache_data.values())
    print(f"Migrated {n} entries from {old_cache_dir} to {cache_file}")
    return cache_data


def format_submission_entry(entry: dict):
    result = {
        "model": entry["model"],
        "question": entry["question"],
        "model_response": entry["model_response"],
        "hallucination": entry["hallucination"],
    }
    possibility = entry.get("hallucination_possibility(%)")
    if possibility is not None:
        result["hallucination_possibility(%)"] = possibility
    return result


def write_submission_file(cache_data: dict, output_file: pathlib.Path):
    submission = [
        format_submission_entry(entry)
        for question_entries in cache_data.values()
        for entry in question_entries.values()
        if entry.get("status") == "complete"
    ]
    with output_file.open("w") as f:
        json.dump(submission, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Figure 6 variant: sample NUM_ALTS alternatives per triple and "
        "detect contradictions with check_factual_multi_score."
    )
    parser.add_argument(
        "--dataset",
        default="dataset_sample.json",
        help="Path to dataset json containing question entries",
    )
    parser.add_argument(
        "--output",
        default="output/dataset_submission_fig6.json",
        help="Path to final submission json",
    )
    parser.add_argument(
        "--cache-file",
        default="output/dataset_submission_fig6_cache.json",
        help="Single JSON cache file keyed by question",
    )
    parser.add_argument(
        "--migrate-from",
        default=None,
        help="Migrate an old per-file cache directory into the new single-file format and exit",
    )
    parser.add_argument(
        "--alm-model",
        default=None,
        help="Override analysis model for contradiction detection",
    )
    parser.add_argument(
        "--num-alts",
        type=int,
        default=NUM_ALTS,
        help="Number of alternative answers to sample per triple (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached entries",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of dataset questions for a partial run",
    )
    args = parser.parse_args()

    cache_file = pathlib.Path(args.cache_file)
    output_file = pathlib.Path(args.output)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.migrate_from:
        migrate_cache(pathlib.Path(args.migrate_from), cache_file)
        return

    cache_data = load_cache(cache_file)
    questions = load_dataset(pathlib.Path(args.dataset))
    if args.limit is not None:
        questions = questions[: args.limit]

    total = len(questions) * len(MODEL_SPECS)
    completed = 0
    lock = threading.Lock()

    def run_one(question, model_label, glm_name, sampling_params):
        nonlocal completed

        with lock:
            cached_entry = cache_data.get(question, {}).get(model_label)

        if (
            cached_entry is not None
            and cached_entry.get("status") == "complete"
            and not args.force
        ):
            with lock:
                completed += 1
                print(f"[{completed}/{total}] skipping cached {model_label}: {question}")
            return

        alm_name = args.alm_model or glm_name
        print(f"running {model_label}: {question}")

        if cached_entry is not None and not args.force:
            model_response = cached_entry["model_response"]
        else:
            model_response = ask_question(question, glm_name)
            with lock:
                cache_data.setdefault(question, {})[model_label] = {
                    "status": "generated",
                    "model": model_label,
                    "question": question,
                    "model_response": model_response,
                    "generator_model_id": glm_name,
                    "analyzer_model_id": alm_name,
                    "sentence_results": [],
                }
                save_cache(cache_file, cache_data)

        try:
            analysis = analyze_response(
                question,
                model_response,
                glm_name,
                alm_name,
                sampling_params=sampling_params,
                num_alts=args.num_alts,
            )
        except Exception as e:
            print(f"skipping {model_label}: {question} — {e}")
            return
        with lock:
            cache_data.setdefault(question, {})[model_label] = {
                "status": "complete",
                "model": model_label,
                "question": question,
                "model_response": model_response,
                "hallucination": analysis["hallucination"],
                "hallucination_possibility(%)": analysis["hallucination_possibility(%)"],
                "generator_model_id": glm_name,
                "analyzer_model_id": alm_name,
                "num_alts": args.num_alts,
                "sampling_params": sampling_params,
                "sentence_results": analysis["sentence_results"],
            }
            completed += 1
            print(f"[{completed}/{total}] done {model_label}: {question}")
            save_cache(cache_file, cache_data)
            write_submission_file(cache_data, output_file)

    def run_model(model_label, glm_name, sampling_params):
        for question in questions:
            run_one(question, model_label, glm_name, sampling_params)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODEL_SPECS)) as executor:
        futures = [executor.submit(run_model, *spec) for spec in MODEL_SPECS]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    write_submission_file(cache_data, output_file)
    print(f"Wrote submission file to {output_file}")


if __name__ == "__main__":
    main()
