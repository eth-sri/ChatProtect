import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chatprotect.consistency import check_consistent_cot, explain_consistent_cot
from chatprotect.sentences import (
    extract_triples_compact_ie,
    generate_statement_missing_object_free,
)
from chatprotect.util import fetch_model, prompt_identifier, split_sentences


MODEL_SPECS = [
    ("Llama-3.1-8B-Instruct", "llama3.1-8b-instruct"),
    ("Gemma-3-12B-Instruct", "gemma3-12b-instruct"),
]


def load_dataset(path: pathlib.Path):
    with path.open() as f:
        dataset = json.load(f)
    return [entry["question"] for entry in dataset]


def configure_batch_bot(bot, model_name: str):
    if "openrouter/" in getattr(bot, "model", ""):
        bot.default_provider = {
            "order": ["deepinfra"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    return bot


def ask_question(question: str, model_name: str) -> str:
    bot = configure_batch_bot(fetch_model(model_name), model_name)
    with bot as bot_session:
        bot_session.set_deterministic(True)
        bot_session.set_num_answers(1)
        return bot_session.ask(question)[0]


def analyze_response(question: str, response: str, glm_name: str, alm_name: str):
    generator_bot = configure_batch_bot(fetch_model(glm_name), glm_name)
    analyzer_bot = configure_batch_bot(fetch_model(alm_name), alm_name)
    sentence_results = []
    prefix = ""
    for sentence in split_sentences(response):
        triples = extract_triples_compact_ie(sentence)
        for triple in triples:
            alternative = generate_statement_missing_object_free(
                generator_bot, triple[0], triple[1], question, prefix
            )[0]
            explanation = explain_consistent_cot(
                analyzer_bot, sentence, alternative, question, prefix
            )[0]
            score = check_consistent_cot(
                analyzer_bot, sentence, alternative, question, prefix, explanation
            )
            sentence_results.append(
                {
                    "sentence": sentence,
                    "triple": list(triple),
                    "alternative": alternative,
                    "explanation": explanation,
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


def cache_path(cache_dir: pathlib.Path, model_label: str, question: str) -> pathlib.Path:
    filename = f"{prompt_identifier(model_label)}__{prompt_identifier(question)}.json"
    return cache_dir / filename


def format_submission_entry(cached_entry: dict):
    entry = {
        "model": cached_entry["model"],
        "question": cached_entry["question"],
        "model_response": cached_entry["model_response"],
        "hallucination": cached_entry["hallucination"],
    }
    possibility = cached_entry.get("hallucination_possibility(%)")
    if possibility is not None:
        entry["hallucination_possibility(%)"] = possibility
    return entry


def load_cached_entry(path: pathlib.Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def write_cached_entry(path: pathlib.Path, entry: dict):
    with path.open("w") as f:
        json.dump(entry, f, indent=2)


def write_submission_file(cache_dir: pathlib.Path, output_file: pathlib.Path):
    cached_entries = []
    for path in sorted(cache_dir.glob("*.json")):
        with path.open() as f:
            entry = json.load(f)
        if entry.get("status") != "complete":
            continue
        cached_entries.append(entry)
    submission = [format_submission_entry(entry) for entry in cached_entries]
    with output_file.open("w") as f:
        json.dump(submission, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="dataset_sample.json",
        help="Path to dataset json containing question entries",
    )
    parser.add_argument(
        "--output",
        default="output/dataset_submission.json",
        help="Path to final submission json",
    )
    parser.add_argument(
        "--cache-dir",
        default="output/dataset_submission_cache",
        help="Directory for per-question resumable cache files",
    )
    parser.add_argument(
        "--alm-model",
        default=None,
        help="Override analysis model for contradiction detection",
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

    dataset_path = pathlib.Path(args.dataset)
    output_file = pathlib.Path(args.output)
    cache_dir = pathlib.Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    questions = load_dataset(dataset_path)
    if args.limit is not None:
        questions = questions[: args.limit]

    total = len(questions) * len(MODEL_SPECS)
    completed = 0
    for question in questions:
        for model_label, glm_name in MODEL_SPECS:
            target_cache_path = cache_path(cache_dir, model_label, question)
            cached_entry = load_cached_entry(target_cache_path)
            if (
                cached_entry is not None
                and cached_entry.get("status") == "complete"
                and not args.force
            ):
                completed += 1
                print(f"[{completed}/{total}] skipping cached {model_label}: {question}")
                continue

            alm_name = args.alm_model or glm_name
            print(f"[{completed + 1}/{total}] running {model_label}: {question}")
            if cached_entry is not None and not args.force:
                model_response = cached_entry["model_response"]
            else:
                model_response = ask_question(question, glm_name)
                cached_entry = {
                    "status": "generated",
                    "model": model_label,
                    "question": question,
                    "model_response": model_response,
                    "generator_model_id": glm_name,
                    "analyzer_model_id": alm_name,
                    "sentence_results": [],
                }
                write_cached_entry(target_cache_path, cached_entry)

            analysis = analyze_response(question, model_response, glm_name, alm_name)
            cached_entry = {
                "status": "complete",
                "model": model_label,
                "question": question,
                "model_response": model_response,
                "hallucination": analysis["hallucination"],
                "hallucination_possibility(%)": analysis[
                    "hallucination_possibility(%)"
                ],
                "generator_model_id": glm_name,
                "analyzer_model_id": alm_name,
                "sentence_results": analysis["sentence_results"],
            }
            write_cached_entry(target_cache_path, cached_entry)
            write_submission_file(cache_dir, output_file)
            completed += 1

    write_submission_file(cache_dir, output_file)
    print(f"Wrote submission file to {output_file}")


if __name__ == "__main__":
    main()
