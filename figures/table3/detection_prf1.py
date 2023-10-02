import sys
import subprocess

SET = "test"

mode = -1
for model, sent_model in [
    ("chatgpt", "gpt4"),
    ("chatgpt", "chatgpt"),
]:
    DIR = f"output/test/{model}/{sent_model}/s3/test_m3"
    SENT_DIR = f"test/{SET}/sentences/{sent_model}"
    res = subprocess.run(
        [
            sys.executable,
            "pipeline/measure_results.py",
            "--test_dir",
            DIR,
            "--test_sentence_dir",
            SENT_DIR,
            "--prf1",
            "--old",
            "--override_sent",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    res = res.stdout.splitlines()[-1]
    res = eval(res)
    res = (round(x * 100, 1) for x in res)
    print(sent_model, ",".join(map(str, res)), flush=True)
for model, sent_model in [
    ("chatgpt", "llama-2-70b-chat"),
]:
    DIR = f"output/test/{model}/{sent_model}/s3/test_m3"
    SENT_DIR = f"test/{SET}/sentences/{sent_model}"
    res = subprocess.run(
        [
            sys.executable,
            "pipeline/measure_results_json.py",
            "--test_dir",
            DIR,
            "--prf1",
            "--override_sent",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    res = res.stdout.splitlines()[-1]
    res = eval(res)
    res = (round(x * 100, 1) for x in res)
    print(sent_model, ",".join(map(str, res)), flush=True)
for model, sent_model in [
    ("chatgpt", "vicuna-13b-1.1"),
]:
    DIR = f"output/test/{model}/{sent_model}/s3/test_m3"
    SENT_DIR = f"test/{SET}/sentences/{sent_model}"
    res = subprocess.run(
        [
            sys.executable,
            "pipeline/measure_results.py",
            "--test_dir",
            DIR,
            "--test_sentence_dir",
            SENT_DIR,
            "--prf1",
            "--old",
            "--override_sent",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    res = res.stdout.splitlines()[-1]
    res = eval(res)
    res = (round(x * 100, 1) for x in res)
    print(sent_model, ",".join(map(str, res)), flush=True)
