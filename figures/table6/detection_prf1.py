import sys
import subprocess

SET = "test"

mode = -1
for model, sent_model in [
    ("chatgpt", "chatgpt"),
]:
    for sent_mode in [1, 10, 4, 3]:
        DIR = f"output/test/{model}/{sent_model}/s3/test_m{sent_mode}"
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
        res = list(round(x * 100, 1) for x in res)
        print(",".join(map(str, res)), flush=True)
