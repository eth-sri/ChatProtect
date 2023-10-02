import sys
import subprocess
import pathlib
from ast import literal_eval

from chatprotect.util import read_entities

SET = "test"
DESC_DIR_ORIG = pathlib.Path(f"./test/{SET}/descriptions/")
DESC_DIR_NEW = pathlib.Path(f"./test/{SET}/new_descriptions/")

mode = 5
for MODEL1, MODEL2 in (
    ("chatgpt", "gpt4"),
    ("chatgpt", "chatgpt"),
    ("chatgpt", "llama-2-70b-chat"),
    ("chatgpt", "vicuna-13b-1.1"),
):
    test_description_dir = (
        DESC_DIR_NEW / MODEL1 if mode >= 0 else DESC_DIR_ORIG
    ) / MODEL2
    res = subprocess.run(
        [
            sys.executable,
            "pipeline/measure_perplexity.py",
            "--mode",
            str(mode),
            "--test_description_dir",
            str(test_description_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    line = res.stdout.splitlines()[-1]
    result = literal_eval(line)
    print(MODEL2, round(result, 2))
