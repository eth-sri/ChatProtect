import sys
import subprocess
import pathlib
from ast import literal_eval

from chatprotect.util import read_entities

SET = "test"
DESC_DIR_ORIG = pathlib.Path(f"./test/{SET}/descriptions/")
DESC_DIR_NEW = pathlib.Path(f"./test/{SET}/new_descriptions/")

until_map = {
    1: "Naively drop",
    2: "Iteration 1",
    3: "Iteration 2",
    4: "Iteration 3",
    5: "Final (ours)",
}
MODEL1, MODEL2 = "chatgpt", "chatgpt"
for mode in (1, 2, 3, 4, 5):
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
    print(until_map[mode], round(result, 2))
