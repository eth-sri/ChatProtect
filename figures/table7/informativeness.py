import sys
import subprocess
import pathlib

SET = "test"
GROUND_TRUTH = True
DESC_DIR_NEW = pathlib.Path(f"./test/{SET}/new_descriptions/")
DESC_DIR_ORIG = pathlib.Path(f"./test/{SET}/descriptions/")
SENT_DIR = pathlib.Path(f"./test/{SET}/sentences/")
model = "chatgpt"
sent_model = "chatgpt"

until_map = {
    -1: "Original",
    1: "Naively drop",
    2: "Iteration 1",
    3: "Iteration 2",
    4: "Iteration 3",
    5: "Final (ours)",
}
for mode in [
    -1,
    2,
    3,
    4,
    5,
    1,
]:
    res = subprocess.run(
        [
            sys.executable,
            "pipeline/measure_facts.py",
            "--set",
            SET,
            "--test_description_dir",
            DESC_DIR_NEW if mode >= 0 else DESC_DIR_ORIG,
            "--test_sentence_dir",
            SENT_DIR,
            "--mode",
            str(mode),
            "--alm-model",
            model,
            "--glm-model",
            sent_model,
            *(["--ground-truth"] if GROUND_TRUTH else []),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    orig_res = subprocess.run(
        [
            sys.executable,
            "pipeline/measure_facts.py",
            "--set",
            SET,
            "--test_description_dir",
            DESC_DIR_ORIG,
            "--test_sentence_dir",
            SENT_DIR,
            "--mode",
            "-1",
            "--alm-model",
            model,
            "--glm-model",
            sent_model,
            *(["--ground-truth"] if GROUND_TRUTH else []),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    res = res.stdout.splitlines()[-1]
    res = eval(res)
    orig_res = orig_res.stdout.splitlines()[-1]
    orig_res = eval(orig_res)
    print(until_map[mode], f"{100*res[0] / orig_res[0]:.1f}")
