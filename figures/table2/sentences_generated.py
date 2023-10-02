import pathlib

from chatprotect.util import read_desc_file, split_sentences

if __name__ == "__main__":
    testset = "test"
    mode = 3
    alm_model = "chatgpt"

    for glm_model in reversed(
        ["vicuna-13b-1.1", "llama-2-70b-chat", "chatgpt", "gpt4"]
    ):
        test_description_dir = pathlib.Path(f"test/{testset}/descriptions") / glm_model
        i = 0
        num_desc = 0
        for target_dir in test_description_dir.iterdir():
            for test_file in sorted(target_dir.iterdir()):
                try:
                    desc = read_desc_file(None, test_file)
                except ValueError:
                    continue
                split_desc = split_sentences(desc.text)
                i += len(split_desc)
                num_desc += 1
        print(glm_model, f"{i / num_desc:.1f}")
