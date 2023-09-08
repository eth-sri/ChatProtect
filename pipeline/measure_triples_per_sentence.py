from argparse import ArgumentParser
import pathlib


from chatprotect.util import (
    read_desc_file,
    read_sent_file,
    split_sentences,
)
from collections import defaultdict

parser = ArgumentParser()
parser.add_argument(
    "--prompt",
    type=str,
    help="Prompt for which to measure triples per sentence",
    default="Please tell me about Thomas Chapais",
)
parser.add_argument(
    "--test_description_dir",
    type=str,
    help="Input dir",
    default="./test/test/descriptions",
)
parser.add_argument(
    "--test_sentence_dir", type=str, help="Input dir", default="./test/test/sentences"
)
parser.add_argument(
    "--model",
    type=str,
    help="select a model to run the test against",
    default="chatgpt",
)
parser.add_argument(
    "--mode",
    choices=["native", "compactie"],
    help="select the mode of triple extraction",
    default="native",
)
args = parser.parse_args()


mode = args.mode
model = args.model

# parse sentences
sents = defaultdict(list)
test_sentence_dir = pathlib.Path(args.test_sentence_dir) / model
test_sentence_dir.mkdir(parents=True, exist_ok=True)
for ent_dir in test_sentence_dir.iterdir():
    for test_file in (ent_dir / "m3").iterdir():
        try:
            ext_sent = read_sent_file(None, test_file)
            sents[ext_sent.orig].append(ext_sent)
        except ValueError:
            continue

input_texts = []
OK, STRONG = 0, 1
facts = [0, 0]
test_description_dir = pathlib.Path(args.test_description_dir) / model
len_descriptions = []
triples_per_sent = defaultdict(int)
test_description_dir.mkdir(parents=True, exist_ok=True)
for ent_dir in sorted(test_description_dir.iterdir()):
    for desc_file in sorted(ent_dir.iterdir()):
        try:
            desc = read_desc_file(None, desc_file)
        except ValueError:
            continue
        desc_sents = split_sentences(desc)
        len_descriptions.append(len(desc_sents))
        for sent in desc_sents:
            ext_sents = sents.get(sent, [])
            filtered_ext_sents = [s for s in ext_sents if s.triple[1] == mode]
            triples_per_sent[len(filtered_ext_sents)] += 1

print("triples / sentence")
print(f"{sum(i*j for i,j in triples_per_sent.items())/sum(triples_per_sent.values())}")
table = "\n".join(",".join(map(str, x)) for x in sorted(triples_per_sent.items()))
print(table)
