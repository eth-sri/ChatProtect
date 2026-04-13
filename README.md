<div align="center">
<img  src="web/static/images/chatprotect-logo.svg" width="340" />
<h2>Catch and Revise Hallucinations in Large Language Models</h2>
</div>

This is the code for the paper ["Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation"](https://arxiv.org/abs/2305.15852).
An easy-to-use website presenting the tool and its use-cases is hosted at https://chatprotect.ai/. 

## Installation

This project was tested with python3.10.
Set up a virtual environment with `uv` and install the requirements for this project:
```bash
$ uv venv --python 3.10 .venv
$ uv sync
$ .venv/bin/python -m spacy download en_core_web_sm
$ .venv/bin/python - <<'PY'
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
PY
```

Create a `secret.py` and enter required API keys

```bash
$ cp secret_template.py secret.py
$ <use your favorite editor to set api keys>
```

If you use an OpenAI-compatible proxy, also set `OPENAI_BASE_URL` in `secret.py`.
This repository includes aliases for the proxy-backed chat models `llama3.1-8b-instruct` and `gemma3-12b-instruct`.
These currently resolve to `openrouter/meta-llama/llama-3.1-8b-instruct` and `gemini/gemma-3-12b-it`.

#### Using CompactIE for triple extraction

Set up [CompactIE](https://aclanthology.org/2022.naacl-main.65/) from the attached zip in this repository.
The zip file contains a forked version that provides a local API for triple extraction.

```bash
$ unzip CompactIE.zip -d CompactIE
$ cd CompactIE
```

Models checkpoint are available in [Zenodo](https://zenodo.org/record/6804440).
Download the Constituent Extraction (`ce_model`) model and put in in the folder [`save_results/models/constituent/`](https://github.com/FarimaFatahi/CompactIE/tree/master/save_results/models/constituent/).
Download the Constituent Linking (`cl_model`) model and put in under [`save_results/models/relation/`](https://github.com/FarimaFatahi/CompactIE/tree/master/save_results/models/relation/) folder.

```bash
$ wget https://zenodo.org/record/6804440/files/ce_model?download=1
$ mv ce_model?download=1 save_results/models/constituent/ce_model
$ wget https://zenodo.org/record/6804440/files/cl_model?download=1
$ mv cl_model?download=1 save_results/models/relation/cl_model
```

Then install the requirements. CompactIE is a separate legacy service, so keep it in its own `uv` environment.
The upstream project targeted python 3.6 and pytorch 1.9.0; on modern machines, use the closest python version available to you in `uv` and install the pinned python packages from the paper setup.
On Apple Silicon, the legacy `transformers==4.2.2` stack may still fail because it depends on `tokenizers==0.9.4`, which does not provide a native wheel.

```bash
$ uv venv --python 3.8 .venv
$ uv sync
```

Run the following command to start the API and return to the root directory and ChatProtect `uv` environment.
Keep this process running and continue with [Running](#running) to run ChatProtect.


```bash
$ .venv/bin/python api.py --config_file config.yml
```

## Running

Follow the below instructions to run the pipeline and website or reproduce the results locally.

#### Running complete process

First install the whole pipeline as described in the section Installation.
Then run the full pipeline on a singular topic via

```bash
$ .venv/bin/python -m chatprotect --glm llama3.1-8b-instruct --alm gemma3-12b-instruct --prompt "Please tell me about Thomas Chapais"
```

#### Running the website API

This API provides the required streams to interact with the demo website.

```bash
.venv/bin/uvicorn pipeline.api:app --reload --port 9113
```

#### Running pipeline step by step

To reproduce the results with GPT-4, ChatGPT, Llama-2-70b-chat and Vicuna-13B-1.1 you will need to set up
FastChat, the Together AI API key and the OpenAI API key as described above.

In order to be able to assess each step of the pipeline and for scaling, the whole pipeline is split into several
seperate scripts that may be run step-by-step.
This corresponds to the steps `gLM.gen_sentence`, `aLM.detect` and `aLM.revise`

```bash
$ # generate answers to prompt
$ .venv/bin/python pipeline/0_generate_descriptions.py --prompt "Please tell me about Thomas Chapais"
$ # generate sentence + alternative sentences pairs /w tag for inconsistency (gen_sentence + detect, Figure 1 + 2)
$ .venv/bin/python pipeline/1_generate_sentences.py --prompt "Please tell me about Thomas Chapais"
$ # generate new descriptions based on the original description and the tags (first step of revise, Figure 3)
$ .venv/bin/python pipeline/2_generate_new_descriptions.py --prompt "Please tell me about Thomas Chapais"
$ # automatically execute further mitigation steps
$ bash pipeline/mitigation.sh --prompt "Please tell me about Thomas Chapais" --test_description_dir test/custom/new_descriptions

$ # run only a specific detect implementation (detect, Figure 2)
$ .venv/bin/python pipeline/direct_sentences.py --prompt "Please tell me about Thomas Chapais"
```

Each script has more information about its parameters (such as employed aLM or gLM) displayed via `--help`.
Note the default sentence generation method is specialized for descriptions generation.
You may use the generalized prompt by changing the sentence method to 4.

#### Reproducing results from paper

To calculate the numbers presented in the figures in the paper, run the bash scripts in the `figures`
directory from root like this.
To compute the perplexity values, you will need a GPU with at least 5GB VRAM or the computation will be quite slow. The computation is disabled by default.
```bash
$ bash figures/run.sh
```
