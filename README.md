<div align="center">
<img  src="web/static/images/chatprotect-logo.svg" width="340" />
<h2>Catch and Revise Hallucinations in Large Language Models</h2>
</div>

This is the code for the paper ["Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation"](https://arxiv.org/abs/2305.15852).

## Installation

This project was tested with python3.10.
Set up a virtual environment (or use conda) and install the requirements for this project:
```bash
$ conda create -n chatprotect pip pytorch python=3.10
$ conda activate chatprotect
$ python3 -m pip install -r requirements.txt
$ python3 -m pip install -e .
```

Create a `secret.py` and enter required API keys

```bash
$ cp secret_template.py secret.py
$ <use your favorite editor to set api keys>
```

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

Then install the requirements. You need python 3.6 and pytorch. We recommend creating a seperate conda environment for this.

```bash
$ conda create -n CompactIE pip python=3.6 pytorch=1.9.0  -c pytorch
$ conda activate CompactIE
$ pip install transformers==4.2.2 configargparse==1.2.3 bidict==0.20.0 PyYAML==6.0.1
```

Run the following command to start the API and return to the root directory and ChatProtect conda environment.
Keep this process running and continue with [Running](#running) to run ChatProtect.


```bash
$ python api.py --config_file config.yml
```

## Running

Follow the below instructions to run the pipeline and website or reproduce the results locally.

#### Running complete process

First install the whole pipeline as described in the section Installation.
Then run the full pipeline on a singular topic via

```bash
$ python3 -m chatprotect --prompt "Please tell me about Thomas Chapais"
```

#### Running the website API

This API provides the required streams to interact with the demo website.

```bash
uvicorn pipeline.api:app --reload --port 9913
```

#### Running pipeline step by step

To reproduce the results with GPT-4, ChatGPT, Llama-2-70b-chat and Vicuna-13B-1.1 you will need to set up
FastChat, the Together AI API key and the OpenAI API key as described above.

In order to be able to assess each step of the pipeline and for scaling, the whole pipeline is split into several
seperate scripts that may be run step-by-step.
This corresponds to the steps `gLM.gen_sentence`, `aLM.detect` and `aLM.revise`

```bash
$ # generate answers to prompt
$ python3 pipeline/0_generate_descriptions.py --prompt "Please tell me about Thomas Chapais"
$ # generate sentence + alternative sentences pairs /w tag for inconsistency (gen_sentence + detect, Figure 1 + 2)
$ python3 pipeline/1_generate_sentences.py --prompt "Please tell me about Thomas Chapais"
$ # generate new descriptions based on the original description and the tags (first step of revise, Figure 3)
$ python3 pipeline/2_generate_new_descriptions.py --prompt "Please tell me about Thomas Chapais"
$ # automatically execute further mitigation steps
$ bash pipeline/mitigation.sh --prompt "Please tell me about Thomas Chapais" --test_description_dir test/custom/new_descriptions

$ # run only a specific detect implementation (detect, Figure 2)
$ python3 pipeline/direct_sentences.py --prompt "Please tell me about Thomas Chapais"
```

Each script has more information about its parameters (such as employed aLM or gLM) displayed via `--help`.
Note the default sentence generation method is specialized for descriptions generation.
You may use the generalized prompt by changing the sentence method to 4.

#### Reproducing results from paper

To calculate the numbers presented in the figures in the paper, we will provide the data and simple bash scripts to compute the numbers from the data.
This will be added to the repository soon.
