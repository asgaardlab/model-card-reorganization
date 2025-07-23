## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
**Model Developer:** NVIDIA

### Model date:
**Model Dates:** Nemotron-4-340B-Instruct was trained between December 2023 and May 2024.

### Model version:
Nemotron-4-340B-Instruct is a fine-tuned version of the Nemotron-4-340B-Base model, optimized for English-based single and multi-turn chat use-cases.

### Model type:
Nemotron-4-340B-Instruct is a large language model (LLM) that can be used as part of a synthetic data generation pipeline to create training data that helps researchers and developers build their own LLMs. It is standard decoder-only Transformer, trained with a sequence length of 4096 tokens, uses Grouped-Query Attention (GQA), and Rotary Position Embeddings (RoPE).

**Architecture Type:** Transformer Decoder (auto-regressive language model)
**Network Architecture:** Nemotron-4

[![Model architecture](https://img.shields.io/badge/Model%20Arch-Transformer%20Decoder-green)](#model-architecture)[![Model size](https://img.shields.io/badge/Params-340B-green)](#model-architecture)[![Language](https://img.shields.io/badge/Language-Multilingual-green)](#datasets)

### Training details:
The base model was pre-trained on a corpus of 9 trillion tokens consisting of a diverse assortment of English based texts, 50+ natural languages, and 40+ coding languages. Subsequently the Nemotron-4-340B-Instruct model went through additional alignment steps including:

- Supervised Fine-tuning (SFT)
- Direct Preference Optimization (DPO)
- Reward-aware Preference Optimization (RPO) ([Additional in-house alignment technique](https://research.nvidia.com/publication/2024-06_nemotron-4-340b)) 

Throughout the alignment process, we relied on only approximately 20K human-annotated data while our data generation pipeline synthesized over 98% of the data used for supervised fine-tuning and preference fine-tuning (DPO & RPO).

**Data Freshness:** The pretraining data has a cutoff of June 2023.

### Paper or other resource for more information:
[technical report](https://research.nvidia.com/publication/2024-06_nemotron-4-340b)
[HelpSteer2 Dataset Paper](https://arxiv.org/abs/2406.08673)
[Instruction-Following Evaluation for Large Language Models](Not available.)
[Measuring Massive Multitask Language Understanding](Not available.)
[Training Verifiers to Solve Math Word Problems](Not available.)
[Evaluating Large Language Models Trained on Code](Not available.)
[Program Synthesis with Large Language Models](Not available.)
[Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators](Not available.)
[CantTalkAboutThis: Aligning Language Models to Stay on Topic in Dialogues](Not available.)
[NeMo Framework](https://docs.nvidia.com/nemo-framework/index.html)
[NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner)
[documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/nemotron/index.html)
[Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/nemotron-4-340b-instruct)

### Citation details:
Not available.

### License:
[NVIDIA Open Model License](https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf)

Under the NVIDIA Open Model License, NVIDIA confirms: 
- Models are commercially usable. 
- You are free to create and distribute Derivative Models. 
- NVIDIA does not claim ownership to any outputs generated using the Models or Derivative Models.

### Contact:
Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
Nemotron-4-340B-Instruct is a chat model intended for use for the English language. 

Nemotron-4-340B-Instruct is designed for Synthetic Data Generation to enable developers and enterprises for building and customizing their own large language models and LLM applications. 

The instruct model itself can be further customized using the [NeMo Framework](https://docs.nvidia.com/nemo-framework/index.html) suite of customization tools including Parameter-Efficient Fine-Tuning (P-tuning, Adapters, LoRA, and more), and Model Alignment (SFT, SteerLM, RLHF, and more) using [NeMo-Aligner](https://github.com/NVIDIA/NeMo-Aligner). Refer to the [documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/llms/nemotron/index.html) for examples.

### Primary intended users:
Nemotron-4-340B-Instruct is designed for Synthetic Data Generation to enable developers and enterprises for building and customizing their own large language models and LLM applications.

### Out-of-scope uses:
Not available.

---

## How to Use
This section outlines how to use the model. 

Deployment and inference with Nemotron-4-340B-Instruct can be done in three steps using NeMo Framework:

Create a Python script to interact with the deployed model.
Create a Bash script to start the inference server
Schedule a Slurm job to distribute the model across 2 nodes and associate them with the inference server.

1. Define the Python script ``call_server.py``

```python
import json
import requests

headers = {"Content-Type": "application/json"}

def text_generation(data, ip='localhost', port=None):
    resp = requests.put(f'http://{ip}:{port}/generate', data=json.dumps(data), headers=headers)
    return resp.json()


def get_generation(prompt, greedy, add_BOS, token_to_gen, min_tokens, temp, top_p, top_k, repetition, batch=False):
    data = {
        "sentences": [prompt] if not batch else prompt,
        "tokens_to_generate": int(token_to_gen),
        "temperature": temp,
        "add_BOS": add_BOS,
        "top_k": top_k,
        "top_p": top_p,
        "greedy": greedy,
        "all_probs": False,
        "repetition_penalty": repetition,
        "min_tokens_to_generate": int(min_tokens),
        "end_strings": ["<|endoftext|>", "<extra_id_1>", "\x11", "<extra_id_1>User"],
    }
    sentences = text_generation(data, port=1424)['sentences']
    return sentences[0] if not batch else sentences

PROMPT_TEMPLATE = """<extra_id_0>System

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
"""

question = "Write a poem on NVIDIA in the style of Shakespeare"
prompt = PROMPT_TEMPLATE.format(prompt=question)
print(prompt)

response = get_generation(prompt, greedy=True, add_BOS=False, token_to_gen=1024, min_tokens=1, temp=1.0, top_p=1.0, top_k=0, repetition=1.0, batch=False)
response = response[len(prompt):]
if response.endswith("<extra_id_1>"):
    response = response[:-len("<extra_id_1>")]
print(response)
```

2. Given this Python script, create a Bash script which spins up the inference server within the [NeMo container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) (```docker pull nvcr.io/nvidia/nemo:24.05```) and calls the Python script ``call_server.py``. The Bash script ``nemo_inference.sh`` is as follows,

```bash
NEMO_FILE=$1
WEB_PORT=1424

depends_on () {
    HOST=$1
    PORT=$2
    STATUS=$(curl -X PUT http://$HOST:$PORT >/dev/null 2>/dev/null; echo $?)
    while [ $STATUS -ne 0 ]
    do
         echo "waiting for server ($HOST:$PORT) to be up"
         sleep 10
         STATUS=$(curl -X PUT http://$HOST:$PORT >/dev/null 2>/dev/null; echo $?)
    done
    echo "server ($HOST:$PORT) is up running"
}


/usr/bin/python3 /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py \
        gpt_model_file=$NEMO_FILE \
        pipeline_model_parallel_split_rank=0 \
        server=True tensor_model_parallel_size=8 \
        trainer.precision=bf16 pipeline_model_parallel_size=2 \
        trainer.devices=8 \
        trainer.num_nodes=2 \
        web_server=False \
        port=${WEB_PORT} &
    SERVER_PID=$!

    readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
    if [ $SLURM_NODEID -eq 0 ] && [ $local_rank -eq 0 ]; then
        depends_on "0.0.0.0" ${WEB_PORT}

        echo "start get json"
        sleep 5

        echo "SLURM_NODEID: $SLURM_NODEID"
        echo "local_rank: $local_rank"
        /usr/bin/python3 /scripts/call_server.py
        echo "clean up dameons: $$"
        kill -9 $SERVER_PID
        pkill python
    fi
    wait
```


3. Launch ``nemo_inference.sh`` with a Slurm script defined like below, which starts a 2-node job for model inference.

```
#!/bin/bash
#SBATCH -A SLURM-ACCOUNT
#SBATCH -p SLURM-PARITION
#SBATCH -N 2
#SBATCH -J generation      
#SBATCH --ntasks-per-node=8   
#SBATCH --gpus-per-node=8
set -x

RESULTS=<PATH_TO_YOUR_SCRIPTS_FOLDER>
OUTFILE="${RESULTS}/slurm-%j-%n.out"
ERRFILE="${RESULTS}/error-%j-%n.out"
MODEL=<PATH_TO>/Nemotron-4-340B-Instruct
CONTAINER="nvcr.io/nvidia/nemo:24.05"
MOUNTS="--container-mounts=<PATH_TO_YOUR_SCRIPTS_FOLDER>:/scripts,MODEL:/model"

read -r -d '' cmd <<EOF
bash /scripts/nemo_inference.sh /model
EOF

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
```

#### Prompt Format

Note: For Nemotron-4-340B-Instruct we recommend keeping the system prompt empty.

##### Single Turn

```text
<extra_id_0>System

<extra_id_1>User
{prompt}
<extra_id_1>Assistant
```

##### Multi-Turn or Few-shot

```text
<extra_id_0>System

<extra_id_1>User
{prompt 1}
<extra_id_1>Assistant
{response 1}
<extra_id_1>User
{prompt 2}
<extra_id_1>Assistant
{response 2}
...
<extra_id_1>User
{prompt N}
<extra_id_1>Assistant
```

An example of a formattable prompt template is available above in this section.

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Not available.

### Evaluation factors:
Not available.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
Evaluated using MT-Bench judging by GPT-4-0125-Preview as described in Appendix H in the [HelpSteer2 Dataset Paper](https://arxiv.org/abs/2406.08673)

| total | writing | roleplay | extraction | stem | humanities | reasoning | math | coding | turn 1 | turn 2 |
| :----- | :------- | :-------- | :---------- | :---- | :---------- | :--------- | :---- | ------ | :------ | :------ | 
| 8.22 | 8.70 | 8.70  | 9.20 | 8.75 | 8.95 | 6.40 | 8.40 | 6.70 | 8.61 | 7.84 | 

Evaluated using the Instruction Following Eval (IFEval) introduced in Instruction-Following Evaluation for Large Language Models.

| Prompt-Strict Acc | Instruction-Strict Acc |
| :----------------------- | :---------------------------- |
| 79.9 | 86.1 |

Evaluated using the Multi-task Language Understanding benchmarks as introduced in Measuring Massive Multitask Language Understanding.

|MMLU 0-shot |
| :----------------- |
| 78.7  | 

Evaluated using the Grade School Math 8K (GSM8K) benchmark as introduced in Training Verifiers to Solve Math Word Problems.

| GSM8K 0-shot |
| :----------------- | 
| 92.3 | 

Evaluated using the HumanEval benchmark as introduced in Evaluating Large Language Models Trained on Code.


| HumanEval 0-shot |
| :----- |
| 73.2 |

Evaluated using the MBPP Dataset as introduced in the Program Synthesis with Large Language Models.

| MBPP 0-shot|
| :----------------- | 
| 75.4 | 


Evaluated using the Arena-Hard Pipeline from the LMSys Org.

| Arena Hard |
| :----------------- | 
| 54.2 | 

Evaluated using the AlpacaEval 2.0 LC (Length Controlled) as introduced in the paper: Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators

| AlpacaEval 2.0 LC|
| :----------------- | 
| 41.5 | 


Evaluated using the CantTalkAboutThis Dataset as introduced in the CantTalkAboutThis: Aligning Language Models to Stay on Topic in Dialogues.

| Distractor F1 | On-topic F1 |
| :----------------------- | :---------------------------- |
| 81.7  | 97.7 |

### Decision thresholds:
Not available.

### Variation approaches:
Not available.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
**MT-Bench (GPT-4-Turbo)**
**IFEval** (Instruction Following Eval)
**MMLU** (Multi-task Language Understanding benchmarks)
**GSM8K** (Grade School Math 8K benchmark)
**HumanEval**
**MBPP Dataset**
**Arena Hard**
**AlpacaEval 2.0 LC** (Length Controlled)
**TFEval** (CantTalkAboutThis Dataset)
**AEGIS** (content safety evaluation dataset and LLM based content safety classifier model)
**Garak** (automated LLM vulnerability scanner)
**Human Content Red Teaming**

### Motivation:
The Nemotron-4 340B-Instruct model underwent safety evaluation including adversarial testing via three distinct methods: 
- [Garak](https://docs.garak.ai/garak), is an automated LLM vulnerability scanner that probes for common weaknesses, including prompt injection and data leakage. 
- AEGIS, is a content safety evaluation dataset and LLM based content safety classifier model, that adheres to a broad taxonomy of 13 categories of critical risks in human-LLM interactions.
- Human Content Red Teaming leveraging human interaction and evaluation of the models' responses.

### Preprocessing:
Not available.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
The base model was pre-trained on a corpus of 9 trillion tokens consisting of a diverse assortment of English based texts, 50+ natural languages, and 40+ coding languages. Approximately 20K human-annotated data was used during alignment. Over 98% of the data used for supervised fine-tuning and preference fine-tuning (DPO & RPO) was synthesized by data generation pipeline.

### Motivation:
This results in a model that is aligned for human chat preferences, improvements in mathematical reasoning, coding and instruction-following, and is capable of generating high quality synthetic data for a variety of use cases.

### Preprocessing:
Not available.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
#### MT-Bench (GPT-4-Turbo)

| total | writing | roleplay | extraction | stem | humanities | reasoning | math | coding | turn 1 | turn 2 |
| :----- | :------- | :-------- | :---------- | :---- | :---------- | :--------- | :---- | ------ | :------ | :------ | 
| 8.22 | 8.70 | 8.70  | 9.20 | 8.75 | 8.95 | 6.40 | 8.40 | 6.70 | 8.61 | 7.84 | 

#### IFEval

| Prompt-Strict Acc | Instruction-Strict Acc |
| :----------------------- | :---------------------------- |
| 79.9 | 86.1 |

#### MMLU

|MMLU 0-shot |
| :----------------- |
| 78.7  | 

#### GSM8K

| GSM8K 0-shot |
| :----------------- | 
| 92.3 | 

#### HumanEval

| HumanEval 0-shot |
| :----- |
| 73.2 |

#### MBPP

| MBPP 0-shot|
| :----------------- | 
| 75.4 | 


#### Arena Hard

| Arena Hard |
| :----------------- | 
| 54.2 | 

#### AlpacaEval 2.0 LC

| AlpacaEval 2.0 LC|
| :----------------- | 
| 41.5 | 


#### TFEval

| Distractor F1 | On-topic F1 |
| :----------------------- | :---------------------------- |
| 81.7  | 97.7 |

### Intersectional results:
Not available.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
Not available.

### Deploying Requirements:
BF16 Inference:
- 8x H200 (1x H200 node)
- 16x H100 (2x H100 nodes)
- 16x A100 80GB (2x A100 80GB nodes)

### Training or Fine-tuning Requirements:
Not available.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions. 

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  For more detailed information on ethical considerations for this model, please see the Model Card++ Explainability, Bias, Safety & Security, and Privacy Subcards [here](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/nemotron-4-340b-instruct).

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
The model was trained on data that contains toxic language, unsafe content, and societal biases originally crawled from the internet. Therefore, the model may amplify those biases and return toxic responses especially when prompted with toxic prompts. The model may generate answers that may be inaccurate, omit key information, or include irrelevant or redundant text producing socially unacceptable or undesirable text, even if the prompt itself does not include anything explicitly offensive.

### Recommendations:
Not available.

---

## Additional Information
Try this model on [build.nvidia.com](https://build.nvidia.com/nvidia/nemotron-4-340b-instruct) now.