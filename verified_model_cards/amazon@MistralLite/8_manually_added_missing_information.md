## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
Developed by: [AWS Contributors](https://github.com/orgs/aws-samples/teams/aws-prototype-ml-apac)

### Model date:
Not available.

### Model version:
MistralLite

### Model type:
MistralLite is a fine-tuned [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) language model, with enhanced capabilities of processing long context (up to 32K tokens). By utilizing an adapted Rotary Embedding and sliding window during fine-tuning, MistralLite is able to **perform significantly better on several long context retrieve and answering tasks**, while keeping the simple model structure of the original model. MistralLite is similar to [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1). Their similarities and differences are summarized below:

|Model|Fine-tuned on long contexts| Max context length| RotaryEmbedding adaptation| Sliding Window Size|
|----------|-------------:|------------:|-----------:|-----------:|
| Mistral-7B-Instruct-v0.1 | up to 8K tokens | 32K | rope_theta = 10000 | 4096 |
| MistralLite | up to 16K tokens | 32K | **rope_theta = 1000000** | **16384** |

### Training details:
Finetuned from weights: [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
By utilizing an adapted Rotary Embedding and sliding window during fine-tuning.

### Paper or other resource for more information:
- [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Huggingface Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference)
- [vLLM](https://github.com/vllm-project/vllm)
- [HuggingFace transformers](https://huggingface.co/docs/transformers/index)
- [FlashAttention-2](https://github.com/Dao-AILab/flash-attention)

### Citation details:
Not available.

### License:
Model License: Apache 2.0

### Contact:
[GitHub issues](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/issues)

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
MistralLite is useful for applications such as long context line and topic retrieval, summarization, question-answering, and etc. MistralLite is able to **perform significantly better on several long context retrieve and answering tasks**.

### Primary intended users:
MistralLite is suitable for applications that require high performance in resource-constrained environments. MistralLite can be deployed on a single AWS `g5.2x` instance with Sagemaker [Huggingface Text Generation Inference (TGI)](https://github.com/huggingface/text-generation-inference) endpoint.

### Out-of-scope uses:
Not available.

---

## How to Use
This section outlines how to use the model.

**Important - Use the prompt template below for MistralLite:**

```<|prompter|>What are the main challenges to support a long context for LLM?</s><|assistant|>```

## How to Use MistralLite from Python Code (HuggingFace transformers) ##

**Important** - For an end-to-end example Jupyter notebook, please refer to [this link](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/huggingface-transformers/example_usage.ipynb).

### Install the necessary packages

Requires: [transformers](https://pypi.org/project/transformers/) 4.34.0 or later, [flash-attn](https://pypi.org/project/flash-attn/) 2.3.1.post1 or later,
and [accelerate](https://pypi.org/project/accelerate/) 0.23.0 or later.

```shell
pip install transformers==4.34.0
pip install flash-attn==2.3.1.post1 --no-build-isolation
pip install accelerate==0.23.0
```
### You can then try the following example code

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch

model_id = "amazon/MistralLite"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             use_flash_attention_2=True,
                                             device_map="auto",)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
prompt = "<|prompter|>What are the main challenges to support a long context for LLM?</s><|assistant|>"

sequences = pipeline(
    prompt,
    max_new_tokens=400,
    do_sample=False,
    return_full_text=False,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
for seq in sequences:
    print(f"{seq['generated_text']}")
```
**Important** - Use the prompt template below for MistralLite:
```
<|prompter|>What are the main challenges to support a long context for LLM?</s><|assistant|>
```

## How to Serve MistralLite on TGI ##
**Important:**
- For an end-to-end example Jupyter notebook using the native TGI container, please refer to [this link](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/tgi/example_usage.ipynb).
- If the **input context length is greater than 12K tokens**, it is recommended using a custom TGI container, please refer to [this link](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/tgi-custom/example_usage.ipynb).

### Start TGI server ###
Use TGI version 1.1.0 or later. The official Docker container is: `ghcr.io/huggingface/text-generation-inference:1.1.0`

Example Docker parameters:

```shell
docker run -d --gpus all --shm-size 1g -p 443:80 -v $(pwd)/models:/data ghcr.io/huggingface/text-generation-inference:1.1.0 \
      --model-id amazon/MistralLite \
      --max-input-length 16000 \
      --max-total-tokens 16384 \
      --max-batch-prefill-tokens 16384 \
      --trust-remote-code
```

### Perform Inference ###
Example Python code for inference with TGI (requires `text_generation` 0.6.1 or later):

```shell
pip install text_generation==0.6.1
```

```python
from text_generation import Client

SERVER_PORT = 443
SERVER_HOST = "localhost"
SERVER_URL = f"{SERVER_HOST}:{SERVER_PORT}"
tgi_client = Client(f"http://{SERVER_URL}", timeout=60)

def invoke_tgi(prompt,
                      random_seed=1,
                      max_new_tokens=400,
                      print_stream=True,
                      assist_role=True):
    if (assist_role):
        prompt = f"<|prompter|>{prompt}</s><|assistant|>"
    output = ""
    for response in tgi_client.generate_stream(
        prompt,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
        #temperature=None,
        #truncate=None,
        #seed=random_seed,
        #typical_p=0.2,
    ):
        if hasattr(response, "token"):
            if not response.token.special:
                snippet = response.token.text
                output += snippet
                if (print_stream):
                    print(snippet, end='', flush=True)
    return output

prompt = "What are the main challenges to support a long context for LLM?"
result = invoke_tgi(prompt)
```

**Important** - When using MistralLite for inference for the first time, it may require a brief 'warm-up' period that can take 10s of seconds. However, subsequent inferences should be faster and return results in a more timely manner. This warm-up period is normal and should not affect the overall performance of the system once the initialisation period has been completed.


## How to Deploy MistralLite on Amazon SageMaker ##
**Important:**
- For an end-to-end example Jupyter notebook using the SageMaker built-in container, please refer to [this link](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/sagemaker-tgi/example_usage.ipynb).
- If the **input context length is greater than 12K tokens**, it is recommended using a custom docker container, please refer to [this link](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/sagemaker-tgi-custom/example_usage.ipynb).

### Install the necessary packages

Requires: [sagemaker](https://pypi.org/project/sagemaker/) 2.192.1 or later.

```shell
pip install sagemaker==2.192.1
```

### Deploy the Model as A SageMaker Endpoint ###
To deploy MistralLite on a SageMaker endpoint, please follow the example code as below.
```python
import sagemaker
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
import time

sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
role = sagemaker.get_execution_role()

image_uri = get_huggingface_llm_image_uri(
  backend="huggingface", # or lmi
  region=region,
 version="1.1.0"
)

model_name = "MistralLite-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

hub = {
    'HF_MODEL_ID':'amazon/MistralLite',
    'HF_TASK':'text-generation',
    'SM_NUM_GPUS':'1',
    "MAX_INPUT_LENGTH": '16000',
    "MAX_TOTAL_TOKENS": '16384',
    "MAX_BATCH_PREFILL_TOKENS": '16384',
    "MAX_BATCH_TOTAL_TOKENS":  '16384',
}

model = HuggingFaceModel(
    name=model_name,
    env=hub,
    role=role,
    image_uri=image_uri
)
predictor = model.deploy(
  initial_instance_count=1,
  instance_type="ml.g5.2xlarge",
  endpoint_name=model_name,

)
```

### Perform Inference ###
To call the endpoint, please follow the example code as below:

```python
input_data = {
  "inputs": "<|prompter|>What are the main challenges to support a long context for LLM?</s><|assistant|>",
  "parameters": {
    "do_sample": False,
    "max_new_tokens": 400,
    "return_full_text": False,
    #"typical_p": 0.2,
    #"temperature":None,
    #"truncate":None,
    #"seed": 1,
  }
}
result = predictor.predict(input_data)[0]["generated_text"]
print(result)
```
or via [boto3](https://pypi.org/project/boto3/), and the example code is shown as below:

```python
import boto3
import json
def call_endpoint(client, prompt, endpoint_name, paramters):
    client = boto3.client("sagemaker-runtime")
    payload = {"inputs": prompt,
               "parameters": parameters}
    response = client.invoke_endpoint(EndpointName=endpoint_name,
                                      Body=json.dumps(payload),
                                      ContentType="application/json")
    output = json.loads(response["Body"].read().decode())
    result = output[0]["generated_text"]
    return result

client = boto3.client("sagemaker-runtime")
parameters = {
    "do_sample": False,
    "max_new_tokens": 400,
    "return_full_text": False,
    #"typical_p": 0.2,
    #"temperature":None,
    #"truncate":None,
    #"seed": 1,
}
endpoint_name = predictor.endpoint_name
prompt = "<|prompter|>What are the main challenges to support a long context for LLM?</s><|assistant|>"
result = call_endpoint(client, prompt, endpoint_name, parameters)
print(result)
```


## How to Serve MistralLite on vLLM ##
Documentation on installing and using vLLM [can be found here](https://vllm.readthedocs.io/en/latest/).

**Important** - For an end-to-end example Jupyter notebook, please refer to [this link](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/vllm/example_usage.ipynb).

### Using vLLM as a server ###
When using vLLM as a server, pass the --model amazon/MistralLite parameter, for example:
```shell
python3 -m vllm.entrypoints.api_server --model amazon/MistralLite
```

### Using vLLM in Python Code ###
When using vLLM from Python code, Please see the example code as below:

```python
from vllm import LLM, SamplingParams

prompts = [
   "<|prompter|>What are the main challenges to support a long context for LLM?</s><|assistant|>",
]
sampling_params = SamplingParams(temperature=0, max_tokens=100)

llm = LLM(model="amazon/MistralLite",)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Long context is a relevant factor that influences the model’s performance. The model is motivated by improving performance on longer context tasks.

### Evaluation factors:
Input length is an evaluation factor analyzed and reported during model evaluation.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
Metrics used to assess the model’s effectiveness are: Accuracy, and retrieval rate (%). Also, Average, hellaswag, arc_challenge, truthful_qa, MMLU scores are used for evaluation.

### Decision thresholds:
Not available.

### Variation approaches:
Methodology is described at: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard, revision=4ececff. Note: we used --model hf-causal-experimental instead of --model hf-causal

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
- [Topic Retrieval](https://lmsys.org/blog/2023-06-29-longchat/)
- [Line Retrieval](https://lmsys.org/blog/2023-06-29-longchat/#longeval-results)
- [Pass key Retrieval](https://github.com/epfml/landmark-attention/blob/main/llama/run_test.py#L101)
- [Question Answering with Long Input Texts](https://nyu-mll.github.io/quality/)

### Motivation:
These datasets were chosen for evaluation because most of the benchmarks are evaluated on `short context`, and not much has been investigated on its performance on long context tasks. These benchmarks are specifically designed to assess the capabilities of LLMs in handling longer context.

### Preprocessing:
Not available.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
- [SLidingEncoder and Decoder (SLED)](https://huggingface.co/datasets/tau/sled)
- [(Long) Natural Questions (NQ)](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections#multi-passage-qa-from-natural-questions)
- [OpenAssistant Conversations Dataset (OASST1)](https://huggingface.co/datasets/OpenAssistant/oasst1)

### Motivation:
The datasets were chosen for training to improve the model's performance on longer context.

### Preprocessing:
Rotary Embedding and sliding window adaptation were utilized during fine-tuning.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:

The detailed `long context evalutaion results` are as below: 

1. [Topic Retrieval](https://lmsys.org/blog/2023-06-29-longchat/)

|Model Name|Input length| Input length | Input length| Input length| Input length|
|----------|-------------:|-------------:|------------:|-----------:|-----------:|
|          | 2851| 5568 |8313 | 11044 | 13780 |
|   Mistral-7B-Instruct-v0.1  | 100%        | 50%       | 2%      | 0%     | 0% |
|   MistralLite   | **100%**        | **100%**       | **100%**      | **100%**     | **98%** |

2. [Line Retrieval](https://lmsys.org/blog/2023-06-29-longchat/#longeval-results)

|Model Name|Input length| Input length | Input length| Input length| Input length|Input length|
|----------|-------------:|-------------:|------------:|-----------:|-----------:|-----------:|
|          | 3818| 5661 |7505 | 9354 | 11188 | 12657
|   Mistral-7B-Instruct-v0.1   | **98%**        | 62%       | 42%      | 42%     | 32% | 30% |
|   MistralLite   | **98%**        | **92%**       | **88%**      | **76%**     | **70%** | **60%** |

3. [Pass key Retrieval](https://github.com/epfml/landmark-attention/blob/main/llama/run_test.py#L101)

|Model Name|Input length| Input length | Input length| Input length|
|----------|-------------:|-------------:|------------:|-----------:|
|          | 3264| 5396 |8329 | 10197 |
|   Mistral-7B-Instruct-v0.1   | **100%**        | 50%       | 20%      | 30%   |
|   MistralLite  | **100%**        | **100%**       | **100%**      | **100%**   |

4. [Question Answering with Long Input Texts](https://nyu-mll.github.io/quality/)
|Model Name| Test set Accuracy | Hard subset Accuracy|
|----------|-------------:|-------------:|
| Mistral-7B-Instruct-v0.1 | 44.3% | 39.7% |
| MistralLite | **64.4%** | **56.2%** |

### MistralLite LM-Eval Results

|Average|hellaswag| arc_challenge|truthful_qa (mc2)| MMLU (acc)|
|----------|-------------:|------------:|-----------:|-----------:|
| 0.57221 | 0.81617 | 0.58874 | 0.38275 | 0.5012 |

### Intersectional results:
Not available.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
Not available.

### Deploying Requirements:
MistralLite can be deployed on a single AWS `g5.2x` instance with Sagemaker [Huggingface Text Generation Inference (TGI)] endpoint. You can also serve the MistralLite model directly using TGI docker containers, other ways of serving like vLLM, in Python by using the HuggingFace transformers and FlashAttention-2 library.
To deploy MistralLite on a SageMaker endpoint, `ml.g5.2xlarge` instance type is used in the example.

### Training or Fine-tuning Requirements:
Not available.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

Before using the MistralLite model, it is important to perform your own independent assessment, and take measures to ensure that your use would comply with your own specific quality control practices and standards, and that your use would comply with the local rules, laws, regulations, licenses and terms that apply to you, and your content.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
- **Important** - When using MistralLite for inference for the first time, it may require a brief 'warm-up' period that can take 10s of seconds.
- If the **input context length is greater than 12K tokens**, it is recommended using a custom TGI container.

### Recommendations:
- **Important - Use the prompt template below for MistralLite:** `<|prompter|>What are the main challenges to support a long context for LLM?</s><|assistant|>`
- For an end-to-end example Jupyter notebook for HuggingFace transformers, please refer to [this link](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/huggingface-transformers/example_usage.ipynb).
- For an end-to-end example Jupyter notebook using the native TGI container, please refer to [this link](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/tgi/example_usage.ipynb).
- If the input context length is greater than 12K tokens and using TGI, please refer to [this link](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/tgi-custom/example_usage.ipynb).
- For an end-to-end example Jupyter notebook using the SageMaker built-in container, please refer to [this link](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/sagemaker-tgi/example_usage.ipynb).
- If the input context length is greater than 12K tokens and using SageMaker, please refer to [this link](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/sagemaker-tgi-custom/example_usage.ipynb).
- For an end-to-end example Jupyter notebook for vLLM, please refer to [this link](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/vllm/example_usage.ipynb).

## Additional Information
## Motivation of Developing MistralLite
Since the release of [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), the model became increasingly popular because its strong performance on a wide range of benchmarks. Although the performance of the models on long context was fairly competitive on long context less than 4096 tokens, 
there were some limitations on its performance on longer context. Mistrallite managed to `significantly boost the performance of long context handling` over Mistral-7B-Instruct-v0.1.

### Example test on long context of 13400 tokens
- Context is from [Amazon Aurora FAQs](https://aws.amazon.com/rds/aurora/faqs/)
- Question: *please tell me how does pgvector help with Generative AI and give me some examples.*
- Answer from MistralLite:
  ```code
  pgvector is an open-source extension for PostgreSQL supported by Amazon Aurora PostgreSQL-Compatible Edition.

  You can use pgvector to store, search, index, and query billions of embeddings that are generated from machine learning (ML) and artificial intelligence (AI) models in your database, such as those from Amazon Bedrock (limited preview) or Amazon SageMaker. A vector embedding is a numerical representation that represents the semantic meaning of content such as text, images, and video.

  With pgvector, you can query embeddings in your Aurora PostgreSQL database to perform efficient semantic similarity searches of these data types, represented as vectors, combined with other tabular data in Aurora. This enables the use of generative AI and other AI/ML systems for new types of applications such as personalized recommendations based on similar text descriptions or images, candidate match based on interview notes, customer service next best action recommendations based on successful transcripts or chat session dialogs, and more.
  ```

## Model Details
- **Language:** English
- **Supported Serving Framework:**
  - [Text-Generation-Inference 1.1.0](https://github.com/huggingface/text-generation-inference/tree/v1.1.0)
  - [vLLM](https://github.com/vllm-project/vllm)
  - [HuggingFace transformers](https://huggingface.co/docs/transformers/index)
  - [HuggingFace Text Generation Inference (TGI) container on SageMaker](https://github.com/awslabs/llm-hosting-container)
- **Inference Code** [Github Repo](https://github.com/awslabs/extending-the-context-length-of-open-source-llms/blob/main/MistralLite/)