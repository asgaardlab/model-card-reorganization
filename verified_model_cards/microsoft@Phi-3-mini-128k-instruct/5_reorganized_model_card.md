```
## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
Microsoft (inferred from links and model names)

### Model date:
June, 2024. Our models were trained between May and June 2024.

### Model version:
This is an update over the original instruction-tuned Phi-3-mini release based on valuable customer feedback.

### Model type:
Phi-3 Mini-128K-Instruct is a 3.8 billion-parameter, lightweight, state-of-the-art open model trained using the Phi-3 datasets. It is a dense decoder-only Transformer model. The model belongs to the Phi-3 family with the Mini version in two variants 4K and 128K which is the context length (in tokens) that it can support.

### Training details:
After initial training, the model underwent a post-training process that involved supervised fine-tuning and direct preference optimization to enhance its ability to follow instructions and adhere to safety measures. The model is trained using the Phi-3 datasets. This dataset includes both synthetic data and filtered publicly available website data, with an emphasis on high-quality and reasoning-dense properties. Training data: 4.9T tokens. GPUs: 512 H100-80G. Training time: 10 days. Our models were trained between May and June 2024. This is a static model trained on an offline dataset with cutoff date October 2023.

### Paper or other resource for more information:
🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3) <br>
📰 [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024) <br>
📖 [Phi-3 Technical Report](https://aka.ms/phi3-tech-report) <br>
🛠️ [Phi-3 on Azure AI Studio](https://aka.ms/phi3-azure-ai) <br>
👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook) <br>
🖥️ [Try It](https://aka.ms/try-phi3)

### Citation details:
Not available.

### License:
The model is licensed under the [MIT license](https://huggingface.co/microsoft/Phi-3-mini-128k/resolve/main/LICENSE).

### Contact:
Not available.

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
The model is intended for commercial and research use in English. The model provides uses for applications which require:

1) Memory/compute constrained environments
2) Latency bound scenarios
3) Strong reasoning (especially code, math and logic)

Our model is designed to accelerate research on language and multimodal models, for use as a building block for generative AI powered features.

### Primary intended users:
Developers

### Out-of-scope uses:
Our models are not specifically designed or evaluated for all downstream purposes. Developers should consider common limitations of language models as they select use cases, and evaluate and mitigate for accuracy, safety, and fariness before using within a specific downstream use case, particularly for high risk scenarios. Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (ex: housing, employment, credit, etc.) without further assessments and additional debiasing techniques. High-Risk Scenarios: Developers should assess suitability of using models in high-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (ex: legal or health advice). Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.

---

## How to Use
This section outlines how to use the model.

Phi-3 Mini-128K-Instruct has been integrated in the development version (4.41.3) of `transformers`. Until the official version is released through `pip`, ensure that you are doing one of the following:

* When loading the model, ensure that `trust_remote_code=True` is passed as an argument of the `from_pretrained()` function.

* Update your local `transformers` to the development version: `pip uninstall -y transformers && pip install git+https://github.com/huggingface/transformers`. The previous command is an alternative to cloning and installing from the source.

The current `transformers` version can be verified with: `pip list | grep transformers`.

Examples of required packages:
```
flash_attn==2.5.8
torch==2.3.1
accelerate==0.31.0
transformers==4.41.2
```

Phi-3 Mini-128K-Instruct is also available in [Azure AI Studio](https://aka.ms/try-phi3)

### Tokenizer

Phi-3 Mini-128K-Instruct supports a vocabulary size of up to `32064` tokens. The [tokenizer files](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/added_tokens.json) already provide placeholder tokens that can be used for downstream fine-tuning, but they can also be extended up to the model's vocabulary size.


### Chat Format

Given the nature of the training data, the Phi-3 Mini-128K-Instruct model is best suited for prompts using the chat format as follows.
You can provide the prompt as a question with a generic template as follow:
```markdown
<|system|>
You are a helpful assistant.<|end|>
<|user|>
Question?<|end|>
<|assistant|>
```

For example:
```markdown
<|system|>
You are a helpful assistant.<|end|>
<|user|>
How to explain Internet for a medieval knight?<|end|>
<|assistant|>
```
where the model generates the text after `<|assistant|>` . In case of few-shots prompt, the prompt can be formatted as the following:

```markdown
<|system|>
You are a helpful travel assistant.<|end|>
<|user|>
I am going to Paris, what should I see?<|end|>
<|assistant|>
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."<|end|>
<|user|>
What is so great about #1?<|end|>
<|assistant|>
```

### Sample inference code

This code snippets show how to get quickly started with running the model on a GPU:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
```

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Languages other than English will experience worse performance. English language varieties with less representation in the training data might experience worse performance than standard American English.

### Evaluation factors:
Benchmarks listed in the Benchmarks section.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
Accuracy on benchmarks listed in the Benchmarks section.

### Decision thresholds:
Not available.

### Variation approaches:
Not available.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
Standard open-source benchmarks measuring the model's reasoning ability (both common sense reasoning and logical reasoning), public and internal benchmark datasets for release notes, RULER, RepoQA, GovReport, QMSum, Qasper, SQuALITY, SummScreenFD, 100 public benchmark datasets.

### Motivation:
Measuring the model's reasoning ability, highlights improvements, long context understanding, code understanding, different categories.

### Preprocessing:
Few-shot prompts, temperature 0.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
Our training data includes a wide variety of sources, totaling 4.9 trillion tokens, and is a combination of
1) Publicly available documents filtered rigorously for quality, selected high-quality educational data, and code;
2) Newly created synthetic, “textbook-like” data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.);
3) High quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness.

### Motivation:
We are focusing on the quality of data that could potentially improve the reasoning ability for the model, and we filter the publicly available documents to contain the correct level of knowledge. The model used additional post-training data leading to substantial gains on long-context understanding, instruction following, and structure output. We also improve multi-turn conversation quality, explicitly support <|system|> tag, and significantly improve reasoning capability.

### Preprocessing:
We filter the publicly available documents to contain the correct level of knowledge.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
Benchmark tables in the Benchmarks section.

### Intersectional results:
Not available.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
NVIDIA A100, NVIDIA A6000, NVIDIA H100. For NVIDIA V100 or earlier generation GPUs: call AutoModelForCausalLM.from_pretrained() with attn_implementation="eager"

### Deploying Requirements:
Optimized inference on GPU, CPU, and Mobile: use the **ONNX** models.

### Training or Fine-tuning Requirements:
512 H100-80G.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

Like other language models, the Phi series models can potentially behave in ways that are unfair, unreliable, or offensive. Some of the limiting behaviors to be aware of include:

+ Quality of Service: the Phi models are trained primarily on English text. Languages other than English will experience worse performance. English language varieties with less representation in the training data might experience worse performance than standard American English.
+ Representation of Harms & Perpetuation of Stereotypes: These models can over- or under-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post-training, these limitations may still be present due to differing levels of representation of different groups or prevalence of examples of negative stereotypes in training data that reflect real-world patterns and societal biases.
+ Inappropriate or Offensive Content: these models may produce other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the use case.
+ Information Reliability: Language models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated.
+ Limited Scope for Code: Majority of Phi-3 training data is based in Python and use common packages such as "typing, math, random, collections, datetime, itertools". If the model generates Python scripts that utilize other packages or scripts in other languages, we strongly recommend users manually verify all API uses.

Developers should apply responsible AI best practices and are responsible for ensuring that a specific use case complies with relevant laws and regulations (e.g. privacy, trade, etc.). Important areas for consideration include:

+ Allocation: Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (ex: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.
+ High-Risk Scenarios: Developers should assess suitability of using models in high-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (ex: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context.
+ Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).
+ Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case.
+ Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
Limitations listed in "Responsible AI Considerations" section. Overall, the model with only 3.8B-param achieves a similar level of language understanding and reasoning ability as much larger models. However, it is still fundamentally limited by its size for certain tasks. The model simply does not have the capacity to store too much world knowledge, which can be seen for example with low performance on TriviaQA.

### Recommendations:
Developers should apply responsible AI best practices. Developers should follow transparency best practices and inform end-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG). Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case. Developers should ensure that their applications do not violate applicable laws and regulations. We believe such weakness can be resolved by augmenting Phi-3-Mini with a search engine.

---

## Additional Information
🎉 **Phi-3.5**: [[mini-instruct]](https://huggingface.co/microsoft/Phi-3.5-mini-instruct); [[MoE-instruct]](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct) ; [[vision-instruct]](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)

Resources and Technical Documentation:

|         | Short Context | Long Context |
| :-      | :-            | :-           |
| Mini    | 4K [[HF]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx) ; [[GGUF]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx)|
| Small   | 8K [[HF]](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-small-8k-instruct-onnx-cuda) | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-small-128k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-small-128k-instruct-onnx-cuda)|
| Medium  | 4K [[HF]](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-cuda) | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct-onnx-cuda)|
| Vision  |  | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cuda)|

Release Notes

These tables below highlights improvements on instruction following, structure output, reasoning, and long-context understanding of the new release on our public and internal benchmark datasets.

| Benchmarks | Original | June 2024 Update |
| :-         | :-       | :-               |
| Instruction Extra Hard | 5.7 | 5.9 |
| Instruction Hard | 5.0 | 5.2 |
| JSON Structure Output | 1.9 | 60.1 |
| XML Structure Output | 47.8 | 52.9 |
| GPQA	| 25.9	| 29.7 |
| MMLU	| 68.1	| 69.7 |
| **Average**	| **25.7**	| **37.3** |

RULER: a retrieval-based benchmark for long context understanding

| Model             | 4K   | 8K   | 16K  | 32K  | 64K  | 128K | Average |
| :-------------------| :------| :------| :------| :------| :------| :------| :---------|
| Original          | 86.7 | 78.1 | 75.6 | 70.3 | 58.9 | 43.3 | **68.8**    |
| June 2024 Update  | 92.4 | 91.1 | 90.8 | 87.9 | 79.8 | 65.6 | **84.6**    |

RepoQA: a benchmark for long context code understanding

| Model             | Python | C++ | Rust | Java | TypeScript | Average |
| :-------------------| :--------| :-----| :------| :------| :------------| :---------|
| Original          | 27     | 29  | 40   | 33   | 33         | **32.4**    |
| June 2024 Update  | 85     | 63  | 72   | 93   | 72         | **77**      |

Notes: if users would like to check out the previous version, use the git commit id **bb5bf1e4001277a606e11debca0ef80323e5f824**. For the model conversion, e.g. GGUF and other formats, we invite the community to experiment with various approaches and share your valuable feedback. Let's innovate together!

Software

* [PyTorch](https://github.com/pytorch/pytorch)
* [Transformers](https://github.com/huggingface/transformers)
* [Flash-Attention](https://github.com/HazyResearch/flash-attention)

Hardware
Note that by default, the Phi-3 Mini-128K-Instruct model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:
* NVIDIA A100
* NVIDIA A6000
* NVIDIA H100

If you want to run the model on:
* NVIDIA V100 or earlier generation GPUs: call AutoModelForCausalLM.from_pretrained() with attn_implementation="eager"
* Optimized inference on GPU, CPU, and Mobile: use the **ONNX** models [128K](https://aka.ms/phi3-mini-128k-instruct-onnx)

Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.

Nothing contained in this Model Card should be interpreted as or deemed a restriction or modification to the license the model is released under.

We appreciate the enthusiastic adoption of the Phi-3 model family, and continue to welcome all feedback from the community.

Let's innovate together!

Notes: If you want to use flash attention, call _AutoModelForCausalLM.from_pretrained()_ with _attn_implementation="flash_attention_2"_

Important areas for consideration include:

Overall, the model with only 3.8B-param achieves a similar level of language understanding and reasoning ability as much larger models. However, it is still fundamentally limited by its size for certain tasks.

We believe such weakness can be resolved by augmenting Phi-3-Mini with a search engine.

Along with DML, ONNX Runtime provides cross platform support for Phi3 mini across a range of devices CPU, GPU, and mobile.

Here are some of the optimized configurations we have added:

1. ONNX models for int4 DML: Quantized to int4 via AWQ
2. ONNX model for fp16 CUDA
3. ONNX model for int4 CUDA: Quantized to int4 via RTN
4. ONNX model for int4 CPU and Mobile: Quantized to int4 via RTN

Note that by default, the Phi-3 Mini-128K-Instruct model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:

* NVIDIA A100
* NVIDIA A6000
* NVIDIA H100

If you want to run the model on:

* NVIDIA V100 or earlier generation GPUs: call AutoModelForCausalLM.from_pretrained() with attn_implementation="eager"
* Optimized inference on GPU, CPU, and Mobile: use the **ONNX** models [128K](https://aka.ms/phi3-mini-128k-instruct-onnx)
```