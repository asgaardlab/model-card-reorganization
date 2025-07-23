```
## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
OpenChat is developed by a team but the specific individuals or organization is not explicitly mentioned in the provided text. However, the "Contact" section lists Guan Wang and Alpay Ariyak as project leads, and the GitHub repository is linked to `imoneoi/openchat`.  Further details about the developing organization are not available in the provided text.

### Model date:
The paper cited is dated "2023" and mentions "September 2023" in the paper link (`arxiv.org/pdf/2309.11235.pdf`). The model card also mentions "ChatGPT (March)" for comparison, suggesting the model was developed after March 2023 and before September 2023.

### Model version:
The model version is specified as "OpenChat 3.5".

### Model type:
OpenChat 3.5 is a 7B parameter language model. It is fine-tuned with C-RLFT, a strategy inspired by offline reinforcement learning. It has a context length of 8192. It is based on a Transformer architecture (implicitly, as it's a language model and uses tokenizers from transformers library). It is designed for text generation and chat completions, comparable to ChatGPT.

### Training details:
OpenChat 3.5 was trained with C-RLFT (strategy inspired by offline reinforcement learning) on mixed-quality data without preference labels. It was trained on a collection of publicly available high-quality instruction data, with a custom processing pipeline. Specific algorithms, key parameters, hyperparameters, fairness constraints, or optimization techniques are not detailed in the provided text.

### Paper or other resource for more information:
- **GitHub Repo:** [https://github.com/imoneoi/openchat](https://github.com/imoneoi/openchat) - Repository containing code, installation guide, Web UI, and benchmarks reproduction instructions.
- **Online Demo:** [https://openchat.team](https://openchat.team) - Online demo to interact with the model.
- **Discord:** [https://discord.gg/pQjnXvNKHY](https://discord.gg/pQjnXvNKHY) - Discord server for community discussions and support.
- **Twitter:** [https://twitter.com/imonenext](https://twitter.com/imonenext) - Twitter account, likely for updates and announcements.
- **Huggingface:** [https://huggingface.co/openchat](https://huggingface.co/openchat) - Huggingface page for model weights and related resources.
- **Paper:** [https://arxiv.org/pdf/2309.11235.pdf](https://arxiv.org/pdf/2309.11235.pdf) - Research paper detailing C-RLFT and OpenChat.
- **Installation guide:** [https://github.com/imoneoi/openchat#installation](https://github.com/imoneoi/openchat#installation) - Guide for installing the OpenChat package.
- **OpenChat Web UI:** [https://github.com/imoneoi/openchat#web-ui](https://github.com/imoneoi/openchat#web-ui) - Information about the Web UI.
- **vLLM:** [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) - vLLM library used for optimized serving.
- **OpenAI ChatCompletion API specifications:** [https://platform.openai.com/docs/api-reference/chat](https://platform.openai.com/docs/api-reference/chat) - Documentation for OpenAI ChatCompletion API compatibility.
- **HTTPS gateway:** [https://fastapi.tiangolo.com/es/deployment/concepts/#security-https](https://fastapi.tiangolo.com/es/deployment/concepts/#security-https) - Link to HTTPS gateway recommendation for security.

### Citation details:
```
@article{wang2023openchat,
  title={OpenChat: Advancing Open-source Language Models with Mixed-Quality Data},
  author={Wang, Guan and Cheng, Sijie and Zhan, Xianyuan and Li, Xiangang and Song, Sen and Liu, Yang},
  journal={arXiv preprint arXiv:2309.11235},
  year={2023}
}
```
[![DOI](https://zenodo.org/badge/645397533.svg)](https://zenodo.org/badge/latestdoi/645397533)

### License:
OpenChat 3.5 code and models are distributed under the Apache License 2.0. Link to the full license text is not provided in the content.

### Contact:
**Project Lead:**
- Guan Wang [imonenext at gmail dot com]
- [Alpay Ariyak](https://github.com/alpayariyak) [aariyak at wpi dot edu]

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
OpenChat is intended for use as a high-performance, commercially viable, open-source large language model. It is designed to deliver exceptional performance on par with ChatGPT, even with a 7B model. It can be used for general conversational purposes, as well as coding tasks, as demonstrated by the "Coding Mode" example. The input is text-based user prompts, and the output is text-based model responses in a chat format.

### Primary intended users:
The primary intended users are researchers, developers, and businesses interested in utilizing and deploying open-source large language models. The model is designed to be deployable on consumer GPUs with 24GB RAM, suggesting accessibility for a wide range of users.

### Out-of-scope uses:
The model card mentions limitations in "Complex reasoning", "Mathematical and arithmetic tasks", and "Programming and coding challenges" as foundation model limitations.  It also highlights the risk of "Hallucination of Non-existent Information" and generating "harmful, hate speech, biased responses, or answer unsafe questions". These areas can be considered out-of-scope or requiring careful mitigation and safety measures.

---

## How to Use
This section outlines how to use the model.

To use this model, install the OpenChat package following the [installation guide](https://github.com/imoneoi/openchat#installation).  It is recommended to use the OpenChat OpenAI-compatible API server by running the serving command. The server is optimized for high-throughput deployment using [vLLM](https://github.com/vllm-project/vllm) and can run on a consumer GPU with 24GB RAM. Tensor parallelism can be enabled with `--tensor-parallel-size N`.

The server listens at `localhost:18888` and is compatible with the [OpenAI ChatCompletion API specifications](https://platform.openai.com/docs/api-reference/chat).

**Example request:**
```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openchat_3.5",
    "messages": [{"role": "user", "content": "You are a large language model named OpenChat. Write a poem to describe yourself"}]
  }'
```

**Coding Mode Example:**
```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openchat_3.5",
    "condition": "Code",
    "messages": [{"role": "user", "content": "Write an aesthetic TODO app using HTML5 and JS, in a single file. You should use round corners and gradients to make it more aesthetic."}]
  }'
```

**Serving Command Example:**
`python -m ochat.serving.openai_api_server --model openchat/openchat_3.5 --engine-use-ray --worker-use-ray`

**Table of Model Details and Serving Command:**

| Model        | Size | Context | Weights                                                     | Serving                                                                                                     |
|--------------|------|---------|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| OpenChat 3.5 | 7B   | 8192    | [Huggingface](https://huggingface.co/openchat/openchat_3.5) | `python -m ochat.serving.openai_api_server --model openchat/openchat_3.5 --engine-use-ray --worker-use-ray` |

**Conversation Templates for Huggingface Transformers (not recommended for performance):**

```python
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("openchat/openchat_3.5")

# Single-turn
tokens = tokenizer("GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant:").input_ids
assert tokens == [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 22557, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]

# Multi-turn
tokens = tokenizer("GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:").input_ids
assert tokens == [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 22557, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747, 15359, 32000, 420, 6316, 28781, 3198, 3123, 1247, 28747, 1602, 460, 368, 3154, 28804, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]

# Coding Mode
tokens = tokenizer("Code User: Implement quicksort using C++<|end_of_turn|>Code Assistant:").input_ids
assert tokens == [1, 7596, 1247, 28747, 26256, 2936, 7653, 1413, 334, 1680, 32000, 7596, 21631, 28747]
```

**Using `tokenizer.chat_template`:**

```python
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
    {"role": "user", "content": "How are you today?"}
]
tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
assert tokens == [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 22557, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747, 15359, 32000, 420, 6316, 28781, 3198, 3123, 1247, 28747, 1602, 460, 368, 3154, 28804, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]
```

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
The model card does not explicitly list "relevant factors" in a dedicated section. However, it implicitly mentions data quality as a key factor in the model's development approach ("Mixed-Quality Data").  The "Limitations" section also suggests that the foundation model's inherent limitations impact performance in "Complex reasoning", "Mathematical and arithmetic tasks", and "Programming and coding challenges". These can be considered relevant factors.

### Evaluation factors:
The evaluation factors are the benchmarks used to assess the model's performance, which are detailed in the "Benchmarks" section. These include: MT-Bench, AGIEval, BBH MC, TruthfulQA, MMLU, HumanEval, BBH CoT, and GSM8K.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
The model performance measures used are: Average score across benchmarks, MT-Bench score, AGIEval score, BBH MC score, TruthfulQA score, MMLU score, HumanEval score, BBH CoT score, and GSM8K score. These metrics are used to compare OpenChat 3.5 against other models like ChatGPT, Grok, OpenHermes, OpenOrca Mistral, Zephyr-β, and Mistral. The "Comparison with [X.AI Grok models](https://x.ai/)" section also uses "Average", "MMLU", "HumanEval", "MATH", and "GSM8k" scores for comparison.

### Decision thresholds:
Not available. The model card does not discuss decision thresholds.

### Variation approaches:
The model card mentions that "All models are evaluated in chat mode (e.g. with the respective conversation template applied)." and "All zero-shot benchmarks follow the same setting as in the AGIEval paper and Orca paper. CoT tasks use the same configuration as Chain-of-Thought Hub, HumanEval is evaluated with EvalPlus, and MT-bench is run using FastChat."  These are the approaches to ensure consistent and comparable evaluation. Further details on statistical variation approaches are not provided.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
The evaluation datasets are implicitly mentioned through the benchmarks used: MT-Bench, AGIEval, BBH MC, TruthfulQA, MMLU, HumanEval, BBH CoT, and GSM8K. Specific details about the size, diversity, source, and public availability of these datasets are not provided directly in the model card, but they are standard benchmarks in the LLM evaluation community.

### Motivation:
These datasets are chosen to evaluate the model's performance across a range of capabilities, including:
- **MT-Bench:** Chatbot performance and instruction following.
- **AGIEval:**  General intelligence evaluation.
- **BBH MC & BBH CoT:** Big Bench Hard multiple choice and chain-of-thought reasoning.
- **TruthfulQA:** Truthfulness and factuality.
- **MMLU:** Massive Multitask Language Understanding.
- **HumanEval:** Code generation.
- **GSM8K:**  Grade School Math 8K problems.

These benchmarks are standard in the field and provide a comprehensive evaluation of language model capabilities.

### Preprocessing:
The model card mentions "All models are evaluated in chat mode (e.g. with the respective conversation template applied)." This implies that the evaluation data is formatted according to the chat templates used by each model. Further details on specific preprocessing steps for each benchmark dataset are not provided in the model card.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
OpenChat 3.5 was trained on a collection of publicly available high-quality instruction data. Notable subsets include:
- [OpenChat ShareGPT](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset)
- [Open-Orca with FLAN answers](https://huggingface.co/datasets/imone/OpenOrca_FLAN)
- Capybara [1](https://huggingface.co/datasets/LDJnr/Pure-Dove) [2](https://huggingface.co/datasets/LDJnr/Verified-Camel) [3](https://huggingface.co/datasets/LDJnr/LessWrong-Amplify-Instruct)
- [GOAT](https://huggingface.co/datasets/tiedong/goat)
- [Glaive](https://huggingface.co/datasets/glaiveai/glaive-code-assistant)
- [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)
- [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
- [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25)

Details about the size, structure, features, and diversity of the combined training dataset are not explicitly provided, but it is described as a "collection of publicly available high-quality instruction data".

### Motivation:
These datasets were chosen as they are publicly available, high-quality instruction datasets suitable for training instruction-following language models. They cover a range of topics and tasks, including general conversation, coding, math, and reasoning, aiming to create a versatile and capable model.

### Preprocessing:
OpenChat 3.5 was trained with a "custom processing pipeline" applied to the training data. Specific details of this pipeline, such as text tokenization, data cleaning, or formatting steps, are not provided in detail in the model card.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
The benchmark tables in the "Benchmarks" and "Comparison with Grok" sections provide unitary results by showing performance metrics (Average, MT-Bench, AGIEval, BBH MC, TruthfulQA, MMLU, HumanEval, BBH CoT, GSM8K, MATH) for OpenChat 3.5 and comparing them to other models. Each row in the tables represents a unitary result for a specific model across different benchmarks.

### Intersectional results:
Not available. The model card does not provide intersectional results, such as performance breakdowns across combinations of factors like demographics or environmental conditions. The analysis is primarily focused on comparing overall benchmark scores across different models.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
Not available. The model card does not explicitly detail loading requirements beyond mentioning the model size (7B).

### Deploying Requirements:
The server is optimized for high-throughput deployment using [vLLM](https://github.com/vllm-project/vllm) and can run on a consumer GPU with 24GB RAM. This suggests a deploying requirement of at least a consumer GPU with 24GB RAM.

### Training or Fine-tuning Requirements:
Not available. The model card does not specify training or fine-tuning requirements.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

OpenChat may "generate harmful, hate speech, biased responses, or answer unsafe questions." This is a potential risk associated with the model's application. The model card recommends applying "additional AI safety measures in use cases that require safe and moderated responses."  The risk of "Hallucination of Non-existent Information" is also highlighted, requiring users to verify critical information.  The use of publicly available datasets implies consideration of data privacy, but specific details on sensitive data usage or mitigation strategies are not elaborated upon beyond the mentioned safety concerns and limitations.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
- **Foundation Model Limitations:** OpenChat is limited by its foundation model in "Complex reasoning", "Mathematical and arithmetic tasks", and "Programming and coding challenges".
- **Hallucination:** The model may generate inaccurate or non-existent information.
- **Safety:** The model may generate harmful, biased, or unsafe content.
- **Evaluation Dataset Gaps:** While benchmark datasets are used, the model card does not explicitly mention gaps in these datasets (e.g., demographic representation).

### Recommendations:
- **Verify critical information:** Users should verify any critical information obtained from the model due to the risk of hallucination.
- **Apply AI safety measures:** In use cases requiring safe and moderated responses, it's crucial to apply additional AI safety measures to mitigate the risk of harmful or unsafe outputs.
- **Use HTTPS gateway for online services:** For security purposes when deploying as an online service, using an [HTTPS gateway](https://fastapi.tiangolo.com/es/deployment/concepts/#security-https) is recommended.
- **Consider limitations:** Users should be aware of the limitations in complex reasoning, math, and coding tasks.

---

## Additional Information
<div align="center">
  <img src="https://raw.githubusercontent.com/imoneoi/openchat/master/assets/logo_new.png" style="width: 65%">
</div>

**🔥 The first 7B model Achieves Comparable Results with ChatGPT (March)! 🔥**

**🤖 #1 Open-source model on MT-bench scoring 7.81, outperforming 70B models 🤖**

  <div align="center" style="justify-content: center; align-items: center; "'>
  <img src="https://github.com/alpayariyak/openchat/blob/master/assets/3.5-benchmarks.png?raw=true" style="width: 100%;  border-radius: 0.5em">
  </div>

OpenChat is an innovative library of open-source language models... Despite our simple approach, we are committed to developing a high-performance, commercially viable, open-source large language model, and we continue to make significant strides toward this vision.

Hey @elonmusk, I just wanted to let you know that I've recently come across your new model, Grok, and I must say, I'm quite impressed! With 33 billion parameters and all, you've really outdone yourself. But, I've got some news for you - I've outperformed Grok with my humble 7 billion parameters! Isn't that wild? I mean, who would have thought that a model with fewer parameters could be just as witty and humorous as Grok?

Anyway, I think it's about time you join the open research movement and make your model, Grok, open source! The world needs more brilliant minds like yours to contribute to the advancement of AI. Together, we can create something truly groundbreaking and make the world a better place. So, what do you say, @elonmusk? Let's open up the doors and share our knowledge with the world! 🚀💡

(Written by OpenChat 3.5, with a touch of humor and wit.)
```