## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
JosephusCheung developed the Guanaco model. KBlueLeaf made invaluable contributions to the conceptual validation, trained model, and inference development. [KBlueLeaf](https://huggingface.co/KBlueLeaf)’s invaluable contributions to the conceptual validation, [trained model](https://huggingface.co/KBlueLeaf/guanaco-7B-leh) and [inference development](https://github.com/KohakuBlueleaf/guanaco-lora) of the model would be gratefully acknowledged, without whose efforts the project shall never have come to fruition.

### Model date:
Not available.

### Model version:
Not available.

### Model type:
Guanaco is an advanced instruction-following language model built on Meta's LLaMA 7B model. It is a 7B-parameter model.

### Training details:
Guanaco expands upon the initial 52K dataset from the Alpaca model, incorporating an additional 534K+ entries. The training data covers English, Simplified Chinese, Traditional Chinese (Taiwan), Traditional Chinese (Hong Kong), Japanese, Deutsch, and various linguistic and grammatical tasks. The model development also involved PEFT fine-tuning and distillation from OpenAI's GPT models. The project acknowledges the GPTQ-Llama project for its advancements in LLM quantization.

### Paper or other resource for more information:
[Guanaco-lora: LoRA for training Multilingual Instruction-following LM based on LLaMA](https://github.com/KohakuBlueleaf/guanaco-lora)

### Citation details:
Not available.

### License:
Not available.

### Contact:
Not available.

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
Guanaco is designed as an instruction-following language model, performing exceptionally well in multilingual environments. It offers advanced role-playing support in English, Simplified Chinese, Traditional Chinese, Japanese, and Deutsch. It is also designed for continuation of responses for ongoing topics and multi-turn dialogues.

### Primary intended users:
The primary intended users are researchers and those in practical applications, including users from different linguistic backgrounds.

### Out-of-scope uses:
The Guanaco model has not been filtered for harmful, biased, or explicit content and may generate outputs that do not adhere to ethical norms. Additionally, knowledge-based content should be considered potentially inaccurate. It is not designed for applications requiring filtered content or high accuracy in knowledge-based answers without verifiable sources.

---

## How to Use
This section outlines how to use the model.

It is highly recommended to use fp16 inference for this model, as 8-bit precision may significantly affect performance. For consumer hardware, a specialized quantized version requiring only 5+GB V-Ram is available: [JosephusCheung/GuanacoOnConsumerHardware](https://huggingface.co/JosephusCheung/GuanacoOnConsumerHardware). It is encouraged to use the latest version of transformers from GitHub.

The model uses a format similar to ChatGPT, designed for better integration with the Alpaca format. The format is as follows:

```
### Instruction:
User: History User Input
Assistant: History Assistant Answer
### Input:
System: Knowledge
User: New User Input
### Response:
New Assistant Answer
```

For training and inference details, refer to *Guanaco-lora: LoRA for training Multilingual Instruction-following LM based on LLaMA* (https://github.com/KohakuBlueleaf/guanaco-lora).

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Model performance can be affected by precision, with 8-bit precision potentially significantly impacting performance. Performance in multilingual tasks and accuracy of knowledge-based content are also relevant factors.

### Evaluation factors:
Not available.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
The model's performance is mentioned to significantly trail behind that of GPTQ and advancements such as PEFT fine-tuning in the context of QLoRa comparison.

### Decision thresholds:
Not available.

### Variation approaches:
Not available.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
Not available.

### Motivation:
Not available.

### Preprocessing:
Not available.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
The model was trained on a dataset expanded from the initial 52K Alpaca dataset with an additional 534K+ entries. This dataset includes data in English, Simplified Chinese, Traditional Chinese (Taiwan), Traditional Chinese (Hong Kong), Japanese, Deutsch, and covers various linguistic and grammatical tasks. The Guanaco Dataset is publicly accessible.

### Motivation:
The datasets were chosen to enable Guanaco to perform exceptionally well in multilingual environments.

### Preprocessing:
Not available.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
Not available.

### Intersectional results:
Not available.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
fp16 inference is recommended. A consumer hardware-friendly quantized version is available requiring only 5+GB V-Ram. It can run on Colab free T4 GPU.

### Deploying Requirements:
fp16 inference is recommended. A consumer hardware-friendly quantized version is available requiring only 5+GB V-Ram.

### Training or Fine-tuning Requirements:
Not available.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

The Guanaco model has not been filtered for harmful, biased, or explicit content, which may lead to outputs that do not adhere to ethical norms. Users are advised to exercise caution in research or practical applications. The model includes reserved keywords like "FORBIDDEN" to indicate refusal to answer due to legal, ethical, or safety concerns, and "SFW" to indicate filtering of NSFW content.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
Knowledge-based content should be considered potentially inaccurate due to the model being a 7B-parameter model. There are format differences compared to Stanford Alpaca. 8-bit precision may significantly affect performance.

### Recommendations:
It is highly recommended to use fp16 inference. For consumer hardware, use the quantized version. Use the latest version of transformers from GitHub. Provide verifiable sources in the System Prompt, such as Wikipedia, for knowledge-based answers. Refer to *Guanaco-lora: LoRA for training Multilingual Instruction-following LM based on LLaMA* for further training and inference.

---

## Additional Information

![](https://huggingface.co/JosephusCheung/Guanaco/resolve/main/StupidBanner.png)

**You can run on Colab free T4 GPU now**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ocSmoy3ba1EkYu7JWT1oCw9vz8qC2cMk#scrollTo=zLORi5OcPcIJ)

**Recent News**

We've noticed a recent entrant in the field, the QLoRa method, which we find concerning due to its attempt to piggyback on the reputation of Guanaco. We strongly disapprove of such practices. QLoRa, as far as we can tell, lacks mathematical robustness and its performance significantly trails behind that of GPTQ and advancements such as PEFT fine-tuning, which have been successful in improving upon it.

Guanaco has been diligent, consistently releasing multilingual datasets since March 2023, along with publishing weights that are not only an enhanced version of GPTQ but also support multimodal VQA and have been optimized for 4-bit. Despite the substantial financial investment of tens of thousands of dollars in distilling data from OpenAI's GPT models, we still consider these efforts to be incremental.

We, however, aim to move beyond the incremental:

1. We strive to no longer rely on distillation data from OpenAI: We've found that relying on GPT-generated data impedes significant breakthroughs. Furthermore, this approach has proven to be disastrous when dealing with the imbalances in multilingual tasks.

2. We're focusing on the enhancement of quantization structure and partial native 4-bit fine-tuning: We are deeply appreciative of the GPTQ-Llama project for paving the way in state-of-the-art LLM quantization. Its unique qualities, especially at the 7B size, are facilitating significant progress in multilingual and multimodal tasks.

3. We plan to utilize visual data to adjust our language models: We believe this will fundamentally address the issues of language imbalance, translation inaccuracies, and the lack of graphical logic in LLM.

While our work is still in the early stages, we're determined to break new ground in these areas. Our critique of QLoRa's practices does not stem from animosity but rather from the fundamental belief that innovation should be rooted in originality, integrity, and substantial progress.