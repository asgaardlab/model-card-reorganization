## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
Developed by: [Cerebras Systems](https://www.cerebras.net/)

### Model date:
Not available.

### Model version:
Not available.

### Model type:
* Model type: Transformer-based Language Model
* Architecture: GPT-3 style architecture
* Vocabulary Size: 50257
* Sequence Length: 2048

### Training details:
* Optimizer: AdamW, (β1, β2) = (0.9, 0.95), adam_eps = 1e−8 (1e−9 for larger models)
* Positional Encoding: Learned
* Learning rate warmed up for 375M tokens (1500 steps for 111M and 256M models) and 10x cosine decayed.
* No dropout was used and weight decay was set to 0.1.
* All models are trained with MSL of 2048.
* All models were trained to Chinchilla point: 20 tokens per model parameter.

| Model Params | Sequence Length | Batch Size | Number of Steps | Tokens | Tokens per Parameter | Flops
------------ | -------------- | ---------- | --------------- | ------ | -------------------- | -----
111M         | 2048           | 120        | 9037            | 2.22E+09 | 20                  | 2.6E+18
256M         | 2048           | 264        | 9468            | 5.12E+09 | 20                  | 1.3E+19
590M         | 2048           | 264        | 21836           | 1.18E+10 | 20                  | 6.1E+19
1.3B         | 2048           | 528        | 24334           | 2.63E+10 | 20                  | 2.8E+20
2.7B         | 2048           | 528        | 49041           | 5.30E+10 | 20                  | 1.1E+21
6.7B         | 2048           | 1040       | 62522           | 1.33E+11 | 20                  | 6.3E+21
13B          | 2048           | 720        | 174335          | 2.57E+11 | 20                  | 2.3E+22

### Paper or other resource for more information:
* Check out our [Blog Post](https://www.cerebras.net/cerebras-gpt) and [arXiv paper](https://arxiv.org/abs/2304.03208)!
* Learn more: Dense Scaling Laws Paper for training procedure, config files, and details on how to use.
* [Andromeda](https://www.cerebras.net/andromeda/) AI supercomputer used to train the models
* Cerebras' [weight streaming technology](https://www.cerebras.net/blog/linear-scaling-made-possible-with-weight-streaming) simplifies the training of LLMs
* [Cerebras Model Studio](https://www.cerebras.net/product-cloud/)
* [Cerebras Model Zoo](https://github.com/Cerebras/modelzoo) for available Cerebras CS-2 compatible checkpoints
* [Eleuther lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
* [Pile paper](https://arxiv.org/abs/2101.00027) for a more detailed breakdown of data sources and methodology
* [EleutherAI](https://www.eleuther.ai)

### Citation details:
Not available.

### License:
The models are released with a fully permissive Apache license (Apache 2.0) for the community to use freely.

### Contact:
To ask questions about Cerebras-GPT models, join the [Cerebras Discord](https://discord.gg/q6bZcMWJVu).

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
The primary intended use is to further research into large language models. These models can be used as a foundation model for NLP, applications, ethics, and alignment research. Our primary intended users are researchers who are working to improve LLMs and practitioners seeking reference implementations, training setups, hyperparameters, or pre-trained models.
You may fine-tune and adapt Cerebras-GPT models for deployment via either Cerebras [Model Studio](https://www.cerebras.net/product-cloud/) or third-party libraries.

### Primary intended users:
Our primary intended users are researchers who are working to improve LLMs and practitioners seeking reference implementations, training setups, hyperparameters, or pre-trained models.

### Out-of-scope uses:
Cerebras-GPT models are trained on the Pile, with English language only, and are not suitable for machine translation tasks.

Cerebras-GPT models have not been tuned for human-facing dialog applications like chatbots and will not respond to prompts in a similar way to models that have received instruction tuning or reinforcement learning from human feedback (RLHF) like Flan-T5 or ChatGPT. Cerebras-GPT models can be tuned using those methods.

---

## How to Use
This section outlines how to use the model.

This model can be easily loaded using the AutoModelForCausalLM functionality:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-13B")
model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-13B")

text = "Generative AI is "
```

And can be used with Hugging Face Pipelines

```python
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
generated_text = pipe(text, max_length=50, do_sample=False, no_repeat_ngram_size=2)[0]
print(generated_text['generated_text'])
```

or with `model.generate()`

```python
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, num_beams=5,
                        max_new_tokens=50, early_stopping=True,
                        no_repeat_ngram_size=2)
text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text_output[0])
```

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Not available.

### Evaluation factors:
Model size (parameters)

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
* PILE test xent
* Hella-Swag
* PIQA
* Wino-Grande
* Lambada
* ARC-e
* ARC-c
* OpenBookQA
* Downstream Average

### Decision thresholds:
Not available.

### Variation approaches:
Not available.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
* Pile validation and test splits
* [Eleuther lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

### Motivation:
We performed upstream (pre-training) evaluations of text prediction cross-entropy using the Pile validation and test splits. We performed downstream evaluations of text generation accuracy on standardized tasks using the [Eleuther lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

### Preprocessing:
Not available.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
* Data set: The Pile
* [the Pile](https://pile.eleuther.ai) dataset from [EleutherAI](https://www.eleuther.ai).
* Our tokenized version of the Pile has 371B tokens.

### Motivation:
All models in the Cerebras-GPT family have been trained in accordance with [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) (20 tokens per model parameter) which is compute-optimal.

### Preprocessing:
* The Pile was cleaned using the ftfy library to normalize the text, then filtered using scripts provided by Eleuther.
* We tokenized the data using byte-pair encoding using the GPT-2 vocabulary.
* We include more details about the training dataset preprocessing in Appendix A.1 of our paper.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:

#### 0-shot Evaluation
| Model   | Params | Training FLOPs | PILE test xent | Hella-Swag | PIQA  | Wino-Grande | Lambada | ARC-e | ARC-c | OpenBookQA | Downstream Average |
| ------- | ----- | -------------- | -------------- | ---------- | ----- | ----------- | ------- | ----- | ----- | ---------- | ------------------ |
| Cerebras-GPT | 111M  | 2.6E+18        | 2.566          | 0.268      | 0.594 | 0.488       | 0.194   | 0.380 | 0.166 | 0.118      | 0.315              |
| Cerebras-GPT | 256M  | 1.3E+19        | 2.299          | 0.274      | 0.613 | 0.511       | 0.293   | 0.410 | 0.170 | 0.158      | 0.347              |
| Cerebras-GPT | 590M  | 6.1E+19        | 2.184          | 0.291      | 0.627 | 0.498       | 0.366   | 0.464 | 0.190 | 0.158      | 0.370              |
| Cerebras-GPT | 1.3B  | 2.8E+20        | 1.996          | 0.325      | 0.664 | 0.521       | 0.462   | 0.508 | 0.224 | 0.166      | 0.410              |
| Cerebras-GPT | 2.7B  | 1.1E+21        | 1.834          | 0.386      | 0.701 | 0.559       | 0.567   | 0.571 | 0.246 | 0.206      | 0.462              |
| Cerebras-GPT | 6.7B  | 6.3E+21        | 1.704          | 0.447      | 0.739 | 0.602       | 0.636   | 0.643 | 0.282 | 0.238      | 0.512              |
| Cerebras-GPT | 13B   | 2.3E+22        | 1.575          | 0.513      | 0.766 | 0.646       | 0.696   | 0.714 | 0.367 | 0.286      | 0.570              |

#### 5-shot Evaluation
| Model    | Params | Hella-Swag | PIQA  | Wino-Grande | Lambada | ARC-e | ARC-c | OpenBookQA |
| -------- | ----- | ----------| ----- | ----------- | -------| ----- | ----- | ---------- |
| Cerebras-GPT | 111M  | 0.267     | 0.588 | 0.475       | 0.158  | 0.356 | 0.166 | 0.136      |
| Cerebras-GPT | 256M  | 0.278     | 0.606 | 0.522       | 0.225  | 0.422 | 0.183 | 0.164      |
| Cerebras-GPT | 590M  | 0.291     | 0.634 | 0.479       | 0.281  | 0.475 | 0.206 | 0.152      |
| Cerebras-GPT | 1.3B  | 0.326     | 0.668 | 0.536       | 0.395  | 0.529 | 0.241 | 0.174      |
| Cerebras-GPT | 2.7B  | 0.382     | 0.697 | 0.543       | 0.487  | 0.590 | 0.267 | 0.224      |
| Cerebras-GPT | 6.7B  | 0.444     | 0.736 | 0.590       | 0.591  | 0.667 | 0.314 | 0.270      |
| Cerebras-GPT | 13B   | 0.514     | 0.768 | 0.674       | 0.655  | 0.743 | 0.398 | 0.318      |

### Intersectional results:
Not available.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
Not available.

### Deploying Requirements:
Not available.

### Training or Fine-tuning Requirements:
* These models were trained on the [Andromeda](https://www.cerebras.net/andromeda/) AI supercomputer comprised of 16 CS-2 wafer scale systems.
* Cerebras systems for pre-training and fine tuning are available in the cloud via the [Cerebras Model Studio](https://www.cerebras.net/product-cloud/).

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

* **Data**: The Pile dataset has been thoroughly analyzed from various ethical standpoints such as toxicity analysis, gender bias, pejorative content, racially sensitive content etc. Please refer to Pile dataset references.
* **Human life**: The outputs from this model may or may not align with human values. The risk needs to be thoroughly investigated before deploying this model in a production environment where it can directly impact human life.
* **Risks and harms**: There can be distributional bias in the Pile dataset that can manifest in various forms in the downstream model deployment. There are other risks associated with large language models such as amplifying stereotypes, memorizing training data, or revealing private or secure information.
* **Mitigations**: Only mitigations in standard Pile dataset pre-processing were employed when pre-training Cerebras-GPT.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
* Due to financial and compute budgets, Cerebras-GPT models were only trained and evaluated following the approaches described in the paper.
* Our models are trained on the standard Pile without deduplication, which may present an opportunity for further improvement with the deduplicated data set.
* Cerebras-GPT models were only trained and evaluated following the approaches described in the paper.

### Recommendations:
Further safety-related testing and mitigations should be applied beore using the Cerebras-GPT model family in production downstream applications.

---

## Additional Information
# Cerebras-GPT 13B
The Cerebras-GPT family is released to facilitate research into LLM scaling laws using open architectures and data sets and demonstrate the simplicity of and scalability of training LLMs on the Cerebras software and hardware stack. All Cerebras-GPT models are available on Hugging Face.

The family includes 111M, 256M, 590M, 1.3B, 2.7B, 6.7B, and 13B models.

| Model         | Parameters | Layers | d_model | Heads | d_head | d_ffn  | LR       | BS (seq) | BS (tokens)     |
|---------------|------------|--------|---------|-------|--------|--------|----------|----------|----------------|
| Cerebras-GPT  | 111M       | 10     | 768     | 12    | 64     | 3072   | 6.0E-04 | 120      | 246K           |
| Cerebras-GPT  | 256M       | 14     | 1088    | 17    | 64     | 4352   | 6.0E-04 | 264      | 541K           |
| Cerebras-GPT  | 590M       | 18     | 1536    | 12    | 128    | 6144   | 2.0E-04 | 264      | 541K           |
| Cerebras-GPT  | 1.3B       | 24     | 2048    | 16    | 128    | 8192   | 2.0E-04 | 528      | 1.08M          |
| Cerebras-GPT  | 2.7B       | 32     | 2560    | 32    | 80     | 10240  | 2.0E-04 | 528      | 1.08M          |
| Cerebras-GPT  | 6.7B       | 32     | 4096    | 32    | 128    | 16384  | 1.2E-04 | 1040     | 2.13M          |
| Cerebras-GPT  | 13B        | 40     | 5120    | 40    | 128    | 20480  | 1.2E-04 | 720 &rarr; 1080 | 1.47M &rarr; 2.21M    |

This is the standard parameterization version of Cerebras-GPT with **13B** parameters

Related models: [Cerebras-GPT Models](https://huggingface.co/models?sort=downloads&search=cerebras-gpt)

These models were trained on the [Andromeda](https://www.cerebras.net/andromeda/) AI supercomputer comprised of 16 CS-2 wafer scale systems. Cerebras' [weight streaming technology](https://www.cerebras.net/blog/linear-scaling-made-possible-with-weight-streaming) simplifies the training of LLMs by disaggregating compute from model storage. This allowed for efficient scaling of training across nodes using simple data parallelism.

## Training data

Recent works find significant duplicate data present in the Pile. Eleuther’s Pythia applies a deduplication process to reduce replicated data, decreasing the Pile dataset size. Pythia was trained on both the standard dataset and deduplicated dataset to characterize the impact.

## Training procedure

We use the GPT-3 style model architecture. All of our layers use full attention as opposed to the GPT-3 style sparse banded attention. The model shapes were selected to either follow aspect ratio 80 or are the same shape as GPT-3 models.

Number of steps was chosen based on optimal batch size (varied by model) and fixed sequence length (2048)

## Evaluations

We trained models from smallest to largest and fit a power law as we went along. The power law was helpful for extrapolating the validation loss of the next largest model we trained and provided confidence about whether the training run was going well.

Results are compared against many publicly available large language models in Section 3 of the paper.

## Acknowledgements

We are thankful to all Cerebras engineers, past and present, that made this work possible.
