## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
GPT-J 6B is a transformer model trained using Ben Wang's [Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax/). Ben Wang also developed the Mesh-Transformer-JAX codebase.

### Model date:
The model card mentions May 2021 as the month of publication for both GPT-J-6B and Mesh-Transformer-JAX.

### Model version:
"GPT-J" refers to the class of model, while "6B" represents the number of trainable parameters. This implicitly suggests a version based on the parameter size (6B). No explicit version number is given beyond "6B".

### Model type:
GPT-J 6B is a transformer model. It is an autoregressive language model, designed for text generation from a prompt.
The model has 6 billion parameters (6053381344).
The context length is 2048 tokens (\\(n_{ctx}\\)).
The architecture details are as follows:
* 28 layers (\\(n_{layers}\\))
* Model dimension of 4096 (\\(d_{model}\\))
* Feedforward dimension of 16384 (\\(d_{ff}\\))
* 16 heads (\\(n_{heads}\\))
* Head dimension of 256 (\\(d_{head}\\))
* Rotary Position Embedding (RoPE) with 64 dimensions.
* Vocabulary size of 50257/50400 (\\(n_{vocab}\\)), using the same tokenizer as GPT-2/3.

### Training details:
The model was trained as an autoregressive language model, using cross-entropy loss to maximize the likelihood of predicting the next token correctly. It was trained for 402 billion tokens over 383,500 steps on TPU v3-256 pod.

### Paper or other resource for more information:
* [Mesh Transformer JAX GitHub Repository](https://github.com/kingoflolz/mesh-transformer-jax/): Codebase used to train the model.
* [The Pile Paper](https://arxiv.org/abs/2101.00027):  Provides a detailed analysis of the biases in the Pile dataset, which GPT-J was trained on.
* [GPT-3 Model Sizes Blog Post](https://blog.eleuther.ai/gpt3-model-sizes/):  Provides more details on evaluation and comparison with other models.

### Citation details:
To cite this model:
```bibtex
@misc{gpt-j,
  author = {Wang, Ben and Komatsuzaki, Aran},
  title = {{GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model}},
  howpublished = {\url{https://github.com/kingoflolz/mesh-transformer-jax}},
  year = 2021,
  month = May
}
```

To cite the codebase that trained this model:
```bibtex
@misc{mesh-transformer-jax,
  author = {Wang, Ben},
  title = {{Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX}},
  howpublished = {\url{https://github.com/kingoflolz/mesh-transformer-jax}},
  year = 2021,
  month = May
}
```

### License:
Not available.

### Contact:
Reach out on [GitHub](https://github.com/kingoflolz/mesh-transformer-jax), Discord, or shoot Ben an email.

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
GPT-J learns an inner representation of the English language that can be used to extract features useful for downstream tasks. The model is best at generating text from a prompt.

### Primary intended users:
Researchers and developers who want to use a large language model for research or downstream tasks, particularly text generation.

### Out-of-scope uses:
GPT-J-6B is **not** intended for deployment without fine-tuning, supervision, and/or moderation. It is not intended for human-facing interactions as a product. It is not suitable for translation or generating text in languages other than English. It is not fine-tuned for specific downstream contexts like writing genre prose or commercial chatbots, and will not respond to prompts like ChatGPT.

---

## How to Use
This section outlines how to use the model.

This model can be easily loaded using the `AutoModelForCausalLM` functionality:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
```

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
The training data (the Pile) contains profanity, lewd, and abrasive language, which can influence the model to produce socially unacceptable text. The model is trained on English-language data only, limiting its performance in other languages. The model's performance is also affected by the prompt provided, as the statistically most likely next token is not always the most "accurate" or desired output.

### Evaluation factors:
Not available.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
The model performance is measured using LAMBADA Perplexity (PPL), LAMBADA Accuracy (Acc), Winogrande, Hellaswag, and PIQA. Lower LAMBADA PPL and higher accuracy scores on other benchmarks indicate better performance.

### Decision thresholds:
Not available.

### Variation approaches:
Not available.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
The evaluation results table includes performance on the LAMBADA, Winogrande, Hellaswag, and PIQA datasets. The table also includes "Dataset Size (GB)" for some models, with GPT-J 6B evaluated in comparison to models trained on the Pile dataset (825GB).

### Motivation:
These datasets are used to benchmark language models on common sense reasoning, reading comprehension, and language modeling tasks, providing a comparative evaluation of GPT-J 6B against other models like GPT-2, GPT-3, and GPT-Neo.

### Preprocessing:
Not available.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
GPT-J 6B was trained on [the Pile](https://pile.eleuther.ai), a large-scale curated dataset created by [EleutherAI](https://www.eleuther.ai). The Pile dataset size is 825GB.

### Motivation:
The Pile was chosen as a large-scale and diverse dataset for training a general-purpose language model.

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
Not available.

### Deploying Requirements:
Not available.

### Training or Fine-tuning Requirements:
TPU v3-256 pod was used for training.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

GPT-J was trained on the Pile, which contains profanity, lewd, and abrasive language. As a result, GPT-J may produce socially unacceptable or offensive text. It is important to remember that the model can generate harmful or offensive text.  It is recommended to have human curation or filtering of the outputs to censor undesirable content and improve quality.  The model may produce factually inaccurate output, and should not be depended upon for factual accuracy.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
As with all language models, it is hard to predict in advance how GPT-J will respond to particular prompts and offensive content may occur without warning. The evaluation datasets might have limitations, such as potential test set contamination in the training data of some compared models.

### Recommendations:
Evaluate the risks associated with your particular use case. We recommend having a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results. Never depend upon GPT-J to produce factually accurate output.

---

## Additional Information

<figure>

| Hyperparameter       | Value      |
|----------------------|------------|
| \\(n_{parameters}\\) | 6053381344 |
| \\(n_{layers}\\)     | 28&ast;    |
| \\(d_{model}\\)      | 4096       |
| \\(d_{ff}\\)         | 16384      |
| \\(n_{heads}\\)      | 16         |
| \\(d_{head}\\)       | 256        |
| \\(n_{ctx}\\)        | 2048       |
| \\(n_{vocab}\\)      | 50257/50400&dagger; (same tokenizer as GPT-2/3)  |
| Positional Encoding  | [Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) |
| RoPE Dimensions      | [64](https://github.com/kingoflolz/mesh-transformer-jax/blob/f2aa66e0925de6593dcbb70e72399b97b4130482/mesh_transformer/layers.py#L223) |
<figcaption><p><strong>&ast;</strong> Each layer consists of one feedforward block and one self attention block.</p>
<p><strong>&dagger;</strong> Although the embedding matrix has a size of 50400, only 50257 entries are used by the GPT-2 tokenizer.</p></figcaption></figure>

<figure>

|  Model                   | Public      | Training FLOPs | LAMBADA PPL ↓ | LAMBADA Acc ↑ | Winogrande ↑ | Hellaswag ↑ | PIQA ↑    | Dataset Size (GB) |
|--------------------------|-------------|----------------|---            |---            |---           |---          |---        |-------------------|
| Random Chance            | &check;     | 0              | ~a lot        | ~0%           | 50%          | 25%         | 25%       | 0                 |
| GPT-3 Ada&ddagger;       | &cross;     | -----          | 9.95          | 51.6%         | 52.9%        | 43.4%       | 70.5%     | -----             |
| GPT-2 1.5B               | &check;     | -----          | 10.63         | 51.21%        | 59.4%        | 50.9%       | 70.8%     | 40                |
| GPT-Neo 1.3B&ddagger;    | &check;     | 3.0e21         | 7.50          | 57.2%         | 55.0%        | 48.9%       | 71.1%     | 825               |
| Megatron-2.5B&ast;       | &cross;     | 2.4e21         | -----         | 61.7%         | -----        | -----       | -----     | 174               |
| GPT-Neo 2.7B&ddagger;    | &check;     | 6.8e21         | 5.63          | 62.2%         | 56.5%        | 55.8%       | 73.0%     | 825               |
| GPT-3 1.3B&ast;&ddagger; | &cross;     | 2.4e21         | 5.44          | 63.6%         | 58.7%        | 54.7%       | 75.1%     | ~800              |
| GPT-3 Babbage&ddagger;   | &cross;     | -----          | 5.58          | 62.4%         | 59.0%        | 54.5%       | 75.5%     | -----             |
| Megatron-8.3B&ast;       | &cross;     | 7.8e21         | -----         | 66.5%         | -----        | -----       | -----     | 174               |
| GPT-3 2.7B&ast;&ddagger; | &cross;     | 4.8e21         | 4.60          | 67.1%         | 62.3%        | 62.8%       | 75.6%     | ~800              |
| Megatron-11B&dagger;     | &check;     | 1.0e22         | -----         | -----         | -----        | -----       | -----     | 161               |
| **GPT-J 6B&ddagger;**    | **&check;** | **1.5e22**     | **3.99**      | **69.7%**     | **65.3%**    | **66.1%**   | **76.5%** | **825**           |
| GPT-3 6.7B&ast;&ddagger; | &cross;     | 1.2e22         | 4.00          | 70.3%         | 64.5%        | 67.4%       | 78.0%     | ~800              |
| GPT-3 Curie&ddagger;     | &cross;     | -----          | 4.00          | 69.3%         | 65.6%        | 68.5%       | 77.9%     | -----             |
| GPT-3 13B&ast;&ddagger;  | &cross;     | 2.3e22         | 3.56          | 72.5%         | 67.9%        | 70.9%       | 78.5%     | ~800              |
| GPT-3 175B&ast;&ddagger; | &cross;     | 3.1e23         | 3.00          | 76.2%         | 70.2%        | 78.9%       | 81.0%     | ~800              |
| GPT-3 Davinci&ddagger;   | &cross;     | -----          | 3.0           | 75%           | 72%          | 78%         | 80%       | -----             |
<figcaption><p>Models roughly sorted by performance, or by FLOPs if not available.</p>

<p><strong>&ast;</strong> Evaluation numbers reported by their respective authors. All other numbers are provided by
running <a href="https://github.com/EleutherAI/lm-evaluation-harness/"><code>lm-evaluation-harness</code></a> either with released
weights or with API access. Due to subtle implementation differences as well as different zero shot task framing, these
might not be directly comparable. See <a href="https://blog.eleuther.ai/gpt3-model-sizes/">this blog post</a> for more
details.</p>

<p><strong>†</strong> Megatron-11B provides no comparable metrics, and several implementations using the released weights do not
reproduce the generation quality and evaluations. (see <a href="https://github.com/huggingface/transformers/pull/10301">1</a>
<a href="https://github.com/pytorch/fairseq/issues/2358">2</a> <a href="https://github.com/pytorch/fairseq/issues/2719">3</a>)
Thus, evaluation was not attempted.</p>

<p><strong>‡</strong> These models have been trained with data which contains possible test set contamination. The OpenAI GPT-3 models
failed to deduplicate training data for certain test sets, while the GPT-Neo models as well as this one is
trained on the Pile, which has not been deduplicated against any test sets.</p></figcaption></figure>

Acknowledgements

This project would not have been possible without compute generously provided by Google through the
[TPU Research Cloud](https://sites.research.google/trc/), as well as the Cloud TPU team for providing early access to the [Cloud TPU VM](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms) Alpha.

Thanks to everyone who have helped out one way or another (listed alphabetically):
- [James Bradbury](https://twitter.com/jekbradbury) for valuable assistance with debugging JAX issues.
- [Stella Biderman](https://www.stellabiderman.com), [Eric Hallahan](https://twitter.com/erichallahan), [Kurumuz](https://github.com/kurumuz/), and [Finetune](https://github.com/finetuneanon/) for converting the model to be compatible with the `transformers` package.
- [Leo Gao](https://twitter.com/nabla_theta) for running zero shot evaluations for the baseline models for the table.
- [Laurence Golding](https://github.com/researcher2/) for adding some features to the web demo.
- [Aran Komatsuzaki](https://twitter.com/arankomatsuzaki) for advice with experiment design and writing the blog posts.
- [Janko Prester](https://github.com/jprester/) for creating the web demo frontend.

If you use this model, we would love to hear about it! Reach out on [GitHub](https://github.com/kingoflolz/mesh-transformer-jax), Discord, or shoot Ben an email.
