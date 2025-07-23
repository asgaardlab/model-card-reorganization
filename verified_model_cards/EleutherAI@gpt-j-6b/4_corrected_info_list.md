# GPT-J 6B

## Model Description
- GPT-J 6B is a transformer model.
- GPT-J 6B was trained using Ben Wang's [Mesh Transformer JAX](https://github.com/kingoflolz/mesh-transformer-jax/).
- "GPT-J" refers to the class of model.
- "6B" represents the number of trainable parameters.
- The model has a total of 6,053,381,344 parameters.
- The model consists of 28 layers.
- The model dimension is 4096.
- The feedforward dimension is 16384.
- The model dimension is split into 16 heads.
- Each head has a dimension of 256.
- Rotary Position Embedding (RoPE) is applied to 64 dimensions of each head.
- The model is trained with a tokenization vocabulary of 50257.
- The model uses the same set of BPEs as GPT-2/GPT-3.
- The model has a table of hyperparameters:
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
- Each layer consists of one feedforward block and one self-attention block.
- Although the embedding matrix has a size of 50400, only 50257 entries are used by the GPT-2 tokenizer.

## Intended Use and Limitations
- GPT-J learns an inner representation of the English language.
- The inner representation can be used to extract features useful for downstream tasks.
- The model is best at generating text from a prompt.
- GPT-J-6B is **not** intended for deployment without fine-tuning, supervision, and/or moderation.
- GPT-J-6B is not a product and cannot be used for human-facing interactions.
- The model may generate harmful or offensive text.
- It is important to evaluate the risks associated with your particular use case.
- GPT-J-6B was trained on an English-language only dataset.
- GPT-J-6B is **not** suitable for translation or generating text in other languages.
- GPT-J-6B has not been fine-tuned for downstream contexts in which language models are commonly deployed, such as writing genre prose, or commercial chatbots.
- GPT-J-6B will **not** respond to prompts like ChatGPT.
- ChatGPT was fine-tuned using methods such as Reinforcement Learning from Human Feedback (RLHF) to better “follow” human instructions..

### Limitations and Biases
- The core functionality of GPT-J is to predict the next token from a string of text.
- Language models are widely used for tasks other than token prediction.
- There are many unknowns with this work.
- The statistically most likely next token is often not the token that produces the most "accurate" text.
- Never depend upon GPT-J to produce factually accurate output.
- GPT-J was trained on the Pile, a dataset known to contain profanity, lewd, and abrasive language.
- Depending on the use case, GPT-J may produce socially unacceptable text.
- See [Sections 5 and 6 of the Pile paper](https://arxiv.org/abs/2101.00027) for a more detailed analysis of the biases in the Pile.
- It is hard to predict how GPT-J will respond to particular prompts.
- Offensive content may occur without warning.
- It is recommended to have a human curate or filter the outputs before releasing them, both to censor undesirable content and to improve the quality of the results.

### How to use
- This model can be easily loaded using the `AutoModelForCausalLM` functionality:
  ```python
  from transformers import AutoTokenizer, AutoModelForCausalLM

  tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
  model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
  ```

## Training data
- GPT-J 6B was trained on [the Pile](https://pile.eleuther.ai).
- The Pile is a large-scale curated dataset created by [EleutherAI](https://www.eleuther.ai).

## Training procedure
- This model was trained for 402 billion tokens.
- The training was conducted over 383,500 steps on a TPU v3-256 pod.
- The model was trained as an autoregressive language model.
- Cross-entropy loss was used to maximize the likelihood of predicting the next token correctly.

## Evaluation results
- The model has a table of evaluation results:
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
- Models are roughly sorted by performance or by FLOPs if not available.
- Evaluation numbers reported by their respective authors.
- All other numbers are provided by running [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/) either with released weights or with API access.
- Due to subtle implementation differences as well as different zero-shot task framing, these numbers might not be directly comparable.
- See [this blog post](https://blog.eleuther.ai/gpt3-model-sizes/) for more details.
- Megatron-11B provides no comparable metrics.
- Several implementations using the released weights do not reproduce the generation quality and evaluations.
- See [1](https://github.com/huggingface/transformers/pull/10301) [2](https://github.com/pytorch/fairseq/issues/2358) [3](https://github.com/pytorch/fairseq/issues/2719) for detail.
- Thus, evaluation was not attempted for Megatron-11B.
- These models have been trained with data which contains possible test set contamination.
- The OpenAI GPT-3 models failed to deduplicate training data for certain test sets.
- The GPT-Neo models and GPT-J are trained on the Pile, which has not been deduplicated against any test sets.

## Citation and Related Information

### BibTeX entry
- To cite this model:
  ```bibtex
  @misc{gpt-j,
    author = {Wang, Ben and Komatsuzaki, Aran},
    title = {{GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model}},
    howpublished = {\url{https://github.com/kingoflolz/mesh-transformer-jax}},
    year = 2021,
    month = May
  }
  ```
- To cite the codebase that trained this model:
  ```bibtex
  @misc{mesh-transformer-jax,
    author = {Wang, Ben},
    title = {{Mesh-Transformer-JAX: Model-Parallel Implementation of Transformer Language Model with JAX}},
    howpublished = {\url{https://github.com/kingoflolz/mesh-transformer-jax}},
    year = 2021,
    month = May
  }
  ```
- If you use this model, reach out on [GitHub](https://github.com/kingoflolz/mesh-transformer-jax), Discord, or email Ben.

## Acknowledgements
- This project was made possible by compute provided by Google through the [TPU Research Cloud](https://sites.research.google/trc/).
- Thanks to the Cloud TPU team for providing early access to the [Cloud TPU VM](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms) Alpha.
- Thanks to everyone who helped out (listed alphabetically):
  - [James Bradbury](https://twitter.com/jekbradbury) for assistance with debugging JAX issues.
  - [Stella Biderman](https://www.stellabiderman.com), [Eric Hallahan](https://twitter.com/erichallahan), [Kurumuz](https://github.com/kurumuz/), and [Finetune](https://github.com/finetuneanon/) for converting the model to be compatible with the `transformers` package.
  - [Leo Gao](https://twitter.com/nabla_theta) for running zero-shot evaluations for the baseline models for the table.
  - [Laurence Golding](https://github.com/researcher2/) for adding features to the web demo.
  - [Aran Komatsuzaki](https://twitter.com/arankomatsuzaki) for advice with experiment design and writing the blog posts.
  - [Janko Prester](https://github.com/jprester/) for creating the web demo frontend.