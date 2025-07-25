# Cerebras-GPT 13B
- Check out our [Blog Post](https://www.cerebras.net/cerebras-gpt).
- Check out our [arXiv paper](https://arxiv.org/abs/2304.03208)!

## Model Description
- The Cerebras-GPT family is released to facilitate research into LLM scaling laws.
- The models use open architectures and data sets.
- The release demonstrates the simplicity of training LLMs on the Cerebras software and hardware stack.
- All Cerebras-GPT models are available on Hugging Face.
- The family includes models with 111M, 256M, 590M, 1.3B, 2.7B, 6.7B, and 13B parameters.
- All models in the Cerebras-GPT family have been trained in accordance with [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556).
- The Chinchilla scaling laws recommend 20 tokens per model parameter, which is compute-optimal.
- These models were trained on the [Andromeda](https://www.cerebras.net/andromeda/) AI supercomputer.
- The Andromeda supercomputer is comprised of 16 CS-2 wafer scale systems.
- Cerebras' [weight streaming technology](https://www.cerebras.net/blog/linear-scaling-made-possible-with-weight-streaming) simplifies the training of LLMs.
- Weight streaming technology disaggregates compute from model storage.
- This technology allows for efficient scaling of training across nodes using simple data parallelism.
- Cerebras systems for pre-training and fine-tuning are available in the cloud via the [Cerebras Model Studio](https://www.cerebras.net/product-cloud/).
- Cerebras CS-2 compatible checkpoints are available in the [Cerebras Model Zoo](https://github.com/Cerebras/modelzoo).

## Model Details
- Developed by: [Cerebras Systems](https://www.cerebras.net/)
- License: Apache 2.0
- Model type: Transformer-based Language Model
- Architecture: GPT-3 style architecture
- Data set: The Pile
- Tokenizer: Byte Pair Encoding
- Vocabulary Size: 50257
- Sequence Length: 2048
- Optimizer: AdamW, (β1, β2) = (0.9, 0.95), adam_eps = 1e−8 (1e−9 for larger models)
- Positional Encoding: Learned
- Language: English
- Learn more: Dense Scaling Laws Paper for training procedure, config files, and details on how to use.
- **Contact**: To ask questions about Cerebras-GPT models, join the [Cerebras Discord](https://discord.gg/q6bZcMWJVu).
- This is the standard parameterization version of Cerebras-GPT with **13B** parameters.
- Related models: [Cerebras-GPT Models](https://huggingface.co/models?sort=downloads&search=cerebras-gpt).

| Model         | Parameters | Layers | d_model | Heads | d_head | d_ffn  | LR       | BS (seq) | BS (tokens)     |
|---------------|------------|--------|---------|-------|--------|--------|----------|----------|----------------|
| Cerebras-GPT  | 111M       | 10     | 768     | 12    | 64     | 3072   | 6.0E-04 | 120      | 246K           |
| Cerebras-GPT  | 256M       | 14     | 1088    | 17    | 64     | 4352   | 6.0E-04 | 264      | 541K           |
| Cerebras-GPT  | 590M       | 18     | 1536    | 12    | 128    | 6144   | 2.0E-04 | 264      | 541K           |
| Cerebras-GPT  | 1.3B       | 24     | 2048    | 16    | 128    | 8192   | 2.0E-04 | 528      | 1.08M          |
| Cerebras-GPT  | 2.7B       | 32     | 2560    | 32    | 80     | 10240  | 2.0E-04 | 528      | 1.08M          |
| Cerebras-GPT  | 6.7B       | 32     | 4096    | 32    | 128    | 16384  | 1.2E-04 | 1040     | 2.13M          |
| Cerebras-GPT  | 13B        | 40     | 5120    | 40    | 128    | 20480  | 1.2E-04 | 720 &rarr; 1080 | 1.47M &rarr; 2.21M    |

## Quickstart 
- This model can be easily loaded using the AutoModelForCausalLM functionality:
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-13B")
model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-13B")

text = "Generative AI is "
```
- The model can be used with Hugging Face Pipelines:
```python
from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
generated_text = pipe(text, max_length=50, do_sample=False, no_repeat_ngram_size=2)[0]
print(generated_text['generated_text'])
```
- The model can also be used with `model.generate()`:
```python
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, num_beams=5, 
                        max_new_tokens=50, early_stopping=True,
                        no_repeat_ngram_size=2)
text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text_output[0])
```

## Training data
- Cerebras-GPT is trained using [the Pile](https://pile.eleuther.ai) dataset from [EleutherAI](https://www.eleuther.ai).
- See the [Pile paper](https://arxiv.org/abs/2101.00027) for a more detailed breakdown of data sources and methodology.
- The Pile was cleaned using the ftfy library to normalize the text.
- The Pile was filtered using scripts provided by Eleuther.
- The data was tokenized using byte-pair encoding with the GPT-2 vocabulary.
- The tokenized version of the Pile has 371B tokens.
- More details about the training dataset preprocessing are included in Appendix A.1 of our paper.
- Recent works find significant duplicate data present in the Pile.
- Eleuther’s Pythia applies a deduplication process to reduce replicated data.
- The deduplication process decreases the Pile dataset size.
- Pythia was trained on both the standard dataset and deduplicated dataset to characterize the impact.
- Our models are trained on the standard Pile without deduplication.
- There may be an opportunity for further improvement with the deduplicated dataset.

## Training procedure
- We use the GPT-3 style model architecture.
- All layers use full attention as opposed to the GPT-3 style sparse banded attention.
- The model shapes were selected to either follow aspect ratio 80 or are the same shape as GPT-3 models.
- The learning rate was warmed up for 375M tokens (1500 steps for 111M and 256M models).
- The learning rate was decayed using a 10x cosine decay.
- No dropout was used.
- Weight decay was set to 0.1.
- All models are trained with a maximum sequence length (MSL) of 2048.
- All models were trained to the Chinchilla point: 20 tokens per model parameter.
- The number of steps was chosen based on optimal batch size (varied by model) and fixed sequence length (2048).
- See Training Table, below, for details.

Model Params | Sequence Length | Batch Size | Number of Steps | Tokens | Tokens per Parameter | Flops
------------ | -------------- | ---------- | --------------- | ------ | -------------------- | -----
111M         | 2048           | 120        | 9037            | 2.22E+09 | 20                  | 2.6E+18
256M         | 2048           | 264        | 9468            | 5.12E+09 | 20                  | 1.3E+19
590M         | 2048           | 264        | 21836           | 1.18E+10 | 20                  | 6.1E+19
1.3B         | 2048           | 528        | 24334           | 2.63E+10 | 20                  | 2.8E+20
2.7B         | 2048           | 528        | 49041           | 5.30E+10 | 20                  | 1.1E+21
6.7B         | 2048           | 1040       | 62522           | 1.33E+11 | 20                  | 6.3E+21
13B          | 2048           | 720        | 174335          | 2.57E+11 | 20                  | 2.3E+22

## Evaluations
- We trained models from smallest to largest.
- We fit a power law as we trained the models.
- The power law was helpful for extrapolating the validation loss of the next largest model.
- The power law provided confidence about whether the training run was going well.
- We performed upstream (pre-training) evaluations of text prediction cross-entropy using the Pile validation and test splits.
- We performed downstream evaluations of text generation accuracy on standardized tasks using the [Eleuther lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- Results are compared against many publicly available large language models in Section 3 of the paper.

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

## Uses and Limitations

### Intended Use
- The primary intended use is to further research into large language models.
- These models can be used as a foundation model for NLP applications, ethics, and alignment research.
- The primary intended users are researchers working to improve LLMs.
- Practitioners seeking reference implementations, training setups, hyperparameters, or pre-trained models are also intended users.
- The models are released with a fully permissive Apache license for the community to use freely.
- Users may fine-tune and adapt Cerebras-GPT models for deployment via either Cerebras [Model Studio](https://www.cerebras.net/product-cloud/) or third-party libraries.
- Further safety-related testing and mitigations should be applied before using the Cerebras-GPT model family in production downstream applications.
- Due to financial and compute budgets, Cerebras-GPT models were only trained and evaluated following the approaches described in the paper.

### Out of Scope Use
- Cerebras-GPT models are trained on the Pile dataset, which is in English only.
- The models are not suitable for machine translation tasks.
- Cerebras-GPT models have not been tuned for human-facing dialog applications like chatbots.
- The models will not respond to prompts in a similar way to models that have received instruction tuning or reinforcement learning from human feedback (RLHF) like Flan-T5 or ChatGPT.
- Cerebras-GPT models can be tuned using those methods.

### Risk, Bias, Ethical Considerations
- **Data**: The Pile dataset has been thoroughly analyzed from various ethical standpoints such as toxicity analysis, gender bias, pejorative content, and racially sensitive content.
- **Human life**: The outputs from this model may or may not align with human values.
- The risk needs to be thoroughly investigated before deploying this model in a production environment where it can directly impact human life.
- **Risks and harms**: There can be distributional bias in the Pile dataset that can manifest in various forms in downstream model deployment.
- There are other risks associated with large language models such as amplifying stereotypes, memorizing training data, or revealing private or secure information.
- **Mitigations**: Only mitigations in standard Pile dataset pre-processing were employed when pre-training Cerebras-GPT.

## Acknowledgements
- We are thankful to all Cerebras engineers, past and present, that made this work possible.