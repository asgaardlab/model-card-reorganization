# MPT-7B
- MPT-7B is a decoder-style transformer.
- MPT-7B was pretrained from scratch on 1 trillion tokens of English text and code.
- This model was trained by [MosaicML](https://www.mosaicml.com).
- MPT-7B is part of the family of MosaicPretrainedTransformer (MPT) models.
- MPT models use a modified transformer architecture optimized for efficient training and inference.
- Architectural changes include performance-optimized layer implementations.
- MPT models eliminate context length limits by replacing positional embeddings with Attention with Linear Biases ([ALiBi](https://arxiv.org/abs/2108.12409)).
- Thanks to these modifications, MPT models can be trained with high throughput efficiency.
- MPT models have stable convergence.
- MPT models can be served efficiently with standard HuggingFace pipelines.
- MPT models can also be served efficiently with NVIDIA's [FasterTransformer](https://github.com/NVIDIA/FasterTransformer).
- This model uses the MosaicML LLM codebase.
- The MosaicML LLM codebase can be found in the [llm-foundry repository](https://github.com/mosaicml/llm-foundry).
- MPT-7B was trained by MosaicML’s NLP team.
- MPT-7B was trained on the [MosaicML platform](https://www.mosaicml.com/training) for LLM pretraining, finetuning, and inference.

### How is this model different?
- MPT-7B is licensed for the possibility of commercial use (unlike [LLaMA](https://arxiv.org/abs/2302.13971)).
- MPT-7B was trained on a large amount of data (1 trillion tokens).
- MPT-7B can handle extremely long inputs thanks to [ALiBi](https://arxiv.org/abs/2108.12409).
- MPT-7B can handle up to 84k tokens compared to 2k-4k for other open-source models.
- MPT-7B is capable of fast training and inference (via [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf) and [FasterTransformer](https://github.com/NVIDIA/FasterTransformer)).
- MPT-7B is equipped with highly efficient open-source training code via the [llm-foundry repository](https://github.com/mosaicml/llm-foundry).

### Models finetuned off MPT-7B:
- The following models are finetuned on MPT-7B:
  - [MPT-7B-StoryWriter-65k+](https://huggingface.co/mosaicml/mpt-7b-storywriter): 
    - A model designed to read and write fictional stories with super long context lengths.
    - Built by finetuning MPT-7B with a context length of 65k tokens on a filtered fiction subset of the [books3 dataset](https://huggingface.co/datasets/the_pile_books3).
    - At inference time, MPT-7B-StoryWriter-65k+ can extrapolate even beyond 65k tokens.
    - Generations as long as 80k tokens can be demonstrated on a single A100-80GB GPU in our [blogpost](www.mosaicml.com/blog/mpt-7b).
    - License: Apache 2.0
  - [MPT-7B-Instruct](https://huggingface.co/mosaicml/mpt-7b-instruct): 
    - A model for short-form instruction following.
    - Built by finetuning MPT-7B on a [dataset](https://huggingface.co/datasets/mosaicml/dolly_hhrlhf) derived from the [Databricks Dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and the [Anthropic Helpful and Harmless (HH-RLHF)](https://huggingface.co/datasets/Anthropic/hh-rlhf) datasets.
    - License: Apache 2.0
  - [MPT-7B-Chat](https://huggingface.co/mosaicml/mpt-7b-chat): 
    - A chatbot-like model for dialogue generation.
    - Built by finetuning MPT-7B on the [ShareGPT-Vicuna](https://huggingface.co/datasets/jeffwan/sharegpt_vicuna), [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3), [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca), [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf), and [Evol-Instruct](https://huggingface.co/datasets/victor123/evol_instruct_70k) datasets.
    - License: _CC-By-NC-SA-4.0_

## Model Date
- The model date is May 5, 2023.

## Model License
- The model license is Apache-2.0.

## Documentation
- [Blog post: Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs](https://www.mosaicml.com/blog/mpt-7b)
- [Codebase (mosaicml/llm-foundry repo)](https://github.com/mosaicml/llm-foundry/)
- Questions: Feel free to contact us via the [MosaicML Community Slack](https://mosaicml.me/slack)!

## How to Use
- This model is best used with the MosaicML [llm-foundry repository](https://github.com/mosaicml/llm-foundry) for training and finetuning.
- Code snippet to use the model:
  ```python
  import transformers
  model = transformers.AutoModelForCausalLM.from_pretrained(
    'mosaicml/mpt-7b',
    trust_remote_code=True
  )
  ```
- Note: This model requires that `trust_remote_code=True` be passed to the `from_pretrained` method.
- The model uses a custom `MPT` model architecture that is not yet part of the Hugging Face `transformers` package.
- `MPT` includes options for many training efficiency features such as [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf), [ALiBi](https://arxiv.org/abs/2108.12409), [QK LayerNorm](https://arxiv.org/abs/2010.04245), and more.
- To use the optimized [triton implementation](https://github.com/openai/triton) of FlashAttention, you can load the model on GPU (`cuda:0`) with `attn_impl='triton'` and with `bfloat16` precision:
  ```python
  import torch
  import transformers

  name = 'mosaicml/mpt-7b'

  config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
  config.attn_config['attn_impl'] = 'triton'
  config.init_device = 'cuda:0' # For fast initialization directly on GPU!

  model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    torch_dtype=torch.bfloat16, # Load model weights in bfloat16
    trust_remote_code=True
  )
  ```
- Although the model was trained with a sequence length of 2048, ALiBi enables users to increase the maximum sequence length during finetuning and/or inference. For example:
  ```python
  import transformers

  name = 'mosaicml/mpt-7b'

  config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
  config.max_seq_len = 4096 # (input + output) tokens can now be up to 4096

  model = transformers.AutoModelForCausalLM.from_pretrained(
    name,
    config=config,
    trust_remote_code=True
  )
  ```
- This model was trained with the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer.
  ```python
  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
  ```
- The model can then be used within a text-generation pipeline.
- Note: when running Torch modules in lower precision, it is best practice to use the [torch.autocast context manager](https://pytorch.org/docs/stable/amp.html).
  ```python
  from transformers import pipeline

  pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, device='cuda:0')

  with torch.autocast('cuda', dtype=torch.bfloat16):
      print(
          pipe('Here is a recipe for vegan banana bread:\n',
              max_new_tokens=100,
              do_sample=True,
              use_cache=True))
  ```

## Model Description
- The architecture is a modification of a standard decoder-only transformer.
- The model has been modified from a standard transformer in the following ways:
  - It uses [FlashAttention](https://arxiv.org/pdf/2205.14135.pdf).
  - It uses [ALiBi (Attention with Linear Biases)](https://arxiv.org/abs/2108.12409) and does not use positional embeddings.
  - It does not use biases.

| Hyperparameter | Value |
|----------------|-------|
| n_parameters    | 6.7B |
| n_layers        | 32   |
| n_heads         | 32   |
| d_model         | 4096 |
| vocab size      | 50432 |
| sequence length  | 2048 |

## Training Data
### Streaming Datasets
- Data was formatted using the MosaicML [StreamingDataset](https://github.com/mosaicml/streaming) library.
- StreamingDataset allows instant resumption of training from any point in the dataset.
- StreamingDataset obviates the need to download the whole dataset before starting training.

### Data Mix
- The model was trained for 1 trillion tokens (with batch size 1760 and sequence length 2048).
- The model was trained on the following data mix:
| Data Source | Number of Tokens in Source | Proportion | Effective Number of Tokens | Epochs |
|-------------|----------------------------|------------|----------------------------|--------|
| mC4 3.1.0 - English | 417.99 B | 0.33 | 330 B | 0.14 |
| C4 - English - SemDedup 80% | 100.42 B | 0.299 | 299 B | 2.98 |
| RedPajama - CommonCrawl | 878.45 B | 0.1 | 100 B | 0.11 |
| The Stack - Selected Languages | 463.78 B | 0.1 | 100 B | 0.22 |
| RedPajama - Wikipedia - En | 4.87 B | 0.04 | 40 B | 8.21 |
| The Stack - Markdown | 107.07 B | 0.035 | 35 B | 0.33 |
| S2ORC | 48.85 B | 0.033 | 33 B | 0.68 |
| RedPajama - Books | 26.02 B | 0.03 | 30 B | 1.15 |
| RedPajama - arXiv | 28.10 B | 0.019 | 19 B | 0.68 |
| RedPajama - StackExchange | 20.54 B | 0.014 | 14 B | 0.68 |

- Samples for each batch were selected from one of the datasets with the probability specified above.
- The examples were shuffled within each dataset.
- Each example was constructed from as many sequences from that dataset as were necessary to fill the 2048 sequence length.
- The data was tokenized using the [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) tokenizer.
- The model vocabulary size of 50432 was set to be a multiple of 128.

### Training Configuration
- This model was trained on 440 A100-40GBs for about 9.5 days using the [MosaicML Platform](https://www.mosaicml.com/platform).
- The model was trained with sharded data parallelism using [FSDP](https://pytorch.org/docs/stable/fsdp.html).
- The model used the [LION](https://arxiv.org/abs/2302.06675) optimizer.

## Limitations and Biases
- MPT-7B (Base) is not intended for deployment without finetuning.
- MPT-7B should not be used for human-facing interactions without further guardrails and user consent.
- MPT-7B can produce factually incorrect output.
- MPT-7B should not be relied on to produce factually accurate information.
- MPT-7B was trained on various public datasets.
- It is possible that this model could generate lewd, biased, or otherwise offensive outputs.

## MosaicML Platform
- If you're interested in [training](https://www.mosaicml.com/training) and [deploying](https://www.mosaicml.com/inference) your own MPT or LLMs on the MosaicML Platform, [sign up here](https://forms.mosaicml.com/demo?utm_source=huggingface&utm_medium=referral&utm_campaign=mpt-7b).

## Disclaimer
- The license on this model does not constitute legal advice.
- We are not responsible for the actions of third parties who use this model.
- Please consult an attorney before using this model for commercial purposes.

## Citation
- Please cite this model using the following format:
```
@online{MosaicML2023Introducing,
    author    = {MosaicML NLP Team},
    title     = {Introducing MPT-7B: A New Standard for Open-Source,
    Commercially Usable LLMs},
    year      = {2023},
    url       = {www.mosaicml.com/blog/mpt-7b},
    note      = {Accessed: 2023-05-05},
    urldate   = {2023-05-05}
}
```