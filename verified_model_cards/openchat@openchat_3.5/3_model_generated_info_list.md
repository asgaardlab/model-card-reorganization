# OpenChat: Advancing Open-source Language Models with Mixed-Quality Data
- OpenChat is an open-source language model.
- OpenChat is fine-tuned with C-RLFT, a strategy inspired by offline reinforcement learning.
- OpenChat learns from mixed-quality data without preference labels.
- OpenChat delivers exceptional performance on par with ChatGPT.
- OpenChat has a model size of 7B parameters.
- OpenChat is committed to developing a high-performance, commercially viable, open-source large language model.
- OpenChat continues to make significant strides toward its vision.

<div align="center">
  <img src="https://raw.githubusercontent.com/imoneoi/openchat/master/assets/logo_new.png" style="width: 65%">
</div>

<p align="center">
  <a href="https://github.com/imoneoi/openchat">GitHub Repo</a> •
  <a href="https://openchat.team">Online Demo</a> •
  <a href="https://discord.gg/pQjnXvNKHY">Discord</a> •
  <a href="https://twitter.com/imonenext">Twitter</a> •
  <a href="https://huggingface.co/openchat">Huggingface</a> •
  <a href="https://arxiv.org/pdf/2309.11235.pdf">Paper</a>
</p>

- The first 7B model achieves comparable results with ChatGPT (March).
- OpenChat is the #1 open-source model on MT-bench scoring 7.81.
- OpenChat outperforms 70B models.

<div align="center" style="justify-content: center; align-items: center; ">
  <img src="https://github.com/alpayariyak/openchat/blob/master/assets/3.5-benchmarks.png?raw=true" style="width: 100%;  border-radius: 0.5em">
</div>

[![DOI](https://zenodo.org/badge/645397533.svg)](https://zenodo.org/badge/latestdoi/645397533)

## Usage
- To use this model, install the OpenChat package by following the [installation guide](https://github.com/imoneoi/openchat#installation).
- Use the OpenChat OpenAI-compatible API server by running the serving command from the table below.
- The server is optimized for high-throughput deployment using [vLLM](https://github.com/vllm-project/vllm).
- The server can run on a consumer GPU with 24GB RAM.
- To enable tensor parallelism, append `--tensor-parallel-size N` to the serving command.
- Once started, the server listens at `localhost:18888` for requests.
- The server is compatible with the [OpenAI ChatCompletion API specifications](https://platform.openai.com/docs/api-reference/chat).
- Refer to the example request below for reference.
- You can use the [OpenChat Web UI](https://github.com/imoneoi/openchat#web-ui) for a user-friendly experience.
- To deploy the server as an online service, use `--api-keys sk-KEY1 sk-KEY2 ...` to specify allowed API keys.
- Use `--disable-log-requests --disable-log-stats --log-file openchat.log` for logging only to a file.
- For security purposes, it is recommended to use an [HTTPS gateway](https://fastapi.tiangolo.com/es/deployment/concepts/#security-https) in front of the server.

<details>
  <summary>Example request (click to expand)</summary>

```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openchat_3.5",
    "messages": [{"role": "user", "content": "You are a large language model named OpenChat. Write a poem to describe yourself"}]
  }'
```

- Coding Mode

```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openchat_3.5",
    "condition": "Code",
    "messages": [{"role": "user", "content": "Write an aesthetic TODO app using HTML5 and JS, in a single file. You should use round corners and gradients to make it more aesthetic."}]
  }'
```

</details>

| Model        | Size | Context | Weights                                                     | Serving                                                                                                     |
|--------------|------|---------|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| OpenChat 3.5 | 7B   | 8192    | [Huggingface](https://huggingface.co/openchat/openchat_3.5) | `python -m ochat.serving.openai_api_server --model openchat/openchat_3.5 --engine-use-ray --worker-use-ray` |

- For inference with Huggingface Transformers (slow and not recommended), follow the conversation template provided below.

<details>
  <summary>Conversation templates (click to expand)</summary>

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

</details>

- The GPT4 template is also available as the integrated `tokenizer.chat_template`.
- The `tokenizer.chat_template` can be used instead of manually specifying the template.

```python
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
    {"role": "user", "content": "How are you today?"}
]
tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
assert tokens == [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 22557, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747, 15359, 32000, 420, 6316, 28781, 3198, 3123, 1247, 28747, 1602, 460, 368, 3154, 28804, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]
```

## Comparison with [X.AI Grok models](https://x.ai/)
- OpenChat has 7 billion parameters.
- Grok has 33 billion parameters.
- OpenChat outperformed Grok with fewer parameters.
- OpenChat is humorous and witty.
- OpenChat encourages open research and collaboration in AI.

|              | License     | # Param | Average  | MMLU | HumanEval | MATH     | GSM8k    |
|--------------|-------------|---------|----------|------|-----------|----------|----------|
| OpenChat 3.5 | Apache-2.0  | 7B      | **56.4** | 64.3 | 55.5      | **28.6** | **77.3** |
| Grok-0       | Proprietary | 33B     | 44.5     | 65.7 | 39.7      | 15.7     | 56.8     |
| Grok-1       | Proprietary | ?       | 55.8     | 73   | 63.2      | 23.9     | 62.9     |

## <a id="benchmarks"></a> Benchmarks
| Model              | # Params | Average  | MT-Bench     | AGIEval  | BBH MC   | TruthfulQA    | MMLU         | HumanEval       | BBH CoT     | GSM8K        |
|--------------------|----------|----------|--------------|----------|----------|---------------|--------------|-----------------|-------------|--------------|
| OpenChat-3.5       | **7B**   | **61.6** | 7.81         | **47.4** | **47.6** | **59.1**      | 64.3         | **55.5**        | 63.5        | **77.3**     |
| ChatGPT (March)*   | ?        | 61.5     | **7.94**     | 47.1     | **47.6** | 57.7          | **67.3**     | 48.1            | **70.1**    | 74.9         |
|                    |          |          |              |          |          |               |              |                 |             |              |
| OpenHermes 2.5     | 7B       | 59.3     | 7.54         | 46.5     | 49.4     | 57.5          | 63.8         | 48.2            | 59.9        | 73.5         |
| OpenOrca Mistral   | 7B       | 52.7     | 6.86         | 42.9     | 49.4     | 45.9          | 59.3         | 38.4            | 58.1        | 59.1         |
| Zephyr-β^          | 7B       | 34.6     | 7.34         | 39.0     | 40.6     | 40.8          | 39.8         | 22.0            | 16.0        | 5.1          |
| Mistral            | 7B       | -        | 6.84         | 38.0     | 39.0     | -             | 60.1         | 30.5            | -           | 52.2         |
| Open-source SOTA** | 13B-70B  | 61.4     | 7.71         | 41.7     | 49.7     | 62.3          | 63.7         | 73.2            | 41.4        | 82.3         |

- ChatGPT (March) results are from [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774), [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub), and evaluation.
- ChatGPT is not a fixed baseline and evolves rapidly over time.
- Zephyr-β often fails to follow few-shot CoT instructions.
- Zephyr-β was aligned with only chat data but not trained on few-shot data.
- Mistral and Open-source SOTA results are taken from reported results in instruction-tuned model papers and official repositories.
- All models are evaluated in chat mode.
- All zero-shot benchmarks follow the same setting as in the AGIEval paper and Orca paper.
- CoT tasks use the same configuration as Chain-of-Thought Hub.
- HumanEval is evaluated with EvalPlus.
- MT-bench is run using FastChat.
- To reproduce results, follow the instructions in [our repository](https://github.com/imoneoi/openchat/#benchmarks).

## Limitations
- OpenChat is bound by limitations inherent in its foundation models.
- Limitations may impact performance in complex reasoning.
- Limitations may impact performance in mathematical and arithmetic tasks.
- Limitations may impact performance in programming and coding challenges.
- OpenChat may generate information that does not exist or is not accurate, known as "hallucination".
- Users should verify any critical information obtained from the model.
- OpenChat may generate harmful, hate speech, biased responses, or answer unsafe questions.
- Additional AI safety measures are crucial in use cases requiring safe and moderated responses.

## License
- OpenChat 3.5 code and models are distributed under the Apache License 2.0.

## Dataset Details
- OpenChat 3.5 was trained with C-RLFT on publicly available high-quality instruction data.
- OpenChat has a custom processing pipeline.
- Notable subsets included:
  - [OpenChat ShareGPT](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset)
  - [Open-Orca with FLAN answers](https://huggingface.co/datasets/imone/OpenOrca_FLAN)
  - Capybara [1](https://huggingface.co/datasets/LDJnr/Pure-Dove) [2](https://huggingface.co/datasets/LDJnr/Verified-Camel) [3](https://huggingface.co/datasets/LDJnr/LessWrong-Amplify-Instruct)
  - [GOAT](https://huggingface.co/datasets/tiedong/goat)
  - [Glaive](https://huggingface.co/datasets/glaiveai/glaive-code-assistant)
  - [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)
  - [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
  - [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25)

## Citation
```
@article{wang2023openchat,
  title={OpenChat: Advancing Open-source Language Models with Mixed-Quality Data},
  author={Wang, Guan and Cheng, Sijie and Zhan, Xianyuan and Li, Xiangang and Song, Sen and Liu, Yang},
  journal={arXiv preprint arXiv:2309.11235},
  year={2023}
}
```

## 💌 Contact
- Project Lead: Guan Wang [imonenext at gmail dot com]
- [Alpay Ariyak](https://github.com/alpayariyak) [aariyak at wpi dot edu]