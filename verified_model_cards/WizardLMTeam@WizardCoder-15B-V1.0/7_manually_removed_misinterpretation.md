## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
🏠 <a href="https://wizardlm.github.io/" target="_blank">Home Page</a>
🤗 <a href="https://huggingface.co/WizardLM" target="_blank">HF Repo</a>  •🐱 <a href="https://github.com/nlpxucan/WizardLM" target="_blank">Github Repo</a> • 🐦 <a href="https://twitter.com/WizardLM_AI" target="_blank">Twitter</a>

### Model date:
[2024/01/04] Release of **WizardCoder-33B-V1.1**.
[2024/01/04] **WizardCoder-33B-V1.1** outperforms **ChatGPT 3.5**, **Gemini Pro**, and **DeepSeek-Coder-33B-instruct** on HumanEval and HumanEval-Plus pass@1.
[2024/01/04] **WizardCoder-33B-V1.1** is comparable with **ChatGPT 3.5**, and surpasses **Gemini Pro** on MBPP and MBPP-Plus pass@1.
[08/11/2023] Release of **WizardMath** Models.
Release of **WizardCoder-15B-v1.0** trained with **78k** evolved code instructions.

### Model version:
WizardCoder-33B-V1.1, WizardCoder-Python-34B-V1.0, WizardCoder-15B-V1.0, WizardCoder-Python-13B-V1.0, WizardCoder-Python-7B-V1.0, WizardCoder-3B-V1.0, WizardCoder-1B-V1.0, WizardMath-70B-V1.0, WizardMath-13B-V1.0, WizardMath-7B-V1.0, WizardLM-13B-V1.2, WizardLM-13B-V1.1, WizardLM-30B-V1.0, WizardLM-13B-V1.0, WizardLM-7B-V1.0.
WizardCoder-33B-V1.1 is trained from deepseek-coder-33b-base, the **SOTA OSS Code LLM** on [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html).

### Model type:
Code Large Language Models.
WizardCoder model is developed by adapting the Evol-Instruct method specifically for coding tasks and fine-tuning the Code LLM, StarCoder, utilizing the newly created instruction-following training set.

### Training details:
To develop WizardCoder model, the Evol-Instruct method is adapted specifically for coding tasks.
Code LLM, StarCoder, is fine-tuned utilizing the newly created instruction-following training set.
Fine-tuning uses modified code `train.py` from [Llama-X](https://github.com/AetherCortex/Llama-X).
Fine-tuning StarCoder-15B with the following hyperparameters:
| Hyperparameter | StarCoder-15B |
|----------------|---------------|
| Batch size     | 512           |
| Learning rate  | 2e-5          |
| Epochs         | 3             |
| Max length     | 2048          |
| Warmup step    | 30            |
| LR scheduler   | cosine        |

### Paper or other resource for more information:
📃 <a href="https://arxiv.org/abs/2304.12244" target="_blank">[WizardLM]</a>
📃 <a href="https://arxiv.org/abs/2306.08568" target="_blank">[WizardCoder]</a>
📃 <a href="https://arxiv.org/abs/2308.09583" target="_blank">[WizardMath]</a>
<a href="https://wizardlm.github.io/" target="_blank">Home Page</a>
<a href="https://huggingface.co/WizardLM" target="_blank">HF Repo</a>
<a href="https://github.com/nlpxucan/WizardLM" target="_blank">Github Repo</a>
<a href="https://twitter.com/WizardLM_AI" target="_blank">Twitter</a>
Join our <a href="https://discord.gg/VZjjHtWrKs" target="_blank">Discord</a>

### Citation details:
```
@article{luo2023wizardcoder,
  title={WizardCoder: Empowering Code Large Language Models with Evol-Instruct},
  author={Luo, Ziyang and Xu, Can and Zhao, Pu and Sun, Qingfeng and Geng, Xiubo and Hu, Wenxiang and Tao, Chongyang and Ma, Jing and Lin, Qingwei and Jiang, Daxin},
  journal={arXiv preprint arXiv:2306.08568},
  year={2023}
}
```

### License:
WizardCoder model follows the same license as StarCoder.
WizardCoder-33B-V1.1: <a href="https://huggingface.co/WizardLM/WizardMath-7B-V1.1/resolve/main/LICENSE" target="_blank">MSFTResearch</a>
WizardCoder-Python-34B-V1.0, WizardMath-70B-V1.0, WizardMath-13B-V1.0, WizardMath-7B-V1.0, WizardLM-13B-V1.2: <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama2</a> or <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama 2  </a> or  <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama 2 License </a>
WizardCoder-15B-V1.0, WizardCoder-3B-V1.0, WizardCoder-1B-V1.0: <a href="https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement" target="_blank">OpenRAIL-M</a>
WizardCoder-Python-13B-V1.0, WizardCoder-Python-7B-V1.0: <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama2</a>
WizardLM-13B-V1.1, WizardLM-30B-V1.0, WizardLM-13B-V1.0, WizardLM-7B-V1.0: Non-commercial

### Contact:
👋 Join our <a href="https://discord.gg/VZjjHtWrKs" target="_blank">Discord</a>
Twitter account: https://twitter.com/WizardLM_AI
HuggingFace Repo: https://huggingface.co/WizardLM
Github Repo: https://github.com/nlpxucan/WizardLM

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
Empowering Code Large Language Models with Evol-Instruct.
Tailored for coding tasks.
Capabilities of the model are demonstrated in benchmarks like HumanEval, MBPP.

### Primary intended users:
We welcome everyone to use your professional and difficult instructions to evaluate WizardCoder.

### Out-of-scope uses:
Not available.

---

## How to Use
This section outlines how to use the model.

We provide Online Demo, Fine-tuning and Inference instructions.
**Online Demo:**
We will provide our latest models for you to try for as long as possible. If you find a link is not working, please try another one. At the same time, please try as many **real-world** and **challenging** code-related problems that you encounter in your work and life as possible. We will continue to evolve our models with your feedbacks.
Online Demo links are available in the tables for WizardMath models:
[Demo](http://47.103.63.15:50083/), [Demo](http://47.103.63.15:50082/), [Demo](http://47.103.63.15:50080/)

**Fine-tuning:**
To reproduce fine-tuning of WizardCoder, follow these steps:
1. According to the instructions of [Llama-X](https://github.com/AetherCortex/Llama-X), install the environment, download the training code, and deploy. (Note: `deepspeed==0.9.2` and `transformers==4.29.2`)
2. Replace the `train.py` with the `train_wizardcoder.py` in our repo (`src/train_wizardcoder.py`)
3. Login Huggingface:
```bash
huggingface-cli login
```
4. Execute the training command:
```bash
deepspeed train_wizardcoder.py \
    --model_name_or_path "bigcode/starcoder" \
    --data_path "/your/path/to/code_instruction_data.json" \
    --output_dir "/your/path/to/ckpt" \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --warmup_steps 30 \
    --logging_steps 2 \
    --lr_scheduler_type "cosine" \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True
```

**Inference:**
Provide decoding script for WizardCoder, which reads an input file and generates corresponding responses for each sample, and finally consolidates them into an output file.
Specify `base_model`, `input_data_path` and `output_data_path` in `src\inference_wizardcoder.py` to set the decoding model, path of input file and path of output file.
```bash
pip install jsonlines
```
Decoding command:
```
python src\inference_wizardcoder.py \
    --base_model "/your/path/to/ckpt" \
    --input_data_path "/your/path/to/input/data.jsonl" \
    --output_data_path "/your/path/to/output/result.jsonl"
```
Format of `data.jsonl`:
```
{"idx": 11, "Instruction": "Write a Python code to count 1 to 10."}
{"idx": 12, "Instruction": "Write a Jave code to sum 1 to 10."}
```
Prompt for WizardCoder in `src\inference_wizardcoder.py`:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
```

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Not available.

### Evaluation factors:
Not available.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
pass@1 on benchmarks like HumanEval, HumanEval+, MBPP, MBPP+, GSM8k, MATH, MT-Bench, AlpacaEval, WizardEval.

### Decision thresholds:
Not available.

### Variation approaches:
Evaluation on HumanEval is done by generating code solutions for each problem utilizing a **single attempt**, and the resulting pass rate percentage is reported.
WizardCoder generates answers using greedy decoding and tests with the same [code](https://github.com/evalplus/evalplus) or [code](https://github.com/openai/human-eval/tree/master).
For HumanEval and MBPP benchmarks, 20 samples are generated for each problem to estimate the pass@1 score.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
HumanEval Benchmarks (<a href="https://github.com/openai/human-eval">https://github.com/openai/human-eval</a>), EvalPlus Leaderboard (<a href="https://evalplus.github.io/leaderboard.html">https://evalplus.github.io/leaderboard.html</a>), MBPP, GSM8k Benchmarks (<a href="https://github.com/openai/grade-school-math">https://github.com/openai/grade-school-math</a>), MATH Benchmarks (<a href="https://github.com/hendrycks/math">https://github.com/hendrycks/math</a>), MT-Bench, AlpacaEval, WizardEval.

### Motivation:
Not available.

### Preprocessing:
Not available.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
Evolved code instructions. For WizardCoder-15B-v1.0, trained with **78k** evolved code instructions.

### Motivation:
Not available.

### Preprocessing:
Evol-Instruct method is used to create instruction-following training set for coding tasks.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
|  Model  |  Checkpoint  | Paper    | HumanEval  |   HumanEval+ | MBPP | MBPP+ | License |
| ----- |------| ---- |------|-------| ----- |  ----- |----- |
|  GPT-4-Turbo (Nov 2023)  | - | - | 85.4  | 81.7 | 83.0 | 70.7 |-|
|  GPT-4 (May 2023)  | - | - | 88.4  | 76.8 | - | - |-|
|  GPT-3.5-Turbo (Nov 2023)  | - | - | 72.6  | 65.9 | 81.7 | 69.4 |-|
|  Gemini Pro  | - | - | 63.4  | 55.5 | 72.9 | 57.9 |-|
|  DeepSeek-Coder-33B-instruct | - | - |  78.7 | 72.6 | 78.7 | 66.7 |-|
|  **WizardCoder-33B-V1.1**  |   🤗 <a href="https://huggingface.co/WizardLM/WizardCoder-33B-V1.1" target="_blank">HF Link</a>   |  📃 <a href="https://arxiv.org/abs/2306.08568" target="_blank">[WizardCoder]</a>  |  79.9  | 73.2 | 78.9 | 66.9 |  <a href="https://huggingface.co/WizardLM/WizardMath-7B-V1.1/resolve/main/LICENSE" target="_blank">MSFTResearch</a>  |
|  WizardCoder-Python-34B-V1.0  |   🤗 <a href="https://huggingface.co/WizardLM/WizardCoder-Python-34B-V1.0" target="_blank">HF Link</a>   |  📃 <a href="https://arxiv.org/abs/2306.08568" target="_blank">[WizardCoder]</a>  |  73.2   | 64.6 | 73.2 | 59.9 |  <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama2</a>  |
|  WizardCoder-15B-V1.0  |   🤗 <a href="https://huggingface.co/WizardLM/WizardCoder-15B-V1.0" target="_blank">HF Link</a>   |  📃 <a href="https://arxiv.org/abs/2306.08568" target="_blank">[WizardCoder]</a>  |  59.8   | 52.4 | -- | -- |  <a href="https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement" target="_blank">OpenRAIL-M</a>  |
|  WizardCoder-Python-13B-V1.0  |   🤗 <a href="https://huggingface.co/WizardLM/WizardCoder-Python-13B-V1.0" target="_blank">HF Link</a>   |  📃 <a href="https://arxiv.org/abs/2306.08568" target="_blank">[WizardCoder]</a>  |  64.0   | -- | -- | -- |  <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama2</a>  |
|  WizardCoder-Python-7B-V1.0  |   🤗 <a href="https://huggingface.co/WizardLM/WizardCoder-Python-7B-V1.0" target="_blank">HF Link</a>   |  📃 <a href="https://arxiv.org/abs/2306.08568" target="_blank">[WizardCoder]</a>  |  55.5   | -- | -- | -- |  <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama2</a>  |
|  WizardCoder-3B-V1.0  |   🤗 <a href="https://huggingface.co/WizardLM/WizardCoder-3B-V1.0" target="_blank">HF Link</a>   |  📃 <a href="https://arxiv.org/abs/2306.08568" target="_blank">[WizardCoder]</a>  |  34.8   | -- | -- | -- |  <a href="https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement" target="_blank">OpenRAIL-M</a>  |
|  WizardCoder-1B-V1.0  |   🤗 <a href="https://huggingface.co/WizardLM/WizardCoder-1B-V1.0" target="_blank">HF Link</a>   |  📃 <a href="https://arxiv.org/abs/2306.08568" target="_blank">[WizardCoder]</a>  |  23.8   | -- | -- | -- |  <a href="https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement" target="_blank">OpenRAIL-M</a>  |

| Model | Checkpoint | Paper  | GSM8k | MATH  |Online Demo| License|
| ----- |------| ---- |------|-------| ----- | ----- |
| WizardMath-70B-V1.0 | 🤗 <a href="https://huggingface.co/WizardLM/WizardMath-70B-V1.0" target="_blank">HF Link</a> |  📃 <a href="https://arxiv.org/abs/2308.09583" target="_blank">[WizardMath]</a>| **81.6**  |  **22.7**	|[Demo](http://47.103.63.15:50083/)| <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama 2  </a> |
| WizardMath-13B-V1.0 | 🤗 <a href="https://huggingface.co/WizardLM/WizardMath-13B-V1.0" target="_blank">HF Link</a> |  📃 <a href="https://arxiv.org/abs/2308.09583" target="_blank">[WizardMath]</a>| **63.9**  |  **14.0** |[Demo](http://47.103.63.15:50082/)| <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama 2 </a> |
| WizardMath-7B-V1.0 | 🤗 <a href="https://huggingface.co/WizardLM/WizardMath-7B-V1.0" target="_blank">HF Link</a>  |  📃 <a href="https://arxiv.org/abs/2308.09583" target="_blank">[WizardMath]</a>| 	 **54.9**  |  **10.7** | [Demo](http://47.103.63.15:50080/)|  <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama 2  </a>|

| <sup>Model</sup> | <sup>Checkpoint</sup> | <sup>Paper</sup> |<sup>MT-Bench</sup> | <sup>AlpacaEval</sup> | <sup>WizardEval</sup> | <sup>HumanEval</sup>  | <sup>License</sup>|
| ----- |------| ---- |------|-------| ----- | ----- | ----- |
| <sup>WizardLM-13B-V1.2</sup> | <sup>🤗 <a href="https://huggingface.co/WizardLM/WizardLM-13B-V1.2" target="_blank">HF Link</a> </sup>|  | <sup>7.06</sup> | <sup>89.17%</sup>	 | <sup>101.4% </sup>|<sup>36.6  pass@1</sup>|<sup> <a href="https://ai.meta.com/resources/models-and-libraries/llama-downloads/" target="_blank">Llama 2 License </a></sup> |
| <sup>WizardLM-13B-V1.1</sup> |<sup> 🤗 <a href="https://huggingface.co/WizardLM/WizardLM-13B-V1.1" target="_blank">HF Link</a> </sup> |  | <sup>6.76</sup>  |<sup>86.32%</sup>	 | <sup>99.3% </sup> |<sup>25.0  pass@1</sup>| <sup>Non-commercial</sup>|
| <sup>WizardLM-30B-V1.0</sup> | <sup>🤗 <a href="https://huggingface.co/WizardLM/WizardLM-30B-V1.0" target="_blank">HF Link</a></sup>  | | <sup>7.01</sup> |  |  <sup>97.8% </sup> | <sup>37.8  pass@1</sup>| <sup>Non-commercial</sup> |
| <sup>WizardLM-13B-V1.0</sup> | <sup>🤗 <a href="https://huggingface.co/WizardLM/WizardLM-13B-V1.0" target="_blank">HF Link</a> </sup> |  | <sup>6.35</sup> | <sup>75.31%</sup> |  <sup>89.1% </sup> |<sup> 24.0 pass@1 </sup> | <sup>Non-commercial</sup>|
| <sup>WizardLM-7B-V1.0 </sup>|  <sup>🤗 <a href="https://huggingface.co/WizardLM/WizardLM-7B-V1.0" target="_blank">HF Link</a> </sup> |<sup> 📃 <a href="https://arxiv.org/abs/2304.12244" target="_blank">[WizardLM]</a> </sup>|  |  |  <sup>78.0% </sup> |<sup>19.1 pass@1 </sup>|<sup> Non-commercial</sup>|

**Comparing WizardCoder with the Open-Source Models**

| Model            | HumanEval Pass@1 | MBPP Pass@1 |
|------------------|------------------|-------------|
| CodeGen-16B-Multi| 18.3             |20.9         |
| CodeGeeX         | 22.9             |24.4         |
| LLaMA-33B        | 21.7             |30.2         |
| LLaMA-65B        | 23.7             |37.7         |
| PaLM-540B        | 26.2             |36.8         |
| PaLM-Coder-540B  | 36.0             |47.0         |
| PaLM 2-S         | 37.6             |50.0         |
| CodeGen-16B-Mono | 29.3             |35.3         |
| Code-Cushman-001 | 33.5             |45.9         |
| StarCoder-15B    | 33.6             |43.6*        |
| InstructCodeT5+  | 35.0             |--           |
| WizardLM-30B  1.0| 37.8             |--           |
| WizardCoder-15B  1.0 | **57.3**     |**51.8**     |

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
According to the instructions of [Llama-X](https://github.com/AetherCortex/Llama-X), environment details are available.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

### Potential risks:
The content produced by any version of WizardCoder is influenced by uncontrollable variables such as randomness, and therefore, the accuracy of the output cannot be guaranteed by this project.

### Risk mitigation strategies:
Evol-Instruct method, fine-tuning.

### Efforts to address these challenges:
Focusing on improving the Evol-Instruct and hope to relieve existing weaknesses and issues in the the next version of WizardCoder.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
The accuracy of the output cannot be guaranteed.
If you are confused with the different scores of our model (57.3 and 59.8), please check the Notes.
If you are confused with the different scores of our model (57.3 and 59.8), please check the Notes.
❗**Note: In this study, we copy the scores for HumanEval and HumanEval+ from the [LLM-Humaneval-Benchmarks](https://github.com/my-other-github-account/llm-humaneval-benchmarks). Notably, all the mentioned models generate code solutions for each problem utilizing a **single attempt**, and the resulting pass rate percentage is reported. Our **WizardCoder** generates answers using greedy decoding and tests with the same [code](https://github.com/evalplus/evalplus).**
❗**Note: The reproduced result of StarCoder on MBPP.**
❗**Note: The "Comparing WizardCoder with the Open-Source Models" table conducts a comprehensive comparison of our **WizardCoder** with other models on the HumanEval and MBPP benchmarks. We adhere to the approach outlined in previous studies by generating **20 samples** for each problem to estimate the pass@1 score and evaluate with the same [code](https://github.com/openai/human-eval/tree/master). The scores of GPT4 and GPT3.5 reported by [OpenAI](https://openai.com/research/gpt-4) are 67.0 and 48.1 (maybe these are the early version GPT4&3.5).**

### Recommendations:
We welcome everyone to use your professional and difficult instructions to evaluate WizardCoder, and show us examples of poor performance and your suggestions in the [issue discussion](https://github.com/nlpxucan/WizardLM/issues) area.
Please try as many **real-world** and **challenging** code-related problems that you encounter in your work and life as possible.
Refer to Twitter account https://twitter.com/WizardLM_AI and HuggingFace Repo https://huggingface.co/WizardLM for new releases.

---

## Additional Information

## WizardCoder: Empowering Code Large Language Models with Evol-Instruct

<p style="font-size:28px;" align="center">
🏠 <a href="https://wizardlm.github.io/" target="_blank">Home Page</a> </p>
<p align="center">
<p align="center">
🤗 <a href="https://huggingface.co/WizardLM" target="_blank">HF Repo</a>  •🐱 <a href="https://github.com/nlpxucan/WizardLM" target="_blank">Github Repo</a> • 🐦 <a href="https://twitter.com/WizardLM_AI" target="_blank">Twitter</a> </p>
<p align="center">
 📃 <a href="https://arxiv.org/abs/2304.12244" target="_blank">[WizardLM]</a>  • 📃 <a href="https://arxiv.org/abs/2306.08568" target="_blank">[WizardCoder]</a>   • 📃 <a href="https://arxiv.org/abs/2308.09583" target="_blank">[WizardMath]</a>  <br>
</p>
<p align="center">
    👋 Join our <a href="https://discord.gg/VZjjHtWrKs" target="_blank">Discord</a>
</p>

## News

[2024/01/04] 🔥 We released **WizardCoder-33B-V1.1**  trained from deepseek-coder-33b-base, the **SOTA OSS Code LLM** on [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html), achieves **79.9 pass@1** on HumanEval, **73.2 pass@1** on HumanEval-Plus, **78.9 pass@1** on MBPP, and **66.9 pass@1** on MBPP-Plus.

[2024/01/04] 🔥 **WizardCoder-33B-V1.1** outperforms **ChatGPT 3.5**, **Gemini Pro**, and **DeepSeek-Coder-33B-instruct** on HumanEval and HumanEval-Plus pass@1.

[2024/01/04] 🔥 **WizardCoder-33B-V1.1** is comparable with **ChatGPT 3.5**, and surpasses **Gemini Pro** on MBPP and MBPP-Plus pass@1.

<p align="center" width="100%">
<a ><img src="https://raw.githubusercontent.com/nlpxucan/WizardLM/main/WizardCoder/imgs/compare_sota.png" alt="WizardCoder" style="width: 96%; min-width: 300px; display: block; margin: auto;"></a>
</p>

- 🔥 [08/11/2023] We release **WizardMath** Models.
- 🔥 Our **WizardMath-70B-V1.0** model slightly outperforms some closed-source LLMs on the GSM8K, including **ChatGPT 3.5**, **Claude Instant 1** and **PaLM 2 540B**.
- 🔥 Our **WizardMath-70B-V1.0** model achieves  **81.6 pass@1** on the [GSM8k Benchmarks](https://github.com/openai/grade-school-math), which is **24.8** points higher than the SOTA open-source LLM.
- 🔥 Our **WizardMath-70B-V1.0** model achieves  **22.7 pass@1** on the [MATH Benchmarks](https://github.com/hendrycks/math), which is **9.2** points higher than the SOTA open-source LLM.


# WizardCoder: Empowering Code Large Language Models with Evol-Instruct


To develop our WizardCoder model, we begin by adapting the Evol-Instruct method specifically for coding tasks. This involves tailoring the prompt to the domain of code-related instructions. Subsequently, we fine-tune the Code LLM, StarCoder, utilizing the newly created instruction-following training set.

## News

- 🔥 Our **WizardCoder-15B-v1.0** model achieves the **57.3 pass@1** on the [HumanEval Benchmarks](https://github.com/openai/human-eval), which is **22.3** points higher than the SOTA open-source Code LLMs.
- 🔥 We released **WizardCoder-15B-v1.0** trained with **78k** evolved code instructions. Please checkout the [Model Weights](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0), and [Paper]().
- &#x1F4E3; Please refer to our Twitter account https://twitter.com/WizardLM_AI and HuggingFace Repo https://huggingface.co/WizardLM . We will use them to announce any new release at the 1st time.


## Comparing WizardCoder with the Closed-Source Models.


🔥 The following figure shows that our **WizardCoder attains the third position in this benchmark**, surpassing Claude-Plus (59.8 vs. 53.0) and Bard (59.8 vs. 44.5). Notably, our model exhibits a substantially smaller size compared to these models.

<p align="center" width="100%">
<a ><img src="https://raw.githubusercontent.com/nlpxucan/WizardLM/main/WizardCoder/imgs/pass1.png" alt="WizardCoder" style="width: 86%; min-width: 300px; display: block; margin: auto;"></a>
</p>

## Comparing WizardCoder with the Open-Source Models.

The "Comparing WizardCoder with the Open-Source Models" table clearly demonstrates that our **WizardCoder** exhibits a substantial performance advantage over all the open-source models. ❗**If you are confused with the different scores of our model (57.3 and 59.8), please check the Notes.**


## Call for Feedbacks
We welcome everyone to use your professional and difficult instructions to evaluate WizardCoder, and show us examples of poor performance and your suggestions in the [issue discussion](https://github.com/nlpxucan/WizardLM/issues) area. We are focusing on improving the Evol-Instruct now and hope to relieve existing weaknesses and issues in the the next version of WizardCoder. After that, we will open the code and pipeline of up-to-date Evol-Instruct algorithm and work with you together to improve it.


## Contents

1. [Online Demo](#online-demo)

2. [Fine-tuning](#fine-tuning)

3. [Inference](#inference)

4. [Evaluation](#evaluation)

5. [Citation](#citation)

6. [Disclaimer](#disclaimer)

## Evaluation

We provide the evaluation script on HumanEval for WizardCoder.

1. According to the instructions of [HumanEval](https://github.com/openai/human-eval), install the environment.
2. Run the following script to generate the answer.
```bash
model="/path/to/your/model"
temp=0.2
max_len=2048
pred_num=200
num_seqs_per_iter=2

output_path=preds/T${temp}_N${pred_num}

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# 164 problems, 21 per GPU if GPU=8
index=0
gpu_num=8
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * 21))
  end_index=$(((i + 1) * 21))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python humaneval_gen.py --model ${model} \
      --start_index ${start_index} --end_index ${end_index} --temperature ${temp} \
      --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path}
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done
```
3. Run the post processing code `src/process_humaneval.py` to collect the code completions from all answer files.
```bash
output_path=preds/T${temp}_N${pred_num}

echo 'Output path: '$output_path
python process_humaneval.py --path ${output_path} --out_path ${output_path}.jsonl --add_prompt

evaluate_functional_correctness ${output_path}.jsonl
```

## Citation

Please cite the repo if you use the data, method or code in this repo.

## Disclaimer

WizardCoder model follows the same license as StarCoder. The content produced by any version of WizardCoder is influenced by uncontrollable variables such as randomness, and therefore, the accuracy of the output cannot be guaranteed by this project. This project does not accept any legal liability for the content of the model output, nor does it assume responsibility for any losses incurred due to the use of associated resources and output results.