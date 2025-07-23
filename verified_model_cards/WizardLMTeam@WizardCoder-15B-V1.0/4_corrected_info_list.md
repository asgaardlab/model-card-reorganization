## WizardCoder: Empowering Code Large Language Models with Evol-Instruct
- 🏠 <a href="https://wizardlm.github.io/" target="_blank">Home Page</a> 
- 🤗 <a href="https://huggingface.co/WizardLM" target="_blank">HF Repo</a>
- 🐱 <a href="https://github.com/nlpxucan/WizardLM" target="_blank">Github Repo</a> 
- 🐦 <a href="https://twitter.com/WizardLM_AI" target="_blank">Twitter</a> 
- 📃 <a href="https://arxiv.org/abs/2304.12244" target="_blank">[WizardLM]</a>  
- 📃 <a href="https://arxiv.org/abs/2306.08568" target="_blank">[WizardCoder]</a>
- 📃 <a href="https://arxiv.org/abs/2308.09583" target="_blank">[WizardMath]</a>
- 👋 Join our <a href="https://discord.gg/VZjjHtWrKs" target="_blank">Discord</a>

## News
- On [2024/01/04], **WizardCoder-33B-V1.1** was released.
- **WizardCoder-33B-V1.1** is trained from deepseek-coder-33b-base.
- **WizardCoder-33B-V1.1** is the **SOTA OSS Code LLM** on the [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html).
- **WizardCoder-33B-V1.1** achieves **79.9 pass@1** on HumanEval.
- **WizardCoder-33B-V1.1** achieves **73.2 pass@1** on HumanEval-Plus.
- **WizardCoder-33B-V1.1** achieves **78.9 pass@1** on MBPP.
- **WizardCoder-33B-V1.1** achieves **66.9 pass@1** on MBPP-Plus.
- **WizardCoder-33B-V1.1** outperforms **ChatGPT 3.5**, **Gemini Pro**, and **DeepSeek-Coder-33B-instruct** on HumanEval and HumanEval-Plus pass@1.
- **WizardCoder-33B-V1.1** is comparable with **ChatGPT 3.5** and surpasses **Gemini Pro** on MBPP and MBPP-Plus pass@1.

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

- An image comparing WizardCoder is available: ![WizardCoder](https://raw.githubusercontent.com/nlpxucan/WizardLM/main/WizardCoder/imgs/compare_sota.png).
- On [08/11/2023], **WizardMath** Models were released.
- **WizardMath-70B-V1.0** slightly outperforms some closed-source LLMs on the GSM8K.
- **WizardMath-70B-V1.0** outperforms **ChatGPT 3.5**, **Claude Instant 1**, and **PaLM 2 540B**.
- **WizardMath-70B-V1.0** achieves **81.6 pass@1** on the [GSM8k Benchmarks](https://github.com/openai/grade-school-math) which is 24.8 points higher than the SOTA open-source LLM.
- **WizardMath-70B-V1.0** achieves **22.7 pass@1** on the [MATH Benchmarks](https://github.com/hendrycks/math) which is 9.2 points higher than the SOTA open-source LLM.

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

## WizardCoder: Empowering Code Large Language Models with Evol-Instruct
- The WizardCoder model is developed by adapting the Evol-Instruct method for coding tasks.
- The prompt is tailored to code-related instructions.
- The Code LLM, StarCoder, is fine-tuned using the instruction-following training set.

## News
- **WizardCoder-15B-v1.0** achieves **57.3 pass@1** on the [HumanEval Benchmarks](https://github.com/openai/human-eval) which is 22.3 points higher than the SOTA open-source Code LLMs..
- **WizardCoder-15B-v1.0** is trained with **78k** evolved code instructions.
- Model weights for **WizardCoder-15B-v1.0** can be found [here](https://huggingface.co/WizardLM/WizardCoder-15B-V1.0).
- The Twitter account for updates is [here](https://twitter.com/WizardLM_AI).
- The HuggingFace Repo for updates is [here](https://huggingface.co/WizardLM).

## Comparing WizardCoder with the Closed-Source Models
- The figure shows that **WizardCoder** attains the third position in the benchmark.
- **WizardCoder** surpasses Claude-Plus (59.8 vs. 53.0) and Bard (59.8 vs. 44.5).
- **WizardCoder** exhibits a smaller size compared to these models.
- An image comparing WizardCoder is available: ![WizardCoder](https://raw.githubusercontent.com/nlpxucan/WizardLM/main/WizardCoder/imgs/pass1.png).
- The scores for HumanEval and HumanEval+ are copied from the [LLM-Humaneval-Benchmarks](https://github.com/my-other-github-account/llm-humaneval-benchmarks).
- All models generate code solutions for each problem using a **single attempt**.
- The resulting pass rate percentage is reported.
- **WizardCoder** generates answers using greedy decoding and tests with the same [code](https://github.com/evalplus/evalplus).

## Comparing WizardCoder with the Open-Source Models
- The table demonstrates that **WizardCoder** has a performance advantage over all open-source models.
- If confused with the different scores of **WizardCoder** (57.3 and 59.8), please check the Notes.

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

- Note: The reproduced result of StarCoder on MBPP.
- Note: The above table conducts a comprehensive comparison of our **WizardCoder** with other models on the HumanEval and MBPP benchmarks. We adhere to the approach outlined in previous studies by generating **20 samples** for each problem to estimate the pass@1 score and evaluate with the same [code](https://github.com/openai/human-eval/tree/master). The scores of GPT4 and GPT3.5 reported by [OpenAI](https://openai.com/research/gpt-4) are 67.0 and 48.1 (maybe these are the early version GPT4&3.5).

## Call for Feedbacks
- Feedback is welcomed for evaluating WizardCoder.
- Users are encouraged to show examples of poor performance and suggestions in the [issue discussion](https://github.com/nlpxucan/WizardLM/issues).
- The focus is on improving Evol-Instruct for the next version of WizardCoder.
- The code and pipeline of the updated Evol-Instruct algorithm will be opened for collaboration.

## Contents
1. [Online Demo](#online-demo)
2. [Fine-tuning](#fine-tuning)
3. [Inference](#inference)
4. [Evaluation](#evaluation)
5. [Citation](#citation)
6. [Disclaimer](#disclaimer)

## Online Demo
- The latest models will be provided for users to try.
- If you find a link is not working, please try another one.
- Users are encouraged to try real-world and challenging code-related problems.
- Feedback will help evolve the models.

## Fine-tuning
- WizardCoder is fine-tuned using the modified code `train.py` from [Llama-X](https://github.com/AetherCortex/Llama-X).
- StarCoder-15B is fine-tuned with specific hyperparameters.

| Hyperparameter | StarCoder-15B |
|----------------|---------------|
| Batch size     | 512           |
| Learning rate  | 2e-5          |
| Epochs         | 3             |
| Max length     | 2048          |
| Warmup step    | 30            |
| LR scheduler   | cosine        |

- Steps to reproduce the fine-tuning of WizardCoder:
1. Install the environment and download the training code according to [Llama-X](https://github.com/AetherCortex/Llama-X) (Note: `deepspeed==0.9.2` and `transformers==4.29.2`).
2. Replace `train.py` with `train_wizardcoder.py` in the repo (`src/train_wizardcoder.py`).
3. Login to Huggingface:
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

## Inference
- A decoding script for WizardCoder is provided.
- The script reads an input file and generates responses for each sample.
- The responses are consolidated into an output file.
- Specify `base_model`, `input_data_path`, and `output_data_path` in `src\inference_wizardcoder.py` to set the decoding model, path of input file and path of output file.

```bash
pip install jsonlines
```

- The decoding command is:
   ```
   python src\inference_wizardcoder.py \
       --base_model "/your/path/to/ckpt" \
       --input_data_path "/your/path/to/input/data.jsonl" \
       --output_data_path "/your/path/to/output/result.jsonl"
   ```
- The format of `data.jsonl` should be:
   ```
   {"idx": 11, "Instruction": "Write a Python code to count 1 to 10."}
   {"idx": 12, "Instruction": "Write a Java code to sum 1 to 10."}
   ```
- The prompt for WizardCoder in `src\inference_wizardcoder.py` is:
   ```
   Below is an instruction that describes a task. Write a response that appropriately completes the request.

   ### Instruction:
   {instruction}

   ### Response:
   ```

## Evaluation
- An evaluation script for HumanEval is provided for WizardCoder.
- Steps to evaluate WizardCoder:
1. Install the environment according to [HumanEval](https://github.com/openai/human-eval).
2. Run the script to generate answers:
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
3. Run the post-processing code `src/process_humaneval.py` to collect code completions from all answer files:
   ```bash
   output_path=preds/T${temp}_N${pred_num}

   echo 'Output path: '$output_path
   python process_humaneval.py --path ${output_path} --out_path ${output_path}.jsonl --add_prompt

   evaluate_functional_correctness ${output_path}.jsonl
   ```

## Citation
- Please cite the repo if you use the data, method, or code in this repo.
```
@article{luo2023wizardcoder,
  title={WizardCoder: Empowering Code Large Language Models with Evol-Instruct},
  author={Luo, Ziyang and Xu, Can and Zhao, Pu and Sun, Qingfeng and Geng, Xiubo and Hu, Wenxiang and Tao, Chongyang and Ma, Jing and Lin, Qingwei and Jiang, Daxin},
  journal={arXiv preprint arXiv:2306.08568},
  year={2023}
}
```

## Disclaimer
- WizardCoder follows the same license as StarCoder.
- The content produced by WizardCoder is influenced by uncontrollable variables such as randomness.
- The accuracy of the output cannot be guaranteed by this project.
- This project does not accept legal liability for the content of the model output.
- The project does not assume responsibility for any losses incurred due to the use of associated resources and output results.