## DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
- DeepSeek-V2 is a strong Mixture-of-Experts (MoE) language model.
- DeepSeek-V2 is characterized by economical training and efficient inference.
- DeepSeek-V2 comprises 236 billion total parameters.
- 21 billion parameters are activated for each token.
- DeepSeek-V2 achieves stronger performance compared to DeepSeek 67B.
- DeepSeek-V2 saves 42.5% of training costs.
- DeepSeek-V2 reduces the KV cache by 93.3%.
- DeepSeek-V2 boosts the maximum generation throughput to 5.76 times.
- DeepSeek-V2 was pretrained on a diverse and high-quality corpus comprising 8.1 trillion tokens.
- The pretraining was followed by Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL).
- Evaluation results validate the effectiveness of the approach.
- DeepSeek-V2 achieves remarkable performance on standard benchmarks and open-ended generation evaluation.

## 2. Model Downloads
- Table titled: Model Downloads
   | **Model** | **Context Length** | **Download** |
   | :------------: | :------------: | :------------: |
   | DeepSeek-V2   | 128k   | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V2)   |
   | DeepSeek-V2-Chat (RL)   | 128k   | [🤗 HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat)   |
- The open-source code currently experiences slower performance than the internal codebase when running on GPUs with Huggingface.
- A dedicated vllm solution is offered to optimize performance for running the model effectively.

## 3. Evaluation Results
### Base Model
#### Standard Benchmark
- Table titled: Standard Benchmark
   | **Benchmark** | **Domain** | **LLaMA3 70B** | **Mixtral 8x22B** | **DeepSeek-V1 (Dense-67B)** | **DeepSeek-V2 (MoE-236B)** |
   |:-----------:|:--------:|:------------:|:---------------:|:-------------------------:|:------------------------:|
   | **MMLU** | English | 78.9 | 77.6 | 71.3 | 78.5 |
   | **BBH** | English | 81.0 | 78.9 | 68.7 | 78.9 |
   | **C-Eval** | Chinese | 67.5 | 58.6 | 66.1 | 81.7 |
   | **CMMLU** | Chinese | 69.3 | 60.0 | 70.8 | 84.0 |
   | **HumanEval** | Code | 48.2 | 53.1 | 45.1 | 48.8 |
   | **MBPP** | Code | 68.6 | 64.2 | 57.4 | 66.6 |
   | **GSM8K** | Math | 83.0 | 80.3 | 63.4 | 79.2 |
   | **Math** | Math | 42.2 | 42.5 | 18.7 | 43.6 |
- For more evaluation details, such as few-shot settings and prompts, please check the paper.
#### Context Window
- Image titled: Context Window Evaluation
   ![Context Window Evaluation](https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/niah.png?raw=true)
- DeepSeek-V2 performs well across all context window lengths up to **128K**.

### Chat Model
#### Standard Benchmark
- Table titled: Chat Model Standard Benchmark
   | Benchmark | Domain         | QWen1.5 72B Chat | Mixtral 8x22B | LLaMA3 70B Instruct | DeepSeek-V1 Chat (SFT) | DeepSeek-V2 Chat (SFT) | DeepSeek-V2 Chat (RL) |
   |:-----------:|:----------------:|:------------------:|:---------------:|:---------------------:|:-------------:|:-----------------------:|:----------------------:|
   | **MMLU**      | English        | 76.2             | 77.8          | 80.3                | 71.1        | 78.4                 | 77.8                 |
   | **BBH**       | English        | 65.9             | 78.4          | 80.1                | 71.7        | 81.3                 | 79.7                 |
   | **C-Eval**    | Chinese        | 82.2             | 60.0          | 67.9                | 65.2        | 80.9                 | 78.0                 |
   | **CMMLU**     | Chinese        | 82.9             | 61.0          | 70.7                | 67.8        | 82.4                 | 81.6                 |
   | **HumanEval** | Code           | 68.9             | 75.0          | 76.2                | 73.8        | 76.8                 | 81.1                 |
   | **MBPP**      | Code           | 52.2             | 64.4          | 69.8                | 61.4        | 70.4                 | 72.0                 |
   |   **LiveCodeBench  (0901-0401)**     | Code           | 18.8             | 25.0          | 30.5                | 18.3        | 28.7                 | 32.5                 |
   | **GSM8K**     | Math           | 81.9             | 87.9          | 93.2                | 84.1        | 90.8                 | 92.2                 |
   | **Math**      | Math           | 40.6             | 49.8          | 48.5                | 32.6        | 52.7                 | 53.9                 |
#### English Open Ended Generation Evaluation
- DeepSeek-V2-Chat-RL shows competitive performance on English conversation generation.
- Image titled: English Open Ended Generation Evaluation
   ![English Open Ended Generation Evaluation](https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/mtbench.png?raw=true)
#### Chinese Open Ended Generation Evaluation
- Table titled: Chinese Open Ended Generation Evaluation
   | **模型** | **开源/闭源** | **总分** | **中文推理** | **中文语言** |
   | :---: | :---: | :---: | :---: | :---: |
   | gpt-4-1106-preview | 闭源 | 8.01 | 7.73 | 8.29 |
   | DeepSeek-V2 Chat (RL) | 开源 | 7.91 | 7.45 | 8.35 |
   | erniebot-4.0-202404 (文心一言) | 闭源 | 7.89 | 7.61 | 8.17 |
   | DeepSeek-V2 Chat (SFT) | 开源 | 7.74 | 7.30 | 8.17 |
   | gpt-4-0613 | 闭源 | 7.53 | 7.47 | 7.59 |
   | erniebot-4.0-202312 (文心一言) | 闭源 | 7.36 | 6.84 | 7.88 |
   | moonshot-v1-32k-202404 (月之暗面) | 闭源 | 7.22 | 6.42 | 8.02 |
   | Qwen1.5-72B-Chat (通义千问) | 开源 | 7.19 | 6.45 | 7.93 |
   | DeepSeek-67B-Chat | 开源 | 6.43 | 5.75 | 7.11 |
   | Yi-34B-Chat (零一万物) | 开源 | 6.12 | 4.86 | 7.38 |
   | gpt-3.5-turbo-0613 | 闭源 | 6.08 | 5.35 | 6.71 |
#### Coding Benchmarks
- DeepSeek-V2 demonstrates considerable proficiency in LiveCodeBench.
- DeepSeek-V2 achieves a Pass@1 score that surpasses several other sophisticated models.
- Image titled: Coding Benchmarks
   ![Coding Benchmarks](https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/code_benchmarks.png?raw=true)

## 4. Model Architecture
- DeepSeek-V2 adopts innovative architectures for economical training and efficient inference.
- For attention, DeepSeek-V2 uses MLA (Multi-head Latent Attention).
- MLA utilizes low-rank key-value union compression.
- MLA eliminates the bottleneck of inference-time key-value cache.
- MLA supports efficient inference.
- For Feed-Forward Networks (FFNs), DeepSeek-V2 adopts DeepSeekMoE architecture.
- DeepSeekMoE architecture enables training stronger models at lower costs.
- Image titled: Model Architecture
   ![Model Architecture](https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/architecture.png?raw=true)

## 5. Chat Website
- You can chat with DeepSeek-V2 on DeepSeek's official website: [chat.deepseek.com](https://chat.deepseek.com/sign_in)

## 6. API Platform
- OpenAI-Compatible API is provided at DeepSeek Platform: [platform.deepseek.com](https://platform.deepseek.com/).
- Sign up for over millions of free tokens.
- Pay-as-you-go option is available at an unbeatable price.
- Image titled: API Platform Pricing
   ![API Platform Pricing](https://github.com/deepseek-ai/DeepSeek-V2/blob/main/figures/model_price.png?raw=true)

## 7. How to run locally
- To utilize DeepSeek-V2 in BF16 format for inference, 80GB*8 GPUs are required.
### Inference with Huggingface's Transformers
- Code snippet for text completion:
   ```python
   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

   model_name = "deepseek-ai/DeepSeek-V2"
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   max_memory = {i: "75GB" for i in range(8)}
   model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="sequential", torch_dtype=torch.bfloat16, max_memory=max_memory, attn_implementation="eager")
   model.generation_config = GenerationConfig.from_pretrained(model_name)
   model.generation_config.pad_token_id = model.generation_config.eos_token_id

   text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
   inputs = tokenizer(text, return_tensors="pt")
   outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

   result = tokenizer.decode(outputs[0], skip_special_tokens=True)
   print(result)
   ```
- Code snippet for chat completion:
   ```python
   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

   model_name = "deepseek-ai/DeepSeek-V2-Chat"
   tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
   max_memory = {i: "75GB" for i in range(8)}
   model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, device_map="sequential", torch_dtype=torch.bfloat16, max_memory=max_memory, attn_implementation="eager")
   model.generation_config = GenerationConfig.from_pretrained(model_name)
   model.generation_config.pad_token_id = model.generation_config.eos_token_id

   messages = [
       {"role": "user", "content": "Write a piece of quicksort code in C++"}
   ]
   input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
   outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)

   result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
   print(result)
   ```
- The complete chat template can be found within `tokenizer_config.json` located in the huggingface model repository.
- Example of chat template:
   ```bash
   <｜begin▁of▁sentence｜>User: {user_message_1}

   Assistant: {assistant_message_1}<｜end▁of▁sentence｜>User: {user_message_2}

   Assistant:
   ```
- Optional system message can be added:
   ```bash
   <｜begin▁of▁sentence｜>{system_message}

   User: {user_message_1}

   Assistant: {assistant_message_1}<｜end▁of▁sentence｜>User: {user_message_2}

   Assistant:
   ```
### Inference with vLLM (recommended)
- To utilize [vLLM](https://github.com/vllm-project/vllm) for model inference, merge this Pull Request into your vLLM codebase: https://github.com/vllm-project/vllm/pull/4650.
- Code snippet for vLLM inference:
   ```python
   from transformers import AutoTokenizer
   from vllm import LLM, SamplingParams

   max_model_len, tp_size = 8192, 8
   model_name = "deepseek-ai/DeepSeek-V2-Chat"
   tokenizer = AutoTokenizer.from_pretrained(model_name)
   llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
   sampling_params = SamplingParams(temperature=0.3, max_tokens=256, stop_token_ids=[tokenizer.eos_token_id])

   messages_list = [
       [{"role": "user", "content": "Who are you?"}],
       [{"role": "user", "content": "Translate the following content into Chinese directly: DeepSeek-V2 adopts innovative architectures to guarantee economical training and efficient inference."}],
       [{"role": "user", "content": "Write a piece of quicksort code in C++."}],
   ]

   prompt_token_ids = [tokenizer.apply_chat_template(messages, add_generation_prompt=True) for messages in messages_list]

   outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

   generated_text = [output.outputs[0].text for output in outputs]
   print(generated_text)
   ```

## 8. License
- This code repository is licensed under [the MIT License](LICENSE-CODE).
- The use of DeepSeek-V2 Base/Chat models is subject to [the Model License](LICENSE-MODEL).
- DeepSeek-V2 series (including Base and Chat) supports commercial use.

## 9. Citation
- Citation for DeepSeek-V2:
   ```
   @misc{deepseekv2,
         title={DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model}, 
         author={DeepSeek-AI},
         year={2024},
         eprint={2405.04434},
         archivePrefix={arXiv},
         primaryClass={cs.CL}
   }
   ```

## 10. Contact
- For questions, please raise an issue or contact at [service@deepseek.com](service@deepseek.com).