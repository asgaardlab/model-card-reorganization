## Model Summary
- 🎉 **Phi-3.5**: [[mini-instruct]](https://huggingface.co/microsoft/Phi-3.5-mini-instruct); [[MoE-instruct]](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct); [[vision-instruct]](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)
- The Phi-3-Mini-128K-Instruct is a 3.8 billion-parameter model.
- It is a lightweight, state-of-the-art open model.
- The model was trained using the Phi-3 datasets.
- The Phi-3 dataset includes both synthetic data and filtered publicly available website data.
- The dataset emphasizes high-quality and reasoning-dense properties.
- The model belongs to the Phi-3 family.
- The Mini version has two variants: [4K](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) and [128K](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct).
- The context length supported by the model is in tokens.
- After initial training, the model underwent a post-training process.
- The post-training process involved supervised fine-tuning and direct preference optimization.
- The enhancements aimed to improve the model's ability to follow instructions and adhere to safety measures.
- The model demonstrated robust and state-of-the-art performance against benchmarks.
- Benchmarks tested include common sense, language understanding, mathematics, coding, long-term context, and logical reasoning.
- The performance was compared among models with fewer than 13 billion parameters.

## Resources and Technical Documentation
- 🏡 [Phi-3 Portal](https://azure.microsoft.com/en-us/products/phi-3)
- 📰 [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024)
- 📖 [Phi-3 Technical Report](https://aka.ms/phi3-tech-report)
- 🛠️ [Phi-3 on Azure AI Studio](https://aka.ms/phi3-azure-ai)
- 👩‍🍳 [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook)
- 🖥️ [Try It](https://aka.ms/try-phi3)

## Context Variants Table
- Table titled: Context Variants
   |         | Short Context | Long Context |
   | :-      | :-            | :-           |
   | Mini    | 4K [[HF]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct); [[ONNX]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx); [[GGUF]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct); [[ONNX]](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx) |
   | Small   | 8K [[HF]](https://huggingface.co/microsoft/Phi-3-small-8k-instruct); [[ONNX]](https://huggingface.co/microsoft/Phi-3-small-8k-instruct-onnx-cuda) | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-small-128k-instruct); [[ONNX]](https://huggingface.co/microsoft/Phi-3-small-128k-instruct-onnx-cuda) |
   | Medium  | 4K [[HF]](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct); [[ONNX]](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-cuda) | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct); [[ONNX]](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct-onnx-cuda) |
   | Vision  |  | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct); [[ONNX]](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct-onnx-cuda) |

## Intended Uses
- **Primary use cases**
   - The model is intended for commercial and research use in English.
   - It is suitable for applications requiring:
     1) Memory/compute constrained environments.
     2) Latency bound scenarios.
     3) Strong reasoning (especially code, math, and logic).
   - The model is designed to accelerate research on language and multimodal models.
   - It serves as a building block for generative AI powered features.

- **Use case considerations**
   - The models are not specifically designed or evaluated for all downstream purposes.
   - Developers should consider common limitations of language models when selecting use cases.
   - Developers should evaluate and mitigate for accuracy, safety, and fairness before using the model in specific downstream use cases.
   - Developers should adhere to applicable laws or regulations relevant to their use case.
   - Nothing in this Model Card should be interpreted as a restriction or modification to the model's license.

## Release Notes
- This is an update over the original instruction-tuned Phi-3-mini release.
- The update is based on valuable customer feedback.
- The model used additional post-training data.
- The update led to substantial gains in long-context understanding, instruction following, and structured output.
- Multi-turn conversation quality has improved.
- The model explicitly supports the <|system|> tag.
- Reasoning capability has significantly improved.
- Users are encouraged to test the model in their specific AI applications.
- Feedback from the community is welcomed.

## Improvement Tables
- Table titled: Improvements on Instruction Following, Structure Output, Reasoning, and Long-Context Understanding
   | Benchmarks | Original | June 2024 Update |
   | :-         | :-       | :-               |
   | Instruction Extra Hard | 5.7 | 5.9 |
   | Instruction Hard | 5.0 | 5.2 |
   | JSON Structure Output | 1.9 | 60.1 |
   | XML Structure Output | 47.8 | 52.9 |
   | GPQA | 25.9 | 29.7 |
   | MMLU | 68.1 | 69.7 |
   | **Average** | **25.7** | **37.3** |

- Table titled: RULER: a retrieval-based benchmark for long context understanding
   | Model             | 4K   | 8K   | 16K  | 32K  | 64K  | 128K | Average |
   | :-------------------| :------| :------| :------| :------| :------| :------| :---------|
   | Original          | 86.7 | 78.1 | 75.6 | 70.3 | 58.9 | 43.3 | **68.8**    |
   | June 2024 Update  | 92.4 | 91.1 | 90.8 | 87.9 | 79.8 | 65.6 | **84.6**    |

- Table titled: RepoQA: a benchmark for long context code understanding
   | Model             | Python | C++ | Rust | Java | TypeScript | Average |
   | :-------------------| :--------| :-----| :------| :------| :------------| :---------|
   | Original          | 27     | 29  | 40   | 33   | 33         | **32.4**    |
   | June 2024 Update  | 85     | 63  | 72   | 93   | 72         | **77**      |

- Notes: Users can check out the previous version using the git commit id **bb5bf1e4001277a606e11debca0ef80323e5f824**.
- Users are invited to experiment with various approaches for model conversion, e.g., GGUF and other formats.

## How to Use
- Phi-3 Mini-128K-Instruct has been integrated into the development version (4.41.3) of `transformers`.
- Until the official version is released through `pip`, users should:
   - When loading the model, ensure that `trust_remote_code=True` is passed as an argument of the `from_pretrained()` function.
   - Update local `transformers` to the development version: `pip uninstall -y transformers && pip install git+https://github.com/huggingface/transformers`.
   - Verify the current `transformers` version with: `pip list | grep transformers`.

- Examples of required packages:
   ```
   flash_attn==2.5.8
   torch==2.3.1
   accelerate==0.31.0
   transformers==4.41.2
   ```

- Phi-3 Mini-128K-Instruct is also available in [Azure AI Studio](https://aka.ms/try-phi3).

### Tokenizer
- Phi-3 Mini-128K-Instruct supports a vocabulary size of up to `32064` tokens.
- The [tokenizer files](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/blob/main/added_tokens.json) provide placeholder tokens for downstream fine-tuning.
- The tokenizer can be extended up to the model's vocabulary size.

### Chat Format
- The model is best suited for prompts using the chat format.
- Example prompt format:
   ```markdown
   <|system|>
   You are a helpful assistant.<|end|>
   <|user|>
   Question?<|end|>
   <|assistant|>
   ```
- Example usage:
   ```markdown
   <|system|>
   You are a helpful assistant.<|end|>
   <|user|>
   How to explain Internet for a medieval knight?<|end|>
   <|assistant|>
   ```

- Few-shot prompt example:
   ```markdown
   <|system|>
   You are a helpful travel assistant.<|end|>
   <|user|>
   I am going to Paris, what should I see?<|end|>
   <|assistant|>
   Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:\n\n1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.\n2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.\n3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.\n\nThese are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world."<|end|>
   <|user|>
   What is so great about #1?<|end|>
   <|assistant|>
   ```

### Sample inference code
- Code snippet to quickly start running the model on a GPU:
   ```python
   import torch 
   from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 

   torch.random.manual_seed(0) 
   model = AutoModelForCausalLM.from_pretrained( 
       "microsoft/Phi-3-mini-128k-instruct",  
       device_map="cuda",  
       torch_dtype="auto",  
       trust_remote_code=True,  
   ) 

   tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct") 

   messages = [ 
       {"role": "system", "content": "You are a helpful AI assistant."}, 
       {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"}, 
       {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."}, 
       {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"}, 
   ] 

   pipe = pipeline( 
       "text-generation", 
       model=model, 
       tokenizer=tokenizer, 
   ) 

   generation_args = { 
       "max_new_tokens": 500, 
       "return_full_text": False, 
       "temperature": 0.0, 
       "do_sample": False, 
   } 

   output = pipe(messages, **generation_args) 
   print(output[0]['generated_text']) 
   ```
- Notes: To use flash attention, call _AutoModelForCausalLM.from_pretrained()_ with _attn_implementation="flash_attention_2"_.

## Responsible AI Considerations
- The Phi series models can potentially behave in ways that are unfair, unreliable, or offensive.
- Quality of Service: The models are trained primarily on English text.
- Languages other than English may experience worse performance.
- English language varieties with less representation may experience worse performance than standard American English.
- Representation of Harms & Perpetuation of Stereotypes: The models can over- or under-represent groups of people.
- Inappropriate or Offensive Content: The models may produce inappropriate or offensive content.
- Information Reliability: Language models can generate nonsensical or inaccurate content.
- Limited Scope for Code: The majority of training data is based in Python and common packages.
- Developers should apply responsible AI best practices and ensure compliance with relevant laws and regulations.

## Training
### Model
- Architecture: Phi-3 Mini-128K-Instruct has 3.8B parameters and is a dense decoder-only Transformer model.
- The model is fine-tuned with Supervised fine-tuning (SFT) and Direct Preference Optimization (DPO).
- Inputs: Text, best suited for prompts using chat format.
- Context length: 128K tokens.
- GPUs: 512 H100-80G.
- Training time: 10 days.
- Training data: 4.9T tokens.
- Outputs: Generated text in response to the input.
- Dates: Models were trained between May and June 2024.
- Status: Static model trained on an offline dataset with cutoff date October 2023.
- Release dates: June 2024.

### Datasets
- Training data includes a wide variety of sources totaling 4.9 trillion tokens.
- The data is a combination of:
   1) Publicly available documents filtered for quality.
   2) Newly created synthetic data for teaching math, coding, common sense reasoning, and general knowledge.
   3) High-quality chat format supervised data covering various topics.
- The focus is on the quality of data to improve reasoning ability.
- More details about data can be found in the [Phi-3 Technical Report](https://aka.ms/phi3-tech-report).

### Fine-tuning
- A basic example of multi-GPUs supervised fine-tuning (SFT) with TRL and Accelerate modules is provided [here](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/resolve/main/sample_finetune.py).

## Benchmarks
- The results are reported under completion format for Phi-3-Mini-128K-Instruct on standard open-source benchmarks.
- The model's reasoning ability is compared to Mistral-7b-v0.1, Mixtral-8x7b, Gemma 7B, Llama-3-8B-Instruct, and GPT-3.5.
- All reported numbers are produced with the same pipeline for comparability.
- The number of k–shot examples is listed per-benchmark.

- Table titled: Benchmark Results
   | Category | Benchmark | Phi-3-Mini-128K-Ins | Gemma-7B | Mistral-7B | Mixtral-8x7B | Llama-3-8B-Ins | GPT3.5-Turbo-1106 |
   | :----------| :-----------| :---------------------| :----------| :------------| :--------------| :----------------| :-------------------|
   | Popular aggregated benchmark | AGI Eval <br>5-shot| 39.5 | 42.1 | 35.1 | 45.2 | 42 | 48.4 |
   | | MMLU <br>5-shot | 69.7 | 63.6 | 61.7 | 70.5 | 66.5 | 71.4 |
   | | BigBench Hard <br>3-shot | 72.1 | 59.6 | 57.3 | 69.7 | 51.5 | 68.3 |
   | Language Understanding | ANLI <br>7-shot | 52.3 | 48.7 | 47.1 | 55.2 | 57.3 | 58.1 |
   | | HellaSwag <br>5-shot | 70.5 | 49.8 | 58.5 | 70.4 | 71.1 | 78.8 |
   | Reasoning | ARC Challenge <br>10-shot | 85.5 | 78.3 | 78.6 | 87.3 | 82.8 | 87.4 |
   | | BoolQ <br>0-shot | 77.1 | 66 | 72.2 | 76.6 | 80.9 | 79.1 |
   | | MedQA <br>2-shot | 56.4 | 49.6 | 50 | 62.2 | 60.5 | 63.4 |
   | | OpenBookQA <br>10-shot | 78.8 | 78.6 | 79.8 | 85.8 | 82.6 | 86 |
   | | PIQA <br>5-shot | 80.1 | 78.1 | 77.7 | 86 | 75.7 | 86.6 |
   | | GPQA <br>0-shot | 29.7 | 2.9 | 15 | 6.9 | 32.4 | 29.9 |
   | | Social IQA <br>5-shot | 74.7 | 65.5 | 74.6 | 75.9 | 73.9 | 68.3 |
   | | TruthfulQA (MC2) <br>10-shot | 64.8 | 52.1 | 53 | 60.1 | 63.2 | 67.7 |
   | | WinoGrande <br>5-shot | 71.0 | 55.6 | 54.2 | 62 | 65 | 68.8 |
   | Factual Knowledge | TriviaQA <br>5-shot | 57.8 | 72.3 | 75.2 | 82.2 | 67.7 | 85.8 |
   | Math | GSM8K CoTT <br>8-shot | 85.3 | 59.8 | 46.4 | 64.7 | 77.4 | 78.1 |
   | Code Generation | HumanEval <br>0-shot | 60.4 | 34.1 | 28.0 | 37.8 | 60.4 | 62.2 |
   | | MBPP <br>3-shot | 70.0 | 51.5 | 50.8 | 60.2 | 67.7 | 77.8 |
   | **Average** | | **66.4** | **56.0** | **56.4** | **64.4** | **65.5** | **70.3** |

- Table titled: Long Context Benchmark Results
   | Benchmark     | Phi-3 Mini-128K-Instruct | Mistral-7B | Mixtral 8x7B | LLaMA-3-8B-Instruct |
   | :---------------| :--------------------------|:------------|:--------------|:---------------------|
   | GovReport     | 25.3                     | 4.9        | 20.3         | 10.3                |
   | QMSum         | 21.9                     | 15.5       | 20.6         | 2.9                 |
   | Qasper        | 41.6                     | 23.5       | 26.6         | 8.1                 |
   | SQuALITY      | 24.1                     | 14.7       | 16.2         | 25                  |
   | SummScreenFD  | 16.8                     | 9.3        | 11.3         | 5.1                 |
   | **Average**   | **25.9**                 | **13.6**   | **19.0**     | **10.3**            |

- The model achieves a similar level of language understanding and reasoning ability as larger models.
- The model is limited by its size for certain tasks.
- The model's capacity to store world knowledge is limited, as seen in low performance on TriviaQA.
- Augmenting Phi-3-Mini with a search engine may resolve such weaknesses.

## Cross Platform Support
- [ONNX runtime](https://onnxruntime.ai/blogs/accelerating-phi-3) supports Phi-3 mini models across platforms and hardware.
- Optimized phi-3 models are published in ONNX format for CPU and GPU across devices.
- DirectML GPU acceleration is supported for Windows desktops GPUs (AMD, Intel, and NVIDIA).
- Cross platform support includes CPU, GPU, and mobile devices.

## Optimized Configurations
- ONNX models for int4 DML: Quantized to int4 via AWQ.
- ONNX model for fp16 CUDA.
- ONNX model for int4 CUDA: Quantized to int4 via RTN.
- ONNX model for int4 CPU and Mobile: Quantized to int4 via RTN.

## Software
- [PyTorch](https://github.com/pytorch/pytorch)
- [Transformers](https://github.com/huggingface/transformers)
- [Flash-Attention](https://github.com/HazyResearch/flash-attention)

## Hardware
- The Phi-3 Mini-128K-Instruct model uses flash attention, requiring specific GPU hardware.
- Tested GPU types include:
   - NVIDIA A100
   - NVIDIA A6000
   - NVIDIA H100
- For running on NVIDIA V100 or earlier generation GPUs, call AutoModelForCausalLM.from_pretrained() with attn_implementation="eager".
- For optimized inference on GPU, CPU, and Mobile, use the **ONNX** models [128K](https://aka.ms/phi3-mini-128k-instruct-onnx).

## License
- The model is licensed under the [MIT license](https://huggingface.co/microsoft/Phi-3-mini-128k/resolve/main/LICENSE).

## Trademarks
- This project may contain trademarks or logos for projects, products, or services.
- Authorized use of Microsoft trademarks or logos must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks).
- Use of Microsoft trademarks or logos in modified versions must not cause confusion or imply Microsoft sponsorship.
- Use of third-party trademarks or logos is subject to those third-party’s policies.