## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
Distil-Whisper was proposed in the paper [Robust Knowledge Distillation via Large-Scale Pseudo Labelling](https://arxiv.org/abs/2311.00430) by Sanchit Gandhi, Patrick von Platen, and Alexander M. Rush. They are researchers in the field of natural language processing and machine learning, affiliated with Hugging Face.

### Model date:
Distil-Whisper was proposed in November 2023, with the paper published on arXiv in November 2023. The distil-large-v2 model was released around the same time, with updates and further models (like distil-large-v3) following.

### Model version:
This model card describes **distil-large-v2**, a distilled variant of [Whisper large-v2](https://huggingface.co/openai/whisper-large-v2). It is one of several Distil-Whisper models, including:
- [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3): An updated model surpassing distil-large-v2 in performance, recommended for use instead of distil-large-v2.
- [distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en)
- [distil-small.en](https://huggingface.co/distil-whisper/distil-small.en)

Distil-large-v2 is 49% smaller and 6 times faster than Whisper large-v2, while performing within 1% WER on out-of-distribution evaluation sets.

### Model type:
Distil-Whisper is an encoder-decoder architecture, inheriting its structure from the Whisper model. It is a distilled version of Whisper, specifically distil-large-v2 is a distilled variant of Whisper large-v2. The encoder maps speech inputs to hidden-state vectors, and the decoder auto-regressively predicts text tokens. The encoder is copied and frozen from the teacher model (Whisper large-v2), while the decoder has only two layers, initialized from the first and last layers of the teacher decoder. This reduction in decoder layers is key to its efficiency. Model size is 756M parameters for distil-large-v2. Context length is not explicitly specified but inherits from Whisper architecture.

### Training details:
The model was trained using knowledge distillation. The encoder of Distil-Whisper is frozen and copied from the Whisper large-v2 model. The decoder is trained using a weighted sum of KL divergence and pseudo-label loss terms. Training was performed for 80,000 optimisation steps (or eight epochs). Whisper large-v2 was used to generate pseudo-labels for the training data. A WER heuristic filter was employed to discard training examples with high WER between pseudo-labels and ground truth labels. Tensorboard training logs are available at: https://huggingface.co/distil-whisper/distil-large-v2/tensorboard?params=scalars#frame

### Paper or other resource for more information:
- **Paper:** [Robust Knowledge Distillation via Large-Scale Pseudo Labelling](https://arxiv.org/abs/2311.00430) - This paper introduces Distil-Whisper and details the distillation process, architecture, and evaluation.
- **Distil-Whisper repository:** [https://github.com/huggingface/distil-whisper/](https://github.com/huggingface/distil-whisper/) - This repository contains the training code and further information.
- **Hugging Face Model Hub:** [https://huggingface.co/distil-whisper/distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2) - Model card and access to the model.
- **Training code:** [https://github.com/huggingface/distil-whisper/tree/main/training](https://github.com/huggingface/distil-whisper/tree/main/training) - Code to reproduce the training process.
- **Hugging Face 🤗 Transformers documentation:** [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/) - For information on using the model with the Transformers library.
- **Hugging Face Candle repository:** [https://github.com/huggingface/candle/tree/main](https://github.com/huggingface/candle/tree/main) - For information on using the model with the Candle library.
- **Whisper.cpp repository:** [https://github.com/ggerganov/whisper.cpp](https://github.com/ggerganov/whisper.cpp) - For information on using the model with Whisper.cpp.
- **Transformers.js documentation:** [https://huggingface.co/docs/transformers.js/api/pipelines#module_pipelines.AutomaticSpeechRecognitionPipeline](https://huggingface.co/docs/transformers.js/api/pipelines#module_pipelines.AutomaticSpeechRecognitionPipeline) - For information on using the model with Transformers.js.

### Citation details:
```
@misc{gandhi2023distilwhisper,
      title={Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling},
      author={Sanchit Gandhi and Patrick von Platen and Alexander M. Rush},
      year={2023},
      eprint={2311.00430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### License:
Distil-Whisper inherits the [MIT license](https://github.com/huggingface/distil-whisper/blob/main/LICENSE) from OpenAI's Whisper model. Full license text is available at the provided link. Users are permitted to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, subject to the conditions of the MIT License.

### Contact:
Not available.

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
Distil-Whisper is primarily intended for English speech recognition tasks. It is designed as a faster and smaller drop-in replacement for the Whisper large-v2 model, offering comparable Word Error Rate (WER) performance, especially on out-of-distribution data. It can be used for both short-form (under 30 seconds) and long-form audio transcription. The model is capable of transcribing audio files from local paths, datasets, and remote URLs. It can also be used as an assistant model for speculative decoding with Whisper, further accelerating inference. Input is audio, and output is transcribed text.

### Primary intended users:
The primary intended users are researchers, developers, and practitioners in the field of speech recognition and natural language processing who require efficient and accurate English speech-to-text capabilities. This includes users of the original Whisper model who seek faster inference without significant performance degradation.

### Out-of-scope uses:
Distil-Whisper is currently only available and trained for English speech recognition. Its use for other languages is out-of-scope in its current version. While multilingual distillation is planned, the current model is not intended for non-English audio transcription.

---

## How to Use
This section outlines how to use the model.

Distil-Whisper is supported in Hugging Face 🤗 Transformers from version 4.35 onwards.

**Installation:**
```bash
pip install --upgrade pip
pip install --upgrade transformers accelerate datasets[audio]
```

**Short-Form Transcription ( < 30-seconds):**
```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```
To transcribe a local audio file, replace `sample` with the path to your audio file: `result = pipe("audio.mp3")`.

**Long-Form Transcription ( > 30-seconds):**
```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "distil-whisper/distil-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15, # Optimal chunk length for Distil-Whisper
    batch_size=16, # Optional batching
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

**Speculative Decoding with Whisper:**
```python
from transformers import pipeline, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
from datasets import load_dataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

assistant_model_id = "distil-whisper/distil-large-v2"

assistant_model = AutoModelForCausalLM.from_pretrained(
    assistant_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
assistant_model.to(device)

model_id = "openai/whisper-large-v2"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    generate_kwargs={"assistant_model": assistant_model},
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

**Running in `openai-whisper` format:**
```python
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from whisper import load_model, transcribe

distil_large_v2 = hf_hub_download(repo_id="distil-whisper/distil-large-v2", filename="original-model.bin")
model = load_model(distil_large_v2)

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = dataset[0]["audio"]["array"]
sample = torch.from_numpy(sample).float()

pred_out = transcribe(model, audio=sample)
print(pred_out["text"])
```
For local audio file: `pred_out = transcribe(model, audio="audio.mp3")`

**Running with Whisper.cpp:**
Instructions provided in the original model card for cloning the repository, downloading weights, and running inference.

**Running with Transformers.js:**
```javascript
import { pipeline } from '@xenova/transformers';

let transcriber = await pipeline('automatic-speech-recognition', 'distil-whisper/distil-large-v2');

let url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav';
let output = await transcriber(url);
// { text: " And so, my fellow Americans, ask not what your country can do for you. Ask what you can do for your country." }
```

**Running with Candle:**
Instructions provided in the original model card for installing Candle, cloning the repository, and running examples.

**Speed and Memory Improvements:**
- **Flash Attention 2:** (If GPU allows)
  ```diff
  - model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
  + model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=True)
  ```
  Installation: `pip install flash-attn --no-build-isolation`

- **Torch Scale-Product-Attention (SDPA) via BetterTransformers:** (If GPU does not support Flash Attention)
  ```diff
  model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
  + model = model.to_bettertransformer()
  ```
  Installation: `pip install --upgrade optimum`

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
The primary relevant factor is language, as Distil-Whisper is currently trained and optimized for English speech recognition. Performance may vary on different English accents and audio qualities. Audio length (short-form vs. long-form) is also a factor, with specific optimizations and usage recommendations provided for each.

### Evaluation factors:
Evaluation factors include Word Error Rate (WER) on various datasets, particularly out-of-distribution datasets and the ESB benchmark. Performance is evaluated across different audio types (short-form and long-form) and compared to the Whisper large-v2 model.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
The primary performance metric used is **Word Error Rate (WER)**. WER is a standard metric for evaluating automatic speech recognition systems, measuring the number of errors (substitutions, insertions, deletions) relative to the number of words in the reference transcript. Lower WER indicates better performance.

### Decision thresholds:
Not available.

### Variation approaches:
Performance metrics are calculated on standard validation datasets like LibriSpeech validation.clean and ESB benchmark datasets. Evaluation is performed in a streaming mode to avoid downloading the entire dataset locally. The evaluation process involves normalizing transcriptions and references using `EnglishTextNormalizer` before computing WER to ensure consistent comparison.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
- **LibriSpeech validation.clean:** A commonly used dataset for evaluating speech recognition models, consisting of read audiobooks. Used for WER evaluation in the provided code example. Publicly available via Hugging Face Datasets.
- **ESB benchmark datasets:** A collection of datasets used for the [OpenASR leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard), used to evaluate Distil-Whisper's performance in a broader context. Details on specific datasets within ESB benchmark are available in the [ESB benchmark paper](https://arxiv.org/abs/2210.13352).
- **Out-of-distribution (OOD) evaluation sets:**  Referenced in the model description as being used to compare Distil-Whisper and Whisper performance, but specific datasets are not named in this model card content. Details likely available in the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430).

### Motivation:
LibriSpeech validation.clean is used as a standard benchmark for speech recognition. ESB benchmark datasets provide a broader evaluation across diverse datasets, as reflected in the OpenASR leaderboard. Out-of-distribution datasets are crucial to assess the robustness and generalization capability of Distil-Whisper compared to Whisper, ensuring it performs well beyond the training data distribution.

### Preprocessing:
During evaluation, audio data is preprocessed into log-mel spectrogram inputs using the model's processor. Text transcriptions and references are normalized using `EnglishTextNormalizer` from `transformers.models.whisper.english_normalizer` to standardize text format before WER calculation. This normalization includes handling English spelling variations.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
Distil-Whisper was trained on approximately 22,000 hours of audio data from 9 open-source, permissively licensed speech datasets on the Hugging Face Hub:

| Dataset                                                                                 | Size / h | Speakers | Domain                      | Licence         |
|-----------------------------------------------------------------------------------------|----------|----------|-----------------------------|-----------------|
| [People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech)             | 12,000   | unknown  | Internet Archive            | CC-BY-SA-4.0    |
| [Common Voice 13](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0) | 3,000    | unknown  | Narrated Wikipedia          | CC0-1.0         |
| [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech)                    | 2,500    | unknown  | Audiobook, podcast, YouTube | apache-2.0      |
| Fisher                                                                                  | 1,960    | 11,900   | Telephone conversations     | LDC             |
| [LibriSpeech](https://huggingface.co/datasets/librispeech_asr)                          | 960      | 2,480    | Audiobooks                  | CC-BY-4.0       |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli)                         | 540      | 1,310    | European Parliament         | CC0             |
| [TED-LIUM](https://huggingface.co/datasets/LIUM/tedlium)                                | 450      | 2,030    | TED talks                   | CC-BY-NC-ND 3.0 |
| SwitchBoard                                                                             | 260      | 540      | Telephone conversations     | LDC             |
| [AMI](https://huggingface.co/datasets/edinburghcstr/ami)                                | 100      | unknown  | Meetings                    | CC-BY-4.0       |
||||||
| **Total**                                                                               | 21,770   | 18,260+  |                             |                 |

These datasets are publicly available on the Hugging Face Hub, except for Fisher and SwitchBoard which are LDC datasets. The combined dataset includes diverse domains such as internet archives, narrated Wikipedia, audiobooks, podcasts, YouTube videos, telephone conversations, TED talks, and meetings.

### Motivation:
The datasets were chosen for their diversity in domain, speaker characteristics, and audio conditions. This diversity is intended to ensure the distilled model is robust and generalizes well across various real-world audio distributions and noise conditions. The use of permissively licensed, open-source datasets aligns with principles of open research and accessibility.

### Preprocessing:
The audio data was pseudo-labeled using the Whisper large-v2 model. These pseudo-labels served as target transcriptions during training. A WER filter was applied: for each training example, the WER between the Whisper pseudo-label and the original ground truth label (normalized) was calculated. If the WER exceeded a threshold, the example was discarded. Text normalization was applied to both pseudo-labels and ground truth labels before WER calculation and training.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
| Model                                                                      | Params / M | Rel. Latency ↑ | Short-Form WER ↓ | Long-Form WER ↓ |
|----------------------------------------------------------------------------|------------|----------------|------------------|-----------------|
| [large-v3](https://huggingface.co/openai/whisper-large-v3)                 | 1550       | 1.0            | **8.4**          | 11.0            |
| [large-v2](https://huggingface.co/openai/whisper-large-v2)                 | 1550       | 1.0            | 9.1              | 11.7            |
|                                                                            |            |                |                  |                 |
| [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3)   | 756        | 6.3            | 9.7              | **10.8**        |
| [distil-large-v2](https://huggingface.co/distil-whisper/distil-large-v2)   | 756        | 5.8            | 10.1             | 11.6            |
| [distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en) | 394        | **6.8**        | 11.1             | 12.4            |
| [distil-small.en](https://huggingface.co/distil-whisper/distil-small.en)   | **166**    | 5.6            | 12.1             | 12.8            |

This table shows the performance of different Whisper and Distil-Whisper models in terms of Word Error Rate (WER) on short-form and long-form audio, along with model size (parameters) and relative latency. Distil-large-v2 achieves comparable WER to large-v2, while being significantly faster (5.8x relative latency improvement) and smaller.

### Intersectional results:
Detailed per-dataset breakdown of evaluation results is available in Tables 16 and 17 of the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430). These tables provide a more granular view of performance across various datasets, likely showing performance variations across different data distributions.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
Distil-Whisper is designed for low CPU memory usage (`low_cpu_mem_usage=True` flag in code examples). Specific RAM/VRAM requirements depend on the hardware and usage scenario, but the model is smaller than Whisper large-v2 (756M parameters vs 1550M), suggesting lower memory footprint. Using `use_safetensors=True` is also recommended for efficient loading.

### Deploying Requirements:
For optimal deployment, especially for faster inference, a GPU is recommended. Code examples demonstrate usage with CUDA. For CPU-based deployment, consider using optimized backends like Candle, which offers CPU optimizations and WASM support. Whisper.cpp also provides CPU-based inference.

### Training or Fine-tuning Requirements:
Training of Distil-Whisper was conducted using Google Cloud TPU v4s. Fine-tuning requirements would depend on the specific fine-tuning task and dataset size, but would likely benefit from GPU or TPU acceleration for efficient training.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

Sensitive data was used in training, as the datasets include speech data from various sources, potentially containing personal information.

Potential risks include mis-transcriptions and hallucinations, which are inherent to speech recognition models. While Distil-Whisper aims to minimize these, they can still occur. The WER filter used during training is a risk mitigation strategy to improve the accuracy of pseudo-labels and reduce hallucinations.

Known model use cases are primarily in English speech recognition, where mis-transcriptions could lead to misunderstandings or inaccuracies in downstream applications. Efforts to address these challenges include rigorous evaluation on diverse datasets and the development of techniques like WER filtering. Further exploration is needed to fully understand and mitigate potential ethical risks in various deployment scenarios.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
- **Language Limitation:** Distil-Whisper is currently limited to English speech recognition. Multilingual support is under development but not yet available.
- **Future Improvements:** 8-bit and 4-bit quantization and further Whisper.cpp optimizations are planned for future releases to improve efficiency and accessibility.
- **Evaluation Dataset Gaps:** While evaluated on several datasets, there might be specific demographic groups or audio conditions not fully represented in the evaluation datasets, potentially leading to performance variations in real-world scenarios. Further testing on diverse datasets is recommended.

### Recommendations:
- **Use distil-large-v3:** For best performance, it is recommended to use the updated [distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) model instead of distil-large-v2, as it offers improved performance.
- **Consider Flash Attention or SDPA:** For GPU-based inference, utilize Flash Attention 2 or Torch SDPA via BetterTransformers for significant speed improvements.
- **Explore Candle for CPU/WASM:** For CPU-based or browser-based applications, explore the Candle integration for optimized performance.
- **Further Evaluation:** Users are encouraged to evaluate Distil-Whisper on datasets relevant to their specific use cases to ensure it meets their performance requirements, especially considering potential variations across different audio conditions and demographic groups.

---

## Additional Information
<div class="course-tip course-tip-orange bg-gradient-to-br dark:bg-gradient-to-r before:border-orange-500 dark:before:border-orange-800 from-orange-50 dark:from-gray-900 to-white dark:to-gray-950 border border-orange-50 text-orange-700 dark:text-gray-400">
  <p><b>Update:</b> following the release of OpenAI's Whisper large-v3, an updated <a href="ttps://huggingface.co/distil-whisper/distil-large-v3"> distil-large-v3</a> model was published. This <a href="ttps://huggingface.co/distil-whisper/distil-large-v3"> distil-large-v3</a> model surpasses the performance of the distil-large-v2 model, with no architecture changes and better support for sequential long-form generation. Thus, it is recommended that the <a href="ttps://huggingface.co/distil-whisper/distil-large-v3"> distil-large-v3</a> model is used in-place of the large-v2 model. </p>
</div>

**Note:** Distil-Whisper is currently only available for English speech recognition. We are working with the community
to distill Whisper on other languages. If you are interested in distilling Whisper in your language, check out the
provided [training code](https://github.com/huggingface/distil-whisper/tree/main/training). We will update the
[Distil-Whisper repository](https://github.com/huggingface/distil-whisper/) with multilingual checkpoints when ready!

## Acknowledgements
* OpenAI for the Whisper [model](https://huggingface.co/openai/whisper-large-v2) and [original codebase](https://github.com/openai/whisper)
* Hugging Face 🤗 [Transformers](https://github.com/huggingface/transformers) for the model integration
* Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/about/) programme for Cloud TPU v4s
* [`@rsonavane`](https://huggingface.co/rsonavane/distil-whisper-large-v2-8-ls) for releasing an early iteration of Distil-Whisper on the LibriSpeech dataset
