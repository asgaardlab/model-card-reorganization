## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
CAMB.AI. At CAMB.AI, we're a research team of Interspeech-published, Carnegie Mellon, ex-Siri engineers. We're an ambitious team, globally distributed, with a singular aim of making everyone's voice count.

### Model date:
Not available.

### Model version:
MARS5.

### Model type:
MARS5 is a two-stage AR-NAR pipeline speech model (TTS) with a novel NAR component. It is trained on raw audio together with byte-pair-encoded text. The model architecture consists of an autoregressive transformer model to obtain coarse (L0) encodec speech features, and a multinomial DDPM model to refine the text, reference, and coarse features to produce the remaining encodec codebook values. The output of the DDPM is then vocoded to produce the final audio. The model can be steered with punctuation and capitalization in the input text.

### Training details:
The model is trained on raw audio together with byte-pair-encoded text.

### Paper or other resource for more information:
- Technical docs: [in the docs folder](docs/architecture.md)
- CAMB.AI website: [https://camb.ai/](https://camb.ai/)
- Demo page with samples: [here](https://6b1a3a8e53ae.ngrok.app/)
- Colab quickstart: [https://colab.research.google.com/github/Camb-ai/mars5-tts/blob/master/mars5_demo.ipynb](https://colab.research.google.com/github/Camb-ai/mars5-tts/blob/master/mars5_demo.ipynb)
- API documentation: [docs.camb.ai](https://docs.camb.ai/)

### Citation details:
Not available.

### License:
MARS in English is open-sourced under GNU AGPL 3.0. For using it under a different license, contact help@camb.ai.

### Contact:
- Email: help@camb.ai
- Hiring inquiries: ack@camb.ai

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
MARS5 is intended for English text-to-speech (TTS) generation, capable of generating speech for prosodically hard and diverse scenarios like sports commentary and anime. It can perform both shallow and deep voice cloning. Shallow clone is faster, while deep clone, which requires the transcript of the reference audio, provides higher quality output. The model takes text and a 5-second audio reference as input to generate speech.

### Primary intended users:
Users who need English TTS capabilities, particularly for scenarios requiring prosody control and voice cloning.

### Out-of-scope uses:
Not available.

---

## How to Use
This section outlines how to use the model.

We use `torch.hub` to make loading the model easy -- no cloning of the repo needed. The steps to perform inference are simple:

1. **Install pip dependencies**: `huggingface_hub`, `torch`, `torchaudio`, `librosa`, `vocos`, and `encodec`. Python must be at version 3.10 or greater, and torch must be v2.0 or greater.

```bash
pip install --upgrade torch torchaudio librosa vocos encodec huggingface_hub
```

2. **Load models**: load the Mars 5 AR and NAR model from the huggingface hub:

```python
from inference import Mars5TTS, InferenceConfig as config_class
import librosa
mars5 = Mars5TTS.from_pretrained("CAMB-AI/MARS5-TTS")
# The `mars5` contains the AR and NAR model, as well as inference code.
# The `config_class` contains tunable inference config settings like temperature.
```
3. **Pick a reference** and optionally its transcript:

```python
# load reference audio between 1-12 seconds.
wav, sr = librosa.load('<path to arbitrary 24kHz waveform>.wav',
                       sr=mars5.sr, mono=True)
wav = torch.from_numpy(wav)
ref_transcript = "<transcript of the reference audio>"
```

The reference transcript is an optional piece of info you need if you wish to do a deep clone.
Mars5 supports 2 kinds of inference: a shallow, fast inference whereby you do not need the transcript of the reference (we call this a _shallow clone_), and a second slower, but typically higher quality way, which we call a _deep clone_.
To use the deep clone, you need the prompt transcript. See the [model docs](docs/architecture.md) for more info on this.

4. **Perform the synthesis**:

```python
# Pick whether you want a deep or shallow clone. Set to False if you don't know prompt transcript or want fast inference. Set to True if you know transcript and want highest quality.
deep_clone = True
# Below you can tune other inference settings, like top_k, temperature, top_p, etc...
cfg = config_class(deep_clone=deep_clone, rep_penalty_window=100,
                      top_k=100, temperature=0.7, freq_penalty=3)

ar_codes, output_audio = mars5.tts("The quick brown rat.", wav,
          ref_transcript,
          cfg=cfg)
# output_audio is (T,) shape float tensor corresponding to the 24kHz output audio.
```

That's it! These default settings provide pretty good results, but feel free to tune the inference settings to optimize the output for your particular example. See the [`InferenceConfig`](inference.py) code or the demo notebook for info and docs on all the different inference settings.

_Some tips for best quality:_
- Make sure reference audio is clean and between 1 second and 12 seconds.
- Use deep clone and provide an accurate transcript for the reference.
- Use proper punctuation -- the model can be guided and made better or worse with proper use of punctuation and capitalization.

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
- Reference audio file between 2-12 seconds (optimal results around 6s).
- Providing the transcript of the reference audio for '_deep clone_' improves quality.
- Punctuation and capitalization in the input transcript can guide the prosody of the generated output.

### Evaluation factors:
Not available.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
Not available.

### Decision thresholds:
Not applicable.

### Variation approaches:
Not available.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
Not available.

### Motivation:
Not available.

### Preprocessing:
Not available.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
Trained on raw audio and byte-pair-encoded text.

### Motivation:
Justification for dataset choice is not available.

### Preprocessing:
Byte-pair encoding was applied to the text data.

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
Two checkpoints:
  - AR fp16 checkpoint: ~750M parameters
  - NAR fp16 checkpoint: ~450M parameters

### Deploying Requirements:
At least 20GB of GPU VRAM is needed to run the model on GPU.

### Training or Fine-tuning Requirements:
Not available.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

Not available.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
Mars 5 is not perfect at the moment, and areas for improvement include:
- Inference stability and consistency
- Speed/performance optimizations
- Improving reference audio selection when given long references.
- Benchmark performance numbers for Mars 5 on standard speech datasets.

### Recommendations:
- Make sure reference audio is clean and between 1 second and 12 seconds.
- Use deep clone and provide an accurate transcript for the reference for best quality.
- Use proper punctuation and capitalization in the input text to guide prosody.

---

## Additional Information
- Checkpoints for MARS5 are provided under the releases tab of this github repo.
- The byte-pair encoding tokenizer used for the L0 encodec codes and the English text is embedded in each checkpoint under the `'vocab'` key, and follows roughly the same format of a saved minbpe tokenizer.
- In general, at least **20GB of GPU VRAM** is needed to run the model on GPU (we plan to further optimize this in the future).
- If you do not have the necessary hardware requirements and just want to use MARS5 in your applications, you can use it via our API: see [docs.camb.ai](https://docs.camb.ai/). If you need some more credits to test it for your use case, feel free to reach out to `help@camb.ai` for help.
- If you would like to contribute any improvement to MARS, please feel free to contribute (guidelines below).
- We welcome any contributions to improving the model. As you may find when experimenting, it can produce really great results, it can still be further improved to create excellent outputs _consistently_.  Please raise a PR/discussion in github.

**Contribution format**:

The preferred way to contribute to our repo is to fork the [master repository](https://github.com/Camb-ai/mars5-tts) on GitHub:

1. Fork the repo on github
2. Clone the repo, set upstream as this repo: `git remote add upstream git@github.com:Camb-ai/mars5-tts.git`
3. Make to a new local branch and make your changes, commit changes.
4. Push changes to new upstream branch: `git push --set-upstream origin <NAME-NEW-BRANCH>`
5. On github, go to your fork and click 'Pull request' to begin the PR process. Please make sure to include a description of what you did/fixed.

Parts of code for this project are adapted from the following repositories -- please make sure to check them out! Thank you to the authors of:

- AWS: For providing much needed compute resources (NVIDIA H100s) to enable training of the model.
- TransFusion: [https://github.com/RF5/transfusion-asr](https://github.com/RF5/transfusion-asr)
- Multinomial diffusion: [https://github.com/ehoogeboom/multinomial_diffusion](https://github.com/ehoogeboom/multinomial_diffusion)
- Mistral-src: [https://github.com/mistralai/mistral-src](https://github.com/mistralai/mistral-src)
- minbpe: [https://github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)
- gemelo-ai's encodec Vocos: [https://github.com/gemelo-ai/vocos](https://github.com/gemelo-ai/vocos)
- librosa for their `.trim()` code: [https://librosa.org/doc/main/generated/librosa.effects.trim.html](https://librosa.org/doc/main/generated/librosa.effects.trim.html)
- We're actively hiring; please drop us an email at ack@camb.ai if you're interested. Visit our [careers page](https://www.camb.ai/careers) for more info.
- ![MARS5 Banner](assets/github-banner.png)
- ![Mars 5 simplified diagram](docs/assets/simplified_diagram.png)
- **Quick links**:
    - [CAMB.AI website](https://camb.ai/) (access MARS5 in 140+ languages for TTS and dubbing)
    - Colab quickstart: <a target="_blank" href="https://colab.research.google.com/github/Camb-ai/mars5-tts/blob/master/mars5_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    - Demo page with samples: [here](https://6b1a3a8e53ae.ngrok.app/)
- **Figure**: the high-level architecture flow of Mars 5. Given text and a reference audio, coarse (L0) encodec speech features are obtained through an autoregressive transformer model. Then, the text, reference, and coarse features are refined in a multinomial DDPM model to produce the remaining encodec codebook values. The output of the DDPM is then vocoded to produce the final audio.