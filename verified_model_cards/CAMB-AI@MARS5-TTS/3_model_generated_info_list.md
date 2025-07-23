## MARS5: A novel speech model for insane prosody.
- ![MARS5 Banner](assets/github-banner.png)
- This is the repo for the MARS5 English speech model (TTS) from CAMB.AI.
- The model follows a two-stage AR-NAR pipeline.
- The model has a distinctively novel NAR component.
- More information can be found in the [docs](docs/architecture.md).
- MARS5 can generate speech with just 5 seconds of audio and a snippet of text.
- MARS5 is capable of handling prosodically hard and diverse scenarios like sports commentary and anime.
- A demo can be found at: 
   https://github.com/Camb-ai/MARS5-TTS/assets/23717819/3e191508-e03c-4ff9-9b02-d73ae0ebefdd

**Quick links**:
- [CAMB.AI website](https://camb.ai/) (access MARS5 in 140+ languages for TTS and dubbing)
- Technical docs: [in the docs folder](docs/architecture.md)
- Colab quickstart: <a target="_blank" href="https://colab.research.google.com/github/Camb-ai/mars5-tts/blob/master/mars5_demo.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
- Demo page with samples: [here](https://6b1a3a8e53ae.ngrok.app/)

- ![Mars 5 simplified diagram](docs/assets/simplified_diagram.png)
- **Figure**: The high-level architecture flow of Mars 5 is described.
- Given text and a reference audio, coarse (L0) encoded speech features are obtained through an autoregressive transformer model.
- The text, reference, and coarse features are refined in a multinomial DDPM model to produce the remaining encoded codebook values.
- The output of the DDPM is then vocoded to produce the final audio.
- The model is trained on raw audio together with byte-pair-encoded text.
- The model can be steered with punctuation and capitalization.
- To add a pause, a comma can be added to that part in the transcript.
- To emphasize a word, it can be put in capital letters in the transcript.
- This enables a natural way for guiding the prosody of the generated output.
- Speaker identity is specified using an audio reference file between 2-12 seconds.
- Lengths around 6 seconds give optimal results.
- By providing the transcript of the reference, MARS5 enables a '_deep clone_'.
- A deep clone improves the quality of the cloning and output.
- The deep clone takes longer to produce the audio.
- For more details on performance and model details, see inside the [docs folder](docs/architecture.md).

## Quickstart
- The model can be loaded easily using `torch.hub` without cloning the repo.
- Steps to perform inference are provided.

1. **Install pip dependencies**: 
   - Required packages: `huggingface_hub`, `torch`, `torchaudio`, `librosa`, `vocos`, and `encodec`.
   - Python must be at version 3.10 or greater.
   - Torch must be v2.0 or greater.
   ```bash
   pip install --upgrade torch torchaudio librosa vocos encodec huggingface_hub
   ```

2. **Load models**: 
   - Load the Mars 5 AR and NAR model from the huggingface hub:
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
- The reference transcript is optional for a deep clone.
- Mars5 supports two kinds of inference: shallow (fast) and deep (higher quality).
- To use the deep clone, the prompt transcript is needed.
- See the [model docs](docs/architecture.md) for more info.

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
- Default settings provide good results, but inference settings can be tuned for optimization.
- See the [`InferenceConfig`](inference.py) code or the demo notebook for info on different inference settings.

_Some tips for best quality:_
- Ensure reference audio is clean and between 1 second and 12 seconds.
- Use deep clone and provide an accurate transcript for the reference.
- Use proper punctuation to guide the model.

## Model details
- **Checkpoints**: 
   - The checkpoints for MARS5 are provided under the releases tab of this GitHub repo.
   - Two checkpoints are provided:
     - AR fp16 checkpoint [~750M parameters], along with config embedded in the checkpoint.
     - NAR fp16 checkpoint [~450M parameters], along with config embedded in the checkpoint.
   - The byte-pair encoding tokenizer used for the L0 encodec codes and the English text is embedded in each checkpoint under the `'vocab'` key.

- **Hardware requirements**:
   - At least 750M+450M params must be stored on GPU.
   - Inference requires 750M of active parameters.
   - At least **20GB of GPU VRAM** is needed to run the model on GPU.
   - Future optimizations are planned.

- If hardware requirements are not met, MARS5 can be used via the API: see [docs.camb.ai](https://docs.camb.ai/).
- For more credits to test, contact `help@camb.ai`.

## Roadmap
- Mars 5 is not perfect and improvements are ongoing.
- Areas for improvement include:
   - Inference stability and consistency.
   - Speed/performance optimizations.
   - Reference audio selection for long references.
   - Benchmark performance numbers on standard speech datasets.
- Contributions are welcome.

## Contributions
- Contributions to improve the model are welcomed.
- The model can produce great results but can be improved for consistency.
- Raise a PR/discussion in GitHub.

**Contribution format**:
- The preferred way to contribute is to fork the [master repository](https://github.com/Camb-ai/mars5-tts) on GitHub:
   1. Fork the repo on GitHub.
   2. Clone the repo and set upstream: `git remote add upstream git@github.com:Camb-ai/mars5-tts.git`.
   3. Create a new local branch and make changes, then commit.
   4. Push changes to the new upstream branch: `git push --set-upstream origin <NAME-NEW-BRANCH>`.
   5. On GitHub, go to your fork and click 'Pull request' to begin the PR process. Include a description of what you did/fixed.

## License
- MARS is open-sourced in English under GNU AGPL 3.0.
- Requests for a different license can be made by emailing help@camb.ai.

## Join our team
- CAMB.AI is a globally distributed research team.
- The team consists of Interspeech-published, Carnegie Mellon, ex-Siri engineers.
- The team is actively hiring; interested individuals can email ack@camb.ai.
- Visit the [careers page](https://www.camb.ai/careers) for more info.

## Acknowledgements
- Parts of code for this project are adapted from various repositories.
- Thanks to the authors of:
   - AWS: For providing compute resources (NVIDIA H100s) for training.
   - TransFusion: [https://github.com/RF5/transfusion-asr](https://github.com/RF5/transfusion-asr)
   - Multinomial diffusion: [https://github.com/ehoogeboom/multinomial_diffusion](https://github.com/ehoogeboom/multinomial_diffusion)
   - Mistral-src: [https://github.com/mistralai/mistral-src](https://github.com/mistralai/mistral-src)
   - minbpe: [https://github.com/karpathy/minbpe](https://github.com/karpathy/minbpe)
   - gemelo-ai's encodec Vocos: [https://github.com/gemelo-ai/vocos](https://github.com/gemelo-ai/vocos)
   - librosa for their `.trim()` code: [https://librosa.org/doc/main/generated/librosa.effects.trim.html](https://librosa.org/doc/main/generated/librosa.effects.trim.html)