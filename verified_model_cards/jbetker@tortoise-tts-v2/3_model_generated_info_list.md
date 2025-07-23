# TorToiSe
- Tortoise is a text-to-speech program.
- Tortoise has the following priorities:
  1. Strong multi-voice capabilities.
  2. Highly realistic prosody and intonation.
- This repository contains all the code needed to run Tortoise TTS in inference mode.

### New features
#### v2.1; 2022/5/2
- Added ability to produce totally random voices.
- Added ability to download voice conditioning latent via a script.
- Added ability to use a user-provided conditioning latent.
- Added ability to use your own pretrained models.
- Refactored directory structures.
- Performance improvements and bug fixes.

## What's in a name?
- The speech-related repositories are named after Mojave desert flora and fauna.
- The name "Tortoise" is tongue-in-cheek because the model is slow.
- Tortoise leverages both an autoregressive decoder and a diffusion decoder.
- Both decoders are known for their low sampling rates.
- On a K80, expect to generate a medium-sized sentence every 2 minutes.

## Demos
- See [this page](http://nonint.com/static/tortoise_v2_examples.html) for a large list of example outputs.

## Usage guide

### Installation
- You must have an NVIDIA GPU to use this on your own computer.
- First, install PyTorch using these instructions: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
- Then, run the following commands:
  ```shell
  git clone https://github.com/neonbjb/tortoise-tts.git
  cd tortoise-tts
  python setup.py install
  ```

### do_tts.py
- This script allows you to speak a single phrase with one or more voices.
- Example command:
  ```shell
  python tortoise/do_tts.py --text "I'm going to speak this" --voice random --preset fast
  ```

### read.py
- This script provides tools for reading large amounts of text.
- Example command:
  ```shell
  python tortoise/read.py --textfile <your text to be read> --voice random
  ```
- This will break up the text file into sentences and convert them to speech one at a time.
- It will output a series of spoken clips as they are generated.
- Once all clips are generated, it will combine them into a single file and output that as well.
- You can re-generate any bad clips by re-running `read.py` with the --regenerate argument.

### API
- Tortoise can be used programmatically.
- Example code:
  ```python
  reference_clips = [utils.audio.load_audio(p, 22050) for p in clips_paths]
  tts = api.TextToSpeech()
  pcm_audio = tts.tts_with_preset("your text here", reference_clips, preset='fast')
  ```

## Voice customization guide
- Tortoise was specifically trained to be a multi-speaker model.
- It consults reference clips to determine properties of the output.
- Reference clips are recordings of a speaker that you provide.
- These clips determine pitch, tone, speaking speed, and speaking defects.
- Reference clips also determine non-voice related aspects like volume, background noise, recording quality, and reverb.

### Random voice
- A feature randomly generates a voice.
- These voices do not actually exist and will be random every time you run it.
- You can use the random voice by passing in 'random' as the voice name.

### Provided voices
- This repository comes with several pre-packaged voices.
- Most provided voices were not found in the training set.
- Voices from the training set produce more realistic outputs than those outside of it.
- Any voice prepended with "train" came from the training set.

### Adding a new voice
- To add new voices to Tortoise, follow these steps:
  1. Gather audio clips of your speaker(s).
  2. Cut your clips into ~10 second segments.
  3. Save the clips as a WAV file with floating point format and a 22,050 sample rate.
  4. Create a subdirectory in voices/.
  5. Put your clips in that subdirectory.
  6. Run tortoise utilities with --voice=<your_subdirectory_name>.

### Picking good reference clips
- Tips for picking good clips:
  1. Avoid clips with background music, noise, or reverb.
  2. Avoid speeches that have distortion caused by amplification systems.
  3. Avoid clips from phone calls.
  4. Avoid clips with excessive stuttering or filler words.
  5. Find clips that are spoken in the desired output style.
  6. The text spoken in the clips does not matter, but diverse text performs better.

## Advanced Usage

### Generation settings
- Tortoise is primarily an autoregressive decoder model combined with a diffusion model.
- The settings can be adjusted for specific use-cases.
- These settings are available in the API.

### Prompt engineering
- It is possible to do prompt engineering with Tortoise.
- You can evoke emotion by including phrases like "I am really sad," before your text.
- An automated redaction system can redact text in the prompt surrounded by brackets.

### Playing with the voice latent
- Tortoise ingests reference clips to produce a point latent.
- The mean of all produced latents is taken.
- You can combine two different voices to output an average of those voices.

#### Generating conditioning latents from voices
- Use the script `get_conditioning_latents.py` to extract conditioning latents.
- Alternatively, use the api.TextToSpeech.get_conditioning_latents() to fetch the latents.

#### Using raw conditioning latents to generate speech
- Create a subdirectory in voices/ with a single ".pth" file containing the pickled conditioning latents.

### Send me feedback!
- Community involvement is important for exploring the capabilities of Tortoise.
- If you find something neat that isn't documented, please report it.

## Tortoise-detect
- A classifier tells the likelihood that an audio clip came from Tortoise.
- Usage example:
  ```commandline
  python tortoise/is_this_from_tortoise.py --clip=<path_to_suspicious_audio_file>
  ```
- This model has 100% accuracy on the contents of the results/ and voices/ folders.
- Treat this classifier as a "strong signal" as it can exhibit false positives.

## Model architecture
- Tortoise TTS is inspired by OpenAI's DALLE.
- It is made up of 5 separate models that work together.
- A write-up of the system architecture is available here: [https://nonint.com/2022/04/25/tortoise-architectural-design-doc/](https://nonint.com/2022/04/25/tortoise-architectural-design-doc/).

## Training
- Models were trained on a "homelab" server with 8 RTX 3090s.
- The dataset consisted of ~50k hours of speech data.
- Most of the data was transcribed by [ocotillo](http://www.github.com/neonbjb/ocotillo).
- Training was done on a [DLAS](https://github.com/neonbjb/DL-Art-School) trainer.
- There are currently no plans to release the training configurations or methodology.

## Ethical Considerations
- Tortoise v2 works better than planned.
- Concerns exist about the potential misuse of voice-cloning technology.
- Reasons for releasing Tortoise include:
  1. It is primarily good at reading books and poetry.
  2. It was trained on a dataset without public figure voices.
  3. The model could be scaled up, but details are withheld pending feedback.
  4. A classifier model is released to identify Tortoise-generated audio.
  5. Open knowledge of ML capabilities is preferred.

### Diversity
- The diversity of ML models is tied to their training datasets.
- Tortoise was trained primarily on audiobooks.
- No effort was made to balance diversity in the dataset.
- Tortoise may be poor at generating voices of minorities or those with strong accents.

## Looking forward
- Tortoise v2 is considered the best achievable with current resources.
- Training large models increases communication bandwidth needs.
- The largest model in Tortoise v2 is smaller than GPT-2 large.
- Collaboration is welcomed from ethical organizations with computational resources.

## Acknowledgements
- Credit is given to several contributors and inspirations:
  - Hugging Face for the GPT model and generate API.
  - [Ramesh et al](https://arxiv.org/pdf/2102.12092.pdf) for the DALLE paper.
  - [Nichol and Dhariwal](https://arxiv.org/pdf/2102.09672.pdf) for the diffusion model code.
  - [Jang et al](https://arxiv.org/pdf/2106.07889.pdf) for the vocoder used in the repo.
  - [lucidrains](https://github.com/lucidrains) for open source PyTorch models.
  - [Patrick von Platen](https://huggingface.co/patrickvonplaten) for guides on setting up wav2vec.

## Notice
- Tortoise was built entirely by the author using personal hardware.
- The employer was not involved in Tortoise's development.
- If you use this repo or its ideas for research, please cite it.