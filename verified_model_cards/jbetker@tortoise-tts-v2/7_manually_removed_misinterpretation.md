## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
neonbjb. Tortoise was built by neonbjb using personal hardware.

### Model date:
- v2.1 release: 2022/5/2
- Architectural design document: 2022/04/25

### Model version:
v2.1, v2

### Model type:
Text-to-speech program. It leverages both an autoregressive decoder and a diffusion decoder. Tortoise TTS is inspired by OpenAI's DALLE, applied to speech data and using a better decoder. It is made up of 5 separate models that work together.

### Training details:
These models were trained on a "homelab" server with 8 RTX 3090s over the course of several months. They were trained on a dataset consisting of ~50k hours of speech data, most of which was transcribed by ocotillo. Training was done on DLAS trainer.  Training configurations or methodology are currently withheld.

### Paper or other resource for more information:
- Example outputs: [http://nonint.com/static/tortoise_v2_examples.html](http://nonint.com/static/tortoise_v2_examples.html)
- PyTorch installation instructions: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
- Architectural design document: [https://nonint.com/2022/04/25/tortoise-architectural-design-doc/](https://nonint.com/2022/04/25/tortoise-architectural-design-doc/)
- DALLE paper: [https://arxiv.org/pdf/2102.12092.pdf](https://arxiv.org/pdf/2102.12092.pdf)
- Diffusion model code (revision): [https://arxiv.org/pdf/2102.09672.pdf](https://arxiv.org/pdf/2102.09672.pdf)
- Univnet vocoder paper: [https://arxiv.org/pdf/2106.07889.pdf](https://arxiv.org/pdf/2106.07889.pdf)
- ocotillo (transcription tool): [http://www.github.com/neonbjb/ocotillo](http://www.github.com/neonbjb/ocotillo)
- DLAS trainer: [https://github.com/neonbjb/DL-Art-School](https://github.com/neonbjb/DL-Art-School)
- Tortoise TTS code repository: [https://github.com/neonbjb/tortoise-tts.git](https://github.com/neonbjb/tortoise-tts.git)
- Patrick von Platen (Hugging Face): [https://huggingface.co/patrickvonplaten](https://huggingface.co/patrickvonplaten)
- lucidrains (pytorch models): [https://github.com/lucidrains](https://github.com/lucidrains)

### Citation details:
A bibtex entree can be found in the right pane on GitHub.

### License:
Not available.

### Contact:
Send feedback to the developer.

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
Text-to-speech. Strong multi-voice capabilities and highly realistic prosody and intonation. Primarily good at reading books and speaking poetry.

### Primary intended users:
Researchers, and individuals interested in text-to-speech technology.

### Out-of-scope uses:
Other forms of speech beyond books and poetry. Clips with background music, noise or reverb. Speeches and clips from phone calls. Clips that have excessive stuttering, stammering or words like "uh" or "like" in them.

---

## How to Use
This section outlines how to use the model.

**Installation:**
If you want to use this on your own computer, you must have an NVIDIA GPU. First, install pytorch using these instructions: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Then:
```shell
git clone https://github.com/neonbjb/tortoise-tts.git
cd tortoise-tts
python setup.py install
```

**do_tts.py:**
This script allows you to speak a single phrase with one or more voices.
```shell
python tortoise/do_tts.py --text "I'm going to speak this" --voice random --preset fast
```

**read.py:**
This script provides tools for reading large amounts of text.

```shell
python tortoise/read.py --textfile <your text to be read> --voice random
```

This will break up the textfile into sentences, and then convert them to speech one at a time. It will output a series
of spoken clips as they are generated. Once all the clips are generated, it will combine them into a single file and
output that as well.

Sometimes Tortoise screws up an output. You can re-generate any bad clips by re-running `read.py` with the --regenerate
argument.

**API:**
Tortoise can be used programmatically, like so:

```python
reference_clips = [utils.audio.load_audio(p, 22050) for p in clips_paths]
tts = api.TextToSpeech()
pcm_audio = tts.tts_with_preset("your text here", reference_clips, preset='fast')
```

**Voice customization guide:**
Tortoise was specifically trained to be a multi-speaker model. It accomplishes this by consulting reference clips.

These reference clips are recordings of a speaker that you provide to guide speech generation. These clips are used to determine many properties of the output, such as the pitch and tone of the voice, speaking speed, and even speaking defects like a lisp or stuttering. The reference clip is also used to determine non-voice related aspects of the audio output like volume, background noise, recording quality and reverb.

**Random voice:**
I've included a feature which randomly generates a voice. These voices don't actually exist and will be random every time you run
it. The results are quite fascinating and I recommend you play around with it!

You can use the random voice by passing in 'random' as the voice name. Tortoise will take care of the rest.

For the those in the ML space: this is created by projecting a random vector onto the voice conditioning latent space.

**Provided voices:**
This repo comes with several pre-packaged voices. You will be familiar with many of them. :)

Most of the provided voices were not found in the training set. Experimentally, it seems that voices from the training set
produce more realistic outputs then those outside of the training set. Any voice prepended with "train" came from the
training set.

**Adding a new voice:**
To add new voices to Tortoise, you will need to do the following:

1. Gather audio clips of your speaker(s). Good sources are YouTube interviews (you can use youtube-dl to fetch the audio), audiobooks or podcasts. Guidelines for good clips are in the next section.
2. Cut your clips into ~10 second segments. You want at least 3 clips. More is better, but I only experimented with up to 5 in my testing.
3. Save the clips as a WAV file with floating point format and a 22,050 sample rate.
4. Create a subdirectory in voices/
5. Put your clips in that subdirectory.
6. Run tortoise utilities with --voice=<your_subdirectory_name>.

**Picking good reference clips:**
As mentioned above, your reference clips have a profound impact on the output of Tortoise. Following are some tips for picking
good clips:

1. Avoid clips with background music, noise or reverb. These clips were removed from the training dataset. Tortoise is unlikely to do well with them.
2. Avoid speeches. These generally have distortion caused by the amplification system.
3. Avoid clips from phone calls.
4. Avoid clips that have excessive stuttering, stammering or words like "uh" or "like" in them.
5. Try to find clips that are spoken in such a way as you wish your output to sound like. For example, if you want to hear your target voice read an audiobook, try to find clips of them reading a book.
6. The text being spoken in the clips does not matter, but diverse text does seem to perform better.

**Advanced Usage:**
Generation settings and Prompt engineering are available through API. Playing with voice latent can be done using scripts like `get_conditioning_latents.py` and by manipulating conditioning latent files.

**Tortoise-detect:**
Out of concerns that this model might be misused, a classifier is built to tell the likelihood that an audio clip came from Tortoise.

```commandline
python tortoise/is_this_from_tortoise.py --clip=<path_to_suspicious_audio_file>
```

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Quality and content of reference clips, background noise in reference clips, speech type in reference clips, audio source of reference clips, speech disfluencies in reference clips, and text diversity.

### Evaluation factors:
Voice realism and intelligibility.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
Voice realism and intelligibility.

### Decision thresholds:
Not available.

### Variation approaches:
Performance metrics were evaluated on thousands of clips generated with various permutations of settings.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
For Tortoise-detect classifier: contents of the `results/` and `voices/` folders in this repo.
For TTS model: example outputs are available on [this page](http://nonint.com/static/tortoise_v2_examples.html) and in the `voices/` folder.

### Motivation:
For Tortoise-detect classifier: to test the accuracy of the classifier.
For TTS model: to showcase the capabilities of the model.

### Preprocessing:
Not available.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
~50k hours of speech data, primarily audiobooks.

### Motivation:
Audiobooks were chosen as the primary training data for text-to-speech.

### Preprocessing:
Clips with background music, noise or reverb were removed from the training dataset.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
Tortoise-detect classifier has 100% accuracy on the contents of the `results/` and `voices/` folders in this repo. No quantitative results are provided for TTS model factors.

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
8 RTX 3090s were used for training.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

Potential misuse of voice-cloning text-to-speech system is a concern. The model is primarily good at reading books and speaking poetry, and less effective for other forms of speech. It was trained on a dataset without public figures' voices, which reduces the risk of mimicking them convincingly. A separate classifier model (`tortoise-detect`) is released to detect audios generated by Tortoise. The diversity of the model is limited by the training dataset, which primarily consists of audiobooks, potentially leading to poor performance for minorities or people with strong accents. Releasing the model openly allows everyone to be aware of the capabilities of ML in this domain.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
- Tortoise is slow, generating a medium-sized sentence every 2 minutes on a K80.
- Tortoise may occasionally produce flawed outputs.
- Tortoise may perform poorly for voices of minorities or people with strong accents due to dataset limitations.
- Evaluation datasets may have gaps, such as missing demographic groups.
- The model's potential could be significantly greater with increased scale, as current models are smaller than models like GPT-2 or DALLE.

### Recommendations:
- Provide feedback on your usage and findings.
- Report any neat applications or undocumented features discovered.
- Ethical organizations with computational resources are encouraged to collaborate on scaling the model.

---

## Additional Information

### What's in a name?
The name "Tortoise" is tongue-in-cheek, referring to the model's slow generation speed. The naming convention for speech-related repos is Mojave desert flora and fauna.

### New features
#### v2.1; 2022/5/2
- Added ability to produce totally random voices.
- Added ability to download voice conditioning latent via a script, and then use a user-provided conditioning latent.
- Added ability to use your own pretrained models.
- Refactored directory structures.
- Performance improvements & bug fixes.

### Acknowledgements
The project acknowledges and credits:
- Hugging Face for the GPT model and generate API.
- Ramesh et al (DALLE paper authors) for inspiration.
- Nichol and Dhariwal (diffusion model code authors).
- Jang et al (univnet vocoder developers).
- lucidrains for open source pytorch models.
- Patrick von Platen for wav2vec setup guides.

### Notice
Tortoise was built entirely by the developer using personal hardware, without employer involvement.