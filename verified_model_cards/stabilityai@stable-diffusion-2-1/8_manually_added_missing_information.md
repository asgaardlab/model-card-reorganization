## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
Robin Rombach, Patrick Esser

### Model date:
Not available.

### Model version:
v2-1

### Model type:
Diffusion-based text-to-image generation model. It is a Latent Diffusion Model that uses a fixed, pretrained text encoder (OpenCLIP-ViT/H).

### Training details:
This `stable-diffusion-2-1` model is fine-tuned from [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) (`768-v-ema.ckpt`) with an additional 55k steps on the same dataset (with `punsafe=0.1`), and then fine-tuned for another 155k extra steps with `punsafe=0.98`.

Stable Diffusion v2 is a latent diffusion model which combines an autoencoder with a diffusion model that is trained in the latent space of the autoencoder. During training:
- Images are encoded through an encoder, which turns images into latent representations. The autoencoder uses a relative downsampling factor of 8 and maps images of shape H x W x 3 to latents of shape H/f x W/f x 4
- Text prompts are encoded through the OpenCLIP-ViT/H text-encoder.
- The output of the text encoder is fed into the UNet backbone of the latent diffusion model via cross-attention.
- The loss is a reconstruction objective between the noise that was added to the latent and the prediction made by the UNet. We also use the so-called _v-objective_, see https://arxiv.org/abs/2202.00512.

We currently provide the following checkpoints:

- `512-base-ema.ckpt`: 550k steps at resolution `256x256` on a subset of [LAION-5B](https://laion.ai/blog/laion-5b/) filtered for explicit pornographic material, using the [LAION-NSFW classifier](https://github.com/LAION-AI/CLIP-based-NSFW-Detector) with `punsafe=0.1` and an [aesthetic score](https://github.com/christophschuhmann/improved-aesthetic-predictor) >= `4.5`.
  850k steps at resolution `512x512` on the same dataset with resolution `>= 512x512`.
- `768-v-ema.ckpt`: Resumed from `512-base-ema.ckpt` and trained for 150k steps using a [v-objective](https://arxiv.org/abs/2202.00512) on the same dataset. Resumed for another 140k steps on a `768x768` subset of our dataset.
- `512-depth-ema.ckpt`: Resumed from `512-base-ema.ckpt` and finetuned for 200k steps. Added an extra input channel to process the (relative) depth prediction produced by [MiDaS](https://github.com/isl-org/MiDaS) (`dpt_hybrid`) which is used as an additional conditioning.
The additional input channels of the U-Net which process this extra information were zero-initialized.
- `512-inpainting-ema.ckpt`: Resumed from `512-base-ema.ckpt` and trained for another 200k steps. Follows the mask-generation strategy presented in [LAMA](https://github.com/saic-mdal/lama) which, in combination with the latent VAE representations of the masked image, are used as an additional conditioning.
The additional input channels of the U-Net which process this extra information were zero-initialized. The same strategy was used to train the [1.5-inpainting checkpoint](https://huggingface.co/runwayml/stable-diffusion-inpainting).
- `x4-upscaling-ema.ckpt`: Trained for 1.25M steps on a 10M subset of LAION containing images `>2048x2048`. The model was trained on crops of size `512x512` and is a text-guided [latent upscaling diffusion model](https://arxiv.org/abs/2112.10752).
In addition to the textual input, it receives a `noise_level` as an input parameter, which can be used to add noise to the low-resolution input according to a [predefined diffusion schedule](configs/stable-diffusion/x4-upscaling.yaml).

**Hardware:** 32 x 8 x A100 GPUs
- **Optimizer:** AdamW
- **Gradient Accumulations**: 1
- **Batch:** 32 x 8 x 2 x 4 = 2048
- **Learning rate:** warmup to 0.0001 for 10,000 steps and then kept constant

### Paper or other resource for more information:
- [GitHub Repository](https://github.com/Stability-AI/)
- [GitHub Repository for codebase](https://github.com/Stability-AI/stablediffusion)
- [🤗's Diffusers library](https://github.com/huggingface/diffusers)
- [Latent Diffusion Models Paper](https://arxiv.org/abs/2112.10752)
- [LAION-5B Dataset](https://laion.ai/blog/laion-5b/)
- [LAION-5B NeurIPS 2022 Paper](https://openreview.net/forum?id=M3Y74vmsMcY)
- [LAION-NSFW classifier](https://github.com/LAION-AI/CLIP-based-NSFW-Detector)
- [Improved aesthetic predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)
- [v-objective paper](https://arxiv.org/abs/2202.00512)
- [MiDaS](https://github.com/isl-org/MiDaS)
- [LAMA](https://github.com/saic-mdal/lama)
- [1.5-inpainting checkpoint](https://huggingface.co/runwayml/stable-diffusion-inpainting)
- [latent upscaling diffusion model paper](https://arxiv.org/abs/2112.10752)
- [predefined diffusion schedule](configs/stable-diffusion/x4-upscaling.yaml)
- [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute)
- [Lacoste et al. (2019) paper on carbon emissions](https://arxiv.org/abs/1910.09700)

### Citation details:
```
@InProceedings{Rombach_2022_CVPR,
    author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
    title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {10684-10695}
}
```

### License:
[CreativeML Open RAIL++-M License](https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL)

### Contact:
Not available.

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
The model is intended for research purposes only. Possible research areas and tasks include:
- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.
- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.
- Research on generative models.

### Primary intended users:
Researchers.

### Out-of-scope uses:
The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

Using the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:

- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.
- Intentionally promoting or propagating discriminatory content or harmful stereotypes.
- Impersonating individuals without their consent.
- Sexual content without consent of the people who might see it.
- Mis- and disinformation
- Representations of egregious violence and gore
- Sharing of copyrighted or licensed material in violation of its terms of use.
- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.

---

## How to Use
This section outlines how to use the model.

Using the [🤗's Diffusers library](https://github.com/huggingface/diffusers) to run Stable Diffusion 2 in a simple and efficient manner.

```bash
pip install diffusers transformers accelerate scipy safetensors
```
Running the pipeline (if you don't swap the scheduler it will run with the default DDIM, in this example we are swapping it to DPMSolverMultistepScheduler):

```python
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]

image.save("astronaut_rides_horse.png")
```

**Notes**:
- Despite not being a dependency, we highly recommend you to install [xformers](https://github.com/facebookresearch/xformers) for memory efficient attention (better performance)
- If you have low GPU RAM available, make sure to add a `pipe.enable_attention_slicing()` after sending it to `cuda` for less VRAM usage (to the cost of speed)

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
- The model was trained mainly with English captions and will not work as well in other languages.
- Texts and images from communities and cultures that use other languages are likely to be insufficiently accounted for.
- The ability of the model to generate content with non-English prompts is significantly worse than with English-language prompts.

### Evaluation factors:
Evaluated using 50 DDIM steps and 10000 random prompts from the COCO2017 validation set, evaluated at 512x512 resolution.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
Evaluations with different classifier-free guidance scales (1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0) and 50 steps DDIM sampling steps show the relative improvements of the checkpoints. Not optimized for FID scores.

### Decision thresholds:
Not available.

### Variation approaches:
Not available.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
COCO2017 validation set.

### Motivation:
Not available.

### Preprocessing:
Evaluated at 512x512 resolution.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
LAION-5B and subsets. The training data is further filtered using LAION's NSFW detector, with a "p_unsafe" score of 0.1 (conservative). The model was trained on a subset of the large-scale dataset [LAION-5B](https://laion.ai/blog/laion-5b/), which contains adult, violent and sexual content.

### Motivation:
Not available.

### Preprocessing:
The training data is filtered using LAION's NSFW detector, with a "p_unsafe" score of 0.1 (conservative).

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
![pareto](model-variants.jpg)

Evaluations with different classifier-free guidance scales (1.5, 2.0, 3.0, 4.0,
5.0, 6.0, 7.0, 8.0) and 50 steps DDIM sampling steps show the relative improvements of the checkpoints.

### Intersectional results:
Not available.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
Not available.

### Deploying Requirements:
GPU is recommended. For low GPU RAM, attention slicing can be enabled to reduce VRAM usage at the cost of speed.

### Training or Fine-tuning Requirements:
32 x 8 x A100 GPUs.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases. Stable Diffusion v2 mirrors and exacerbates biases to such a degree that viewer discretion must be advised irrespective of the input or its intent.

The model was trained on a subset of the large-scale dataset [LAION-5B](https://laion.ai/blog/laion-5b/), which contains adult, violent and sexual content. To partially mitigate this, we have filtered the dataset using LAION's NFSW detector.

The model can generate content that is cruel to individuals, demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc., and can be misused for intentionally promoting or propagating discriminatory content or harmful stereotypes, impersonating individuals without their consent, generating sexual content without consent, mis- and disinformation, and representations of egregious violence and gore.

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
- The model does not achieve perfect photorealism
- The model cannot render legible text
- The model does not perform well on more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”
- Faces and people in general may not be generated properly.
- The model was trained mainly with English captions and will not work as well in other languages.
- The autoencoding part of the model is lossy
- The model was trained on a subset of the large-scale dataset
  [LAION-5B](https://laion.ai/blog/laion-5b/), which contains adult, violent and sexual content.

### Recommendations:
- We highly recommend you to install [xformers](https://github.com/facebookresearch/xformers) for memory efficient attention (better performance).
- If you have low GPU RAM available, make sure to add a `pipe.enable_attention_slicing()` after sending it to `cuda` for less VRAM usage (to the cost of speed).
- Viewer discretion must be advised irrespective of the input or its intent due to potential biases.

---

## Additional Information
- This model card focuses on the model associated with the Stable Diffusion v2-1 model, codebase available [here](https://github.com/Stability-AI/stablediffusion).
- This `stable-diffusion-2-1` model is fine-tuned from [stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2) (`768-v-ema.ckpt`) with an additional 55k steps on the same dataset (with `punsafe=0.1`), and then fine-tuned for another 155k extra steps with `punsafe=0.98`.
- Use it with the [`stablediffusion`](https://github.com/Stability-AI/stablediffusion) repository: download the `v2-1_768-ema-pruned.ckpt` [here](https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.ckpt).
- Use it with 🧨 [`diffusers`](#examples)
- **Model Description:** This is a model that can be used to generate and modify images based on text prompts.
- _Note: The **Out-of-scope uses** section is originally taken from the [DALLE-MINI model card](https://huggingface.co/dalle-mini/dalle-mini), was used for Stable Diffusion v1, but applies in the same way to Stable Diffusion v2_.
- Stable Diffusion was primarily trained on subsets of [LAION-2B(en)](https://laion.ai/blog/laion-5b/), which consists of images that are limited to English descriptions. 
- Texts and images from communities and cultures that use other languages are likely to be insufficiently accounted for. This affects the overall output of the model, as white and western cultures are often set as the default.
- *This model card was written by: Robin Rombach, Patrick Esser and David Ha and is based on the [Stable Diffusion v1](https://github.com/CompVis/stable-diffusion/blob/main/Stable_Diffusion_v1_Model_Card.md) and [DALL-E Mini model card](https://huggingface.co/dalle-mini/dalle-mini).*

## Environmental Impact

**Stable Diffusion v1** **Estimated Emissions**
Based on that information, we estimate the following CO2 emissions using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). The hardware, runtime, cloud provider, and compute region were utilized to estimate the carbon impact.

- **Hardware Type:** A100 PCIe 40GB
- **Hours used:** 200000
- **Cloud Provider:** AWS
- **Compute Region:** US-east
- **Carbon Emitted (Power consumption x Time x Carbon produced based on location of power grid):** 15000 kg CO2 eq.

---

## Citation
    @InProceedings{Rombach_2022_CVPR,
        author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
        title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2022},
        pages     = {10684-10695}
    }

---