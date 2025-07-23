

# IF-I-XL-v1.0

DeepFloyd-IF is a pixel-based text-to-image triple-cascaded diffusion model, that can generate pictures with new state-of-the-art for photorealism and language understanding. The result is a highly efficient model that outperforms current state-of-the-art models, achieving a zero-shot FID-30K score of `6.66` on the COCO dataset.

*Inspired by* [*Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding*](https://arxiv.org/pdf/2205.11487.pdf)

![](./pics/deepfloyd_if_scheme.jpg)

## Model Details

- **Developed by:** DeepFloyd, StabilityAI
- **Model type:** pixel-based text-to-image cascaded diffusion model
- **Cascade Stage:** I
- **Num Parameters:** 4.3B
- **Language(s):** primarily English and, to a lesser extent, other Romance languages
- **License:** <span style="color:blue"><a href="https://huggingface.co/spaces/DeepFloyd/deepfloyd-if-license">DeepFloyd IF License Agreement</a></span>
- **Model Description:** DeepFloyd-IF is modular composed of frozen text mode and three pixel cascaded diffusion modules, each designed to generate images of increasing resolution: 64x64, 256x256, and 1024x1024. All stages of the model utilize a frozen text encoder based on the T5 transformer to extract text embeddings, which are then fed into a UNet architecture enhanced with cross-attention and attention-pooling 
- **Resources for more information:** [GitHub](https://github.com/deep-floyd/IF), [deepfloyd.ai](https://deepfloyd.ai), [All Links](https://linktr.ee/deepfloyd)
- **Cite as (Soon):** -

## Using with `diffusers`

IF is integrated with the 🤗 Hugging Face [🧨 diffusers library](https://github.com/huggingface/diffusers/), which is optimized to run on GPUs with as little as 14 GB of VRAM.

Before you can use IF, you need to accept its usage conditions. To do so:
1. Make sure to have a [Hugging Face account](https://huggingface.co/join) and be loggin in
2. Accept the license on the model card of [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)
3. Make sure to login locally. Install `huggingface_hub`
```sh
pip install huggingface_hub --upgrade
```

run the login function in a Python shell

```py
from huggingface_hub import login

login()
```

and enter your [Hugging Face Hub access token](https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens).

Next we install `diffusers` and dependencies:

```sh
pip install diffusers accelerate transformers safetensors sentencepiece
```

And we can now run the model locally.

By default `diffusers` makes use of [model cpu offloading](https://huggingface.co/docs/diffusers/optimization/fp16#model-offloading-for-fast-inference-and-memory-savings) to run the whole IF pipeline with as little as 14 GB of VRAM.

If you are using `torch>=2.0.0`, make sure to **remove all** `enable_xformers_memory_efficient_attention()` functions.

* **Load all stages and offload to CPU**

```py
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch

# stage 1
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_1.enable_model_cpu_offload()

# stage 2
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
stage_2.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_2.enable_model_cpu_offload()

# stage 3
safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16)
stage_3.enable_xformers_memory_efficient_attention()  # remove line if torch.__version__ >= 2.0.0
stage_3.enable_model_cpu_offload()
```

* **Retrieve Text Embeddings**

```py
prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"'

# text embeds
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
```

* **Run stage 1**

```py
generator = torch.manual_seed(0)

image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
pt_to_pil(image)[0].save("./if_stage_I.png")
```

* **Run stage 2**

```py
image = stage_2(
    image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
).images
pt_to_pil(image)[0].save("./if_stage_II.png")
```

* **Run stage 3**

```py
image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
image[0].save("./if_stage_III.png")
```

 There are multiple ways to speed up the inference time and lower the memory consumption even more with `diffusers`. To do so, please have a look at the Diffusers docs:

- 🚀 [Optimizing for inference time](https://huggingface.co/docs/diffusers/api/pipelines/if#optimizing-for-speed)
- ⚙️ [Optimizing for low memory during inference](https://huggingface.co/docs/diffusers/api/pipelines/if#optimizing-for-memory)

For more in-detail information about how to use IF, please have a look at [the IF blog post](https://huggingface.co/blog/if) and the [documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/if) 📖.

Diffusers dreambooth scripts also supports fine-tuning 🎨 [IF](https://huggingface.co/docs/diffusers/main/en/training/dreambooth#if).
With parameter efficient finetuning, you can add new concepts to IF with a single GPU and ~28 GB VRAM.

## Training

**Training Data:**

1.2B text-image pairs (based on LAION-A and few additional internal datasets)

Test/Valid parts of datasets are not used at any cascade and stage of training. Valid part of COCO helps to demonstrate "online" loss behaviour during training (to catch incident and other problems), but dataset is never used for train.

**Training Procedure:** IF-I-XL-v1.0 is a pixel-based diffusion cascade which uses T5-Encoder embeddings (hidden states) to generate 64px image. During training,

- Images are cropped to square via shifted-center-crop augmentation (randomly shift from center up to 0.1 of size) and resized to 64px using `Pillow==9.2.0` BICUBIC resampling with reducing_gap=None (it helps to avoid aliasing) and processed to tensor BxCxHxW

- Text prompts are encoded through open-sourced frozen T5-v1_1-xxl text-encoder (that completely was trained by Google team), random 10% of texts are dropped to empty string to add ability for classifier free guidance (CFG)

- The non-pooled output of the text encoder is fed into the projection (linear layer without activation) and is used in UNet backbone of the diffusion model via controlled hybrid self- and cross- attention

- Also, the output of the text encode is pooled via attention-pooling (64 heads) and is used in time embed as additional features

- Diffusion process is limited by 1000 discrete steps, with cosine beta schedule of noising image

- The loss is a reconstruction objective between the noise that was added to the image and the prediction made by the UNet

- The training process for checkpoint IF-I-XL-v1.0 has 2_420_000 steps at resolution 64x64 on all datasets, OneCycleLR policy, few-bit backward GELU activations, optimizer AdamW8bit + DeepSpeed-Zero1, fully frozen T5-Encoder

![](./pics/loss.jpg)

**Hardware:** 64 x 8 x A100 GPUs

**Optimizer:** [AdamW8bit](https://arxiv.org/abs/2110.02861) + [DeepSpeed ZeRO-1](https://www.deepspeed.ai/tutorials/zero/)

**Batch:** 3072

**Learning rate**: [one-cycle](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html) cosine strategy, warmup 10000 steps, start_lr=2e-6, max_lr=5e-5, final_lr=5e-9


![](./pics/lr.jpg)


## Evaluation Results

`FID-30K: 6.66`

![](./pics/fid30k_if.jpg)

# Uses

## Direct Use 
The model is released for research purposes. Any attempt to deploy the model in production requires not only that the LICENSE is followed but full liability over the person deploying the model.

Possible research areas and tasks include:
- Generation of artistic imagery and use in design and other artistic processes.
- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.
- Applications in educational or creative tools.
- Research on generative models.

Excluded uses are described below.

 ### Misuse, Malicious Use, and Out-of-Scope Use
_Note: This section is originally taken from the [DALLE-MINI model card](https://huggingface.co/dalle-mini/dalle-mini), was used for Stable Diffusion but applies in the same way for IF_.

The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

#### Out-of-Scope Use
The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

#### Misuse and Malicious Use
Using the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:

- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.
- Intentionally promoting or propagating discriminatory content or harmful stereotypes.
- Impersonating individuals without their consent.
- Sexual content without consent of the people who might see it.
- Mis- and disinformation
- Representations of egregious violence and gore
- Sharing of copyrighted or licensed material in violation of its terms of use.
- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.

## Limitations and Bias

### Limitations

- The model does not achieve perfect photorealism
- The model was trained mainly with English captions and will not work as well in other languages.
- The model was trained on a subset of the large-scale dataset
  [LAION-5B](https://laion.ai/blog/laion-5b/), which contains adult, violent and sexual content. To partially mitigate this, we have... (see Training section).

### Bias
While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases. 
IF was primarily trained on subsets of [LAION-2B(en)](https://laion.ai/blog/laion-5b/), 
which consists of images that are limited to English descriptions. 
Texts and images from communities and cultures that use other languages are likely to be insufficiently accounted for. 
This affects the overall output of the model, as white and western cultures are often set as the default. Further, the 
ability of the model to generate content with non-English prompts is significantly worse than with English-language prompts.
IF mirrors and exacerbates biases to such a degree that viewer discretion must be advised irrespective of the input or its intent.


## Citation (Soon)

*This model card was written by: DeepFloyd-Team and is based on the [StableDiffusion model card](https://huggingface.co/CompVis/stable-diffusion-v1-4).*