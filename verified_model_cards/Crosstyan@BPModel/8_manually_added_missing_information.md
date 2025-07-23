## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
BPModel is an experimental Stable Diffusion model based on [ACertainty](https://huggingface.co/JosephusCheung/ACertainty) from [Joseph Cheung](https://huggingface.co/JosephusCheung). The dataset is public in [Crosstyan/BPDataset](https://huggingface.co/datasets/Crosstyan/BPDataset).

### Model date:
**2023-01-02:** Update.
**2023-01-06:** Update.

### Model version:
Check out [bp_mk3.safetensors](bp_mk3.safetensors) and [bp_mk5.safetensors](bp_mk5.safetensors). Checkout [NMFSAN](NMFSAN/README.md) for a new model trained with custom embeddings.

### Model type:
BPModel is an experimental Stable Diffusion model. It is trained with a base resolution (`base_res`) of 768 and 1024.

### Training details:
Trained with 5k high quality images from [Sankaku Complex](https://chan.sankakucomplex.com) with annotations. The dataset is public in [Crosstyan/BPDataset](https://huggingface.co/datasets/Crosstyan/BPDataset). 10 V100 GPU hours were spent on training 30 epochs with a resolution of 512, while 60 V100 GPU hours were spent on training 30 epochs with a resolution of 768. An additional 100 V100 GPU hours were also spent on training a model with a resolution of 1024, although **ONLY** 10 epochs were run. [Mikubill/naifu-diffusion](https://github.com/Mikubill/naifu-diffusion) is used as training script.

### Paper or other resource for more information:
- [ACertainty](https://huggingface.co/JosephusCheung/ACertainty)
- [Joseph Cheung](https://huggingface.co/JosephusCheung)
- [Sankaku Complex](https://chan.sankakucomplex.com)
- [Crosstyan/BPDataset](https://huggingface.co/datasets/Crosstyan/BPDataset)
- [Mikubill/naifu-diffusion](https://github.com/Mikubill/naifu-diffusion)
- [CCRcmcpe/scal-sdt](https://github.com/CCRcmcpe/scal-sdt)
- [SCAL-SDT Wiki](https://github.com/CCRcmcpe/scal-sdt/wiki#what-you-should-expect)
- [Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models](https://arxiv.org/abs/2212.03860)
- [JosephusCheung/ACertainThing](https://huggingface.co/JosephusCheung/ACertainThing)
- [sks?](https://www.reddit.com/r/StableDiffusion/comments/yju5ks/from_one_of_the_original_dreambooth_authors_stop/)
- [OFA](https://github.com/OFA-Sys/OFA)
- [convnext-tagger](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger)
- [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [chatGPT](https://openai.com/blog/chatgpt/)
- [CreativeML OpenRAIL-M license](https://huggingface.co/spaces/CompVis/stable-diffusion-license/blob/main/license.txt)
- [xformers](https://github.com/facebookresearch/xformers)

### Citation details:
Not available.

### License:
This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage. The CreativeML OpenRAIL License specifies:

1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content
2. The authors claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license
3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully) Please read the full license [here](https://huggingface.co/spaces/CompVis/stable-diffusion-license/blob/main/license.txt)

### Contact:
Not available.

---

## Intended Use
This section outlines the intended applications of the model.

### Primary intended uses:
BPModel is an experimental Stable Diffusion model for anime style image generation. It is trained to generate high resolution images (768, 1024).

### Primary intended users:
It is for Stable Diffusion users who are interested in anime style models and high resolution image generation.

### Out-of-scope uses:
SD cannot generate human body properly, like generating 6 fingers on one hand.

---

## How to Use
This section outlines how to use the model.

The [`bp_1024_e10.ckpt`](bp_1024_e10.ckpt) doesn't include any VAE and you should using other popular VAE in the community when using with [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui). Use [`bp_1024_with_vae_te.ckpt`](bp_1024_with_vae_te.ckpt) if you don't have VAE and text encoder with you. If you want to continue training, use [`bp_1024_e10_ema.ckpt`](bp_1024_e10_ema.ckpt).

For better performance, it is strongly recommended to use Clip skip (CLIP stop at last layers) 2. It's also recommended to use turn on "`Upscale latent space image when doing hires. fix`" in the settings of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

**Example Prompts:**

```txt
by (fkey:1) (shion:0.4) [sketch:0.75] (closed mouth expressionless:1) cat ears nekomimi 1girl, wearing a white sailor uniform with a short skirt and white pantyhose standing on the deck of a yacht, cowboy shot, and the sun setting behind her in the background, light particle, bokeh
Negative prompt: lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, worst quality, low quality, normal quality, lipstick, 2koma, 3koma, dutch angle, blush, from behind
Steps: 28, Sampler: Euler a, CFG scale: 12, Seed: 4236324744, Size: 960x1600, Model hash: 855959a4, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, First pass size: 0x0
```

```txt
1girl in black serafuku standing in a field solo, food, fruit, lemon, bubble, planet, moon, orange \(fruit\), lemon slice, leaf, fish, orange slice, by (tabi:1.25), spot color, looking at viewer, closeup cowboy shot
Negative prompt: (bad:0.81), (comic:0.81), (cropped:0.81), (error:0.81), (extra:0.81), (low:0.81), (lowres:0.81), (speech:0.81), (worst:0.81), (blush:0.9), 2koma, 3koma, 4koma, collage, lipstick
Steps: 18, Sampler: DDIM, CFG scale: 7, Seed: 2017390109, Size: 768x1600, Model hash: fed5b383, Batch size: 4, Batch pos: 1, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, First pass size: 0x0
```

```txt
[sketch:0.75] [(oil painting:0.5)::0.75] by (fuzichoco:0.8) shion (fkey:0.9), fang solo cat ears nekomimi girl with multicolor streaked messy hair blue [black|blue] long hair bangs blue eyes in blue sailor collar school uniform serafuku short sleeves hand on own cheek hand on own face sitting, upper body, strawberry sweets ice cream food fruit spoon orange parfait
Negative prompt: (bad:0.98), (normal:0.98), (comic:0.81), (cropped:0.81), (error:0.81), (extra:0.81), (low:0.81), (lowres:1), (speech:0.81), (worst:0.81), 2koma, 3koma, 4koma, collage, lipstick
Steps: 40, Sampler: Euler a, CFG scale: 8, Seed: 910302581, Size: 960x1600, Model hash: fed5b383, Batch size: 4, Batch pos: 2, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, First pass size: 0x0
```

```txt
(best:0.7), highly detailed,1girl,upper body,beautiful detailed eyes, medium_breasts, long hair,grey hair, grey eyes, curly hair, bangs,empty eyes,expressionless,twintails, beautiful detailed sky, beautiful detailed water, [cinematic lighting:0.6], upper body, school uniform,black ribbon,light smile
Negative prompt: (bad:0.98), (normal:0.98), (comic:0.81), (cropped:0.81), (error:0.81), (extra:0.81), (low:0.81), (lowres:1), (speech:0.81), (worst:0.81), 2koma, 3koma, 4koma, collage, lipstick
Steps: 40, Sampler: Euler, CFG scale: 8.5, Seed: 2311603025, Size: 960x1600, Model hash: fed5b383, Batch size: 4, Batch pos: 3, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, First pass size: 0x0
```

```txt
by [shion (fkey:0.9):momoko \(momopoco\):0.15], fang solo cat ears nekomimi girl with multicolor streaked messy hair blue [black|blue] long hair bangs blue eyes in blue sailor collar school uniform serafuku short sleeves hand on own cheek (middle finger:1.1) sitting, upper body, strawberry sweets ice cream food fruit spoon orange parfait
Negative prompt: (bad:0.98), (normal:0.98), (comic:0.81), (cropped:0.81), (error:0.81), (extra:0.81), (low:0.81), (lowres:1), (speech:0.81), (worst:0.81), 2koma, 3koma, 4koma, collage, lipstick
Steps: 40, Sampler: Euler a, CFG scale: 8, Seed: 2496891010, Size: 960x1600, Model hash: fed5b383, Batch size: 4, Batch pos: 1, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, First pass size: 0x0
```

```txt
by [shion (fkey:0.9):momoko \(momopoco\):0.55], closed mouth fang solo cat ears nekomimi girl with multicolor streaked messy hair blue [black|blue] long hair bangs blue eyes in blue sailor collar school uniform serafuku short sleeves (middle finger:1.1) sitting, upper body, strawberry sweets ice cream food fruit spoon orange parfait
Negative prompt: (bad:0.98), (normal:0.98), (comic:0.81), (cropped:0.81), (extra:0.81), (low:0.81), (lowres:1), (speech:0.81), (worst:0.81), 2koma, 3koma, 4koma, collage, lipstick, (chibi:0.8)
Steps: 40, Sampler: Euler a, CFG scale: 8, Seed: 2668993375, Size: 960x1600, Model hash: fed5b383, Batch size: 4, Batch pos: 3, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, First pass size: 0x0
```

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
Training resolution (512, 768, 1024). Dataset from Sankaku Complex. Base model ACertainty. Training script Mikubill/naifu-diffusion. Aspect ratio bucket configuration.

### Evaluation factors:
Not available.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
Not available.

### Decision thresholds:
Not available.

### Variation approaches:
Not available.

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
Cherry picked samples are provided in the model card.

### Motivation:
Not available.

### Preprocessing:
Not available.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
Trained with BPDataset from [Sankaku Complex](https://chan.sankakucomplex.com), which contains 5k high quality images with annotations. The dataset is publicly available at [Crosstyan/BPDataset](https://huggingface.co/datasets/Crosstyan/BPDataset).

### Motivation:
The dataset was chosen to train a Stable Diffusion model with high base resolution (768, 1024).

### Preprocessing:
Pure combination of tags from Sankaku Complex annotations was used as preprocessing.

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
Not available.

### Deploying Requirements:
Not available.

### Training or Fine-tuning Requirements:
Training was done using V100 GPUs. 10 V100 GPU hours for 512 resolution, 60 V100 GPU hours for 768 resolution, and 100 V100 GPU hours for 1024 resolution.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

Sensitive data (NSFW content) from [Sankaku Complex](https://chan.sankakucomplex.com) was used in training. Potential risks include generating illegal or harmful outputs or content. Risk mitigation strategies include the use restrictions outlined in the license.

---

## Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
The limitation described in [SCAL-SDT Wiki](https://github.com/CCRcmcpe/scal-sdt/wiki#what-you-should-expect) is still applied. SD cannot generate human body properly. BPModel is prone to overfit. Language drift problem and artist name tokenization issues may exist.

### Recommendations:
It is recommended to use Clip skip 2 and turn on "`Upscale latent space image when doing hires. fix`" in the settings of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) for better performance.

---

## Additional Information

![BPModel](images/BPModel.png)
![car](images/00976-3769766671_20221226155509.png)
![building](images/00167-4082916932_20230102081230.png)
![sunset](images/00121-4236324744_20230102073128.png)
![orange](images/00317-2017390109_20221220015645.png)
![icecream](images/00748-910302581_20221220073123.png)
![girl](images/01101-2311603025_20221220161819.png)
![middle_f](images/00819-2496891010_20221220080243.png)
![middle_f_2](images/01073-2668993375_20221220100952.png)

The BPModel, a Stable Diffusion model you may love or hate.

I won't feed any AI generated image to the model even it might outlaw the model from being used in some countries.

The training of a high resolution model requires a significant amount of GPU hours and can be costly. The results of the training on the 1024 resolution model did not show a significant improvement compared to the 768 resolution model, and the resource demands, achieving a batch size of 1 on a V100 with 32G VRAM, were high. However, training on the 768 resolution did yield better results than training on the 512 resolution, and it is worth considering as an option. It is worth noting that Stable Diffusion 2.x also chose to train on a 768 resolution model. However, it may be more efficient to start with training on a 512 resolution model due to the slower training process and the need for additional prior knowledge to speed up the training process when working with a 768 resolution.

The configuration for 1024 and 768 resolution with aspect ratio bucket is

```yaml
# 768
arb:
  enabled: true
  debug: false
  base_res: [768, 768]
  max_size: [1152, 768]
  divisible: 64
  max_ar_error: 4
  min_dim: 512
  dim_limit: 1792

# 1024
arb:
  enabled: true
  debug: false
  base_res: [1024, 1024]
  max_size: [1536, 1024]
  divisible: 64
  max_ar_error: 4
  min_dim: 960
  dim_limit: 2389
```

BPModel can generate [more proper kitty cat](https://twitter.com/crosstyan/status/1606026536246685696) (if you know what I mean) than other anime model, but it's still not perfect. As results presented in [Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models](https://arxiv.org/abs/2212.03860), the copy and paste effect is still observed.

Anything v3™ has been proven to be the most popular anime model in the community, but it's not perfect either as described in [JosephusCheung/ACertainThing](https://huggingface.co/JosephusCheung/ACertainThing)

> It does not always stay true to your prompts; it adds irrelevant details, and sometimes these details are highly homogenized. 

BPModel, which has been fine-tuned on a relatively small dataset, is prone to overfit inherently. This is not surprising given the size of the dataset, but the strong prior knowledge of ACertainty (full Danbooru) and Stable Diffusion (LAION) helps to minimize the impact of overfitting. However I believe it would perform better than some artist style DreamBooth model which only train with a few hundred images or even less. I also oppose changing style by merging model since You could apply different style by training with proper captions and prompting.

Besides some of images in my dataset have the artist name in the caption, however some artist name will be misinterpreted by CLIP when tokenizing. For example, *as109* will be tokenized as `[as, 1, 0, 9]` and
*fuzichoco* will become `[fu, z, ic, hoco]`. Romanized Japanese suffers from the problem a lot and I don't have a good solution to fix it other than changing the artist name in the caption, which is time consuming and you can't promise the token you choose is unique enough. [Remember the sks?](https://www.reddit.com/r/StableDiffusion/comments/yju5ks/from_one_of_the_original_dreambooth_authors_stop/)

Language drift problem is still exist. There's nothing I can do unless I can find a way to generate better caption or caption the image manually. [OFA](https://github.com/OFA-Sys/OFA) combined with [convnext-tagger](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger) could provide a better result for SFW content. However fine tune is necessary for NSFW content, which I don't think anyone would like to do. (Could Unstable Diffusion give us surprise?)

I were using [xformers](https://github.com/facebookresearch/xformers) when generating the outputs from the example prompts and it might yield slight different result even with the same seed (welcome to the non deterministic field). "`Upscale latent space image when doing hires. fix`" is enabled also.

more samples can be found in [images](images/00976-3769766671_20221226155509.png) folder.

If you don't use a VAE while using `bp_1024_e10.ckpt`, you would see the LaTeNt SpAcE!

EMA weight is not included in [`bp_1024_with_vae_te.ckpt`] and it's fp16.

[`bp_1024_e10_ema.ckpt`](bp_1024_e10_ema.ckpt) contains the ema unet weight and is in fp32 precision.

"`Upscale latent space image when doing hires. fix`" in the settings of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
adds intricate details when using `Highres. fix`.

I asked the [chatGPT](https://openai.com/blog/chatgpt/) what the proper explanation of abbreviation BP could be.

```txt
Here are a few more ideas for creative interpretations of the abbreviation "BP":

- Brightest Point - This could refer to a moment of exceptional brilliance or clarity.
- Brainpower - the abbreviation refers to something that requires a lot of mental effort or intelligence to accomplish.
- Bespoke Partition - A custom made section that separates two distinct areas.
- Bukkake Picchi - A Japanese style of rice dish.
- Bokeh Picker - A traditional Japanese photography technique that involves selecting points of light from a digital image.
- Bipolarity - Two distinct and opposing elements or perspectives.

Note that "BP" is often used as an abbreviation for "blood pressure," so it is important to context to determine the most appropriate interpretation of the abbreviation.
```

Personally, I would call it "Big Pot".

*I don't think other model can do that.*

**2023-01-02:** I wasted more GPU hours to train it a little bit more overfitting. Check out [bp_mk3.safetensors](bp_mk3.safetensors) and [bp_mk5.safetensors](bp_mk5.safetensors). Prepare yourself own VAE! Update your WebUI if you can't load [safetensors](https://github.com/huggingface/safetensors). Adds lots of samples in `images` folder!

**2023-01-06:** Checkout [NMFSAN](NMFSAN/README.md) for a new model trained with custom embeddings.
