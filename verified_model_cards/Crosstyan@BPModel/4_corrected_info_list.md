# BPModel
- ![BPModel](images/BPModel.png)

## Update
- **2023-01-02:** More GPU hours were wasted to train the model a little bit more overfitting.
- Check out [bp_mk3.safetensors](bp_mk3.safetensors).
- Check out [bp_mk5.safetensors](bp_mk5.safetensors).
- Prepare your own VAE.
- Update your WebUI if you can't load [safetensors](https://github.com/huggingface/safetensors).
- Lots of samples are added in the `images` folder.
- **2023-01-06:** Check out [NMFSAN](NMFSAN/README.md) for a new model trained with custom embeddings.

## Introduction
- ![car](images/00976-3769766671_20221226155509.png)
- BPModel is an experimental Stable Diffusion model.
- BPModel is based on [ACertainty](https://huggingface.co/JosephusCheung/ACertainty).
- ACertainty was created by [Joseph Cheung](https://huggingface.co/JosephusCheung).
- There are many Stable Diffusion models, especially anime style models.
- This model exists because there are no Stable Diffusion models trained with base resolution (`base_res`) 768 or even 1024 before.
- BPModel is a Stable Diffusion model that you may love or hate.
- The model was trained with 5k high-quality images that suit the author's taste.
- The dataset is from [Sankaku Complex](https://chan.sankakucomplex.com) with annotations.
- The dataset is public in [Crosstyan/BPDataset](https://huggingface.co/datasets/Crosstyan/BPDataset) for full disclosure.
- Pure combination of tags may not optimally describe the image.
- The author will not feed any AI-generated image to the model.
- Training the model with AI-generated image might outlaw the model from being used in some countries.
- Training a high-resolution model requires a significant amount of GPU hours and can be costly.
- 10 V100 GPU hours were spent on training 30 epochs with a resolution of 512.
- 60 V100 GPU hours were spent on training 30 epochs with a resolution of 768.
- An additional 100 V100 GPU hours were spent on training a model with a resolution of 1024.
- Only 10 epochs were run for the 1024 resolution model.
- The results of the training on the 1024 resolution model did not show significant improvement compared to the 768 resolution model.
- The resource demands for the 1024 resolution model were high, achieving a batch size of 1 on a V100 with 32G VRAM.
- Training on the 768 resolution yielded better results than training on the 512 resolution.
- It is worth considering training on a 768 resolution model.
- Stable Diffusion 2.x also chose to train on a 768 resolution model.
- It may be more efficient to start training on a 512 resolution model due to the slower training process.
- Additional prior knowledge is needed to speed up the training process when working with a 768 resolution.
- [Mikubill/naifu-diffusion](https://github.com/Mikubill/naifu-diffusion) is used as the training script.
- The author recommends checking out [CCRcmcpe/scal-sdt](https://github.com/CCRcmcpe/scal-sdt).
- The configuration for 1024 and 768 resolution with aspect ratio bucket is presented.

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

## Limitation
- ![building](images/00167-4082916932_20230102081230.png)
- The limitation described in [SCAL-SDT Wiki](https://github.com/CCRcmcpe/scal-sdt/wiki#what-you-should-expect) still applies.
- SD cannot generate human bodies properly, such as generating 6 fingers on one hand.
- BPModel can generate [more proper kitty cat](https://twitter.com/crosstyan/status/1606026536246685696) than other anime models.
- The model is still not perfect.
- The copy and paste effect is still observed as presented in [Diffusion Art or Digital Forgery? Investigating Data Replication in Diffusion Models](https://arxiv.org/abs/2212.03860).
- Anything v3™ has been proven to be the most popular anime model in the community.
- Anything v3™ is not perfect either as described in [JosephusCheung/ACertainThing](https://huggingface.co/JosephusCheung/ACertainThing).
- Anything v3™ does not always stay true to prompts; it adds irrelevant details.
- Sometimes the details added by Anything v3™ are highly homogenized.
- BPModel, which has been fine-tuned on a relatively small dataset, is prone to overfit inherently.
- The size of the dataset contributes to the overfitting.
- The strong prior knowledge of ACertainty (full Danbooru) and Stable Diffusion (LAION) helps minimize the impact of overfitting.
- BPModel is believed to perform better than some artist style DreamBooth models trained with only a few hundred images or less.
- The author opposes changing style by merging models.
- Different styles can be applied by training with proper captions and prompting.
- Some images in the dataset have the artist name in the caption.
- Some artist names will be misinterpreted by CLIP when tokenizing.
- For example, *as109* will be tokenized as `[as, 1, 0, 9]`.
- *fuzichoco* will become `[fu, z, ic, hoco]`.
- Romanized Japanese suffers from tokenization problems.
- The author does not have a good solution to fix tokenization issues other than changing the artist name in the caption.
- Changing artist names in captions is time-consuming.
- There is no guarantee that the token chosen is unique enough.
- [Remember the sks?](https://www.reddit.com/r/StableDiffusion/comments/yju5ks/from_one_of_the_original_dreambooth_authors_stop/)
- Language drift problem still exists.
- There is nothing the author can do unless a way to generate better captions or manually caption the images is found.
- [OFA](https://github.com/OFA-Sys/OFA) combined with [convnext-tagger](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger) could provide better results for SFW content.
- Fine-tuning is necessary for NSFW content, which the author does not think anyone would like to do.
- The author questions if Unstable Diffusion could provide a surprise.

## Cherry Picked Samples
- Here are some **cherry picked** samples.
- The author was using [xformers](https://github.com/facebookresearch/xformers) when generating these samples.
- The use of xformers might yield slightly different results even with the same seed.
- "`Upscale latent space image when doing hires. fix`" is enabled.

![sunset](images/00121-4236324744_20230102073128.png)

```txt
by (fkey:1) (shion:0.4) [sketch:0.75] (closed mouth expressionless:1) cat ears nekomimi 1girl, wearing a white sailor uniform with a short skirt and white pantyhose standing on the deck of a yacht, cowboy shot, and the sun setting behind her in the background, light particle, bokeh
Negative prompt: lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, worst quality, low quality, normal quality, lipstick, 2koma, 3koma, dutch angle, blush, from behind
Steps: 28, Sampler: Euler a, CFG scale: 12, Seed: 4236324744, Size: 960x1600, Model hash: 855959a4, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, First pass size: 0x0
```

![orange](images/00317-2017390109_20221220015645.png)

```txt
1girl in black serafuku standing in a field solo, food, fruit, lemon, bubble, planet, moon, orange \(fruit\), lemon slice, leaf, fish, orange slice, by (tabi:1.25), spot color, looking at viewer, closeup cowboy shot
Negative prompt: (bad:0.81), (comic:0.81), (cropped:0.81), (error:0.81), (extra:0.81), (low:0.81), (lowres:0.81), (speech:0.81), (worst:0.81), (blush:0.9), 2koma, 3koma, 4koma, collage, lipstick
Steps: 18, Sampler: DDIM, CFG scale: 7, Seed: 2017390109, Size: 768x1600, Model hash: fed5b383, Batch size: 4, Batch pos: 1, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, First pass size: 0x0
```

![icecream](images/00748-910302581_20221220073123.png)

```txt
[sketch:0.75] [(oil painting:0.5)::0.75] by (fuzichoco:0.8) shion (fkey:0.9), fang solo cat ears nekomimi girl with multicolor streaked messy hair blue [black|blue] long hair bangs blue eyes in blue sailor collar school uniform serafuku short sleeves hand on own cheek hand on own face sitting, upper body, strawberry sweets ice cream food fruit spoon orange parfait
Negative prompt: (bad:0.98), (normal:0.98), (comic:0.81), (cropped:0.81), (error:0.81), (extra:0.81), (low:0.81), (lowres:1), (speech:0.81), (worst:0.81), 2koma, 3koma, 4koma, collage, lipstick
Steps: 40, Sampler: Euler a, CFG scale: 8, Seed: 910302581, Size: 960x1600, Model hash: fed5b383, Batch size: 4, Batch pos: 2, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, First pass size: 0x0
```

![girl](images/01101-2311603025_20221220161819.png)

```txt
(best:0.7), highly detailed,1girl,upper body,beautiful detailed eyes, medium_breasts, long hair,grey hair, grey eyes, curly hair, bangs,empty eyes,expressionless,twintails, beautiful detailed sky, beautiful detailed water, [cinematic lighting:0.6], upper body, school uniform,black ribbon,light smile
Negative prompt: (bad:0.98), (normal:0.98), (comic:0.81), (cropped:0.81), (error:0.81), (extra:0.81), (low:0.81), (lowres:1), (speech:0.81), (worst:0.81), 2koma, 3koma, 4koma, collage, lipstick
Steps: 40, Sampler: Euler, CFG scale: 8.5, Seed: 2311603025, Size: 960x1600, Model hash: fed5b383, Batch size: 4, Batch pos: 3, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, First pass size: 0x0
```

*I don't think other models can do that.*

![middle_f](images/00819-2496891010_20221220080243.png)

```txt
by [shion (fkey:0.9):momoko \(momopoco\):0.15], fang solo cat ears nekomimi girl with multicolor streaked messy hair blue [black|blue] long hair bangs blue eyes in blue sailor collar school uniform serafuku short sleeves hand on own cheek (middle finger:1.1) sitting, upper body, strawberry sweets ice cream food fruit spoon orange parfait
Negative prompt: (bad:0.98), (normal:0.98), (comic:0.81), (cropped:0.81), (error:0.81), (extra:0.81), (low:0.81), (lowres:1), (speech:0.81), (worst:0.81), 2koma, 3koma, 4koma, collage, lipstick
Steps: 40, Sampler: Euler a, CFG scale: 8, Seed: 2496891010, Size: 960x1600, Model hash: fed5b383, Batch size: 4, Batch pos: 1, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, First pass size: 0x0
```

![middle_f_2](images/01073-2668993375_20221220100952.png)

```txt
by [shion (fkey:0.9):momoko \(momopoco\):0.55], closed mouth fang solo cat ears nekomimi girl with multicolor streaked messy hair blue [black|blue] long hair bangs blue eyes in blue sailor collar school uniform serafuku short sleeves (middle finger:1.1) sitting, upper body, strawberry sweets ice cream food fruit spoon orange parfait
Negative prompt: (bad:0.98), (normal:0.98), (comic:0.81), (cropped:0.81), (error:0.81), (extra:0.81), (low:0.81), (lowres:1), (speech:0.81), (worst:0.81), 2koma, 3koma, 4koma, collage, lipstick, (chibi:0.8)
Steps: 40, Sampler: Euler a, CFG scale: 8, Seed: 2668993375, Size: 960x1600, Model hash: fed5b383, Batch size: 4, Batch pos: 3, Denoising strength: 0.7, Clip skip: 2, ENSD: 31337, First pass size: 0x0
```

- More samples can be found in the [images](images/00976-3769766671_20221226155509.png) folder.

## Usage
- The [`bp_1024_e10.ckpt`](bp_1024_e10.ckpt) doesn't include any VAE.
- You should use other popular VAE in the community when using with [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- If you don't use a VAE, you would see the LaTeNt SpAcE!
- Use [`bp_1024_with_vae_te.ckpt`](bp_1024_with_vae_te.ckpt) if you don't have VAE and text encoder.
- EMA weight is not included in [`bp_1024_with_vae_te.ckpt`] and it's fp16.
- If you want to continue training, use [`bp_1024_e10_ema.ckpt`](bp_1024_e10_ema.ckpt).
- [`bp_1024_e10_ema.ckpt`](bp_1024_e10_ema.ckpt) contains the ema unet weight and is in fp32 precision.
- For better performance, it is strongly recommended to use Clip skip (CLIP stop at last layers) 2.
- It is also recommended to turn on "`Upscale latent space image when doing hires. fix`" in the settings of [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- This setting adds intricate details when using `Highres. fix`.

## About the Model Name
- The author asked [chatGPT](https://openai.com/blog/chatgpt/) for the proper explanation of the abbreviation BP.
  
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
- Personally, the author would call it "Big Pot".

## License
- This model is open access and available to all.
- The model has a CreativeML OpenRAIL-M license further specifying rights and usage.
- The CreativeML OpenRAIL License specifies:
  1. You can't use the model to deliberately produce or share illegal or harmful outputs or content.
  2. The authors claim no rights on the outputs you generate.
  3. You are free to use the outputs and are accountable for their use, which must not go against the provisions set in the license.
  4. You may re-distribute the weights and use the model commercially and/or as a service.
  5. If you do, you must include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M with all your users.
- Please read the full license [here](https://huggingface.co/spaces/CompVis/stable-diffusion-license/blob/main/license.txt).