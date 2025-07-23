## Model Details
This section provides fundamental information about the model, helping stakeholders understand its context and key characteristics.

### Person or organization developing model:
Justin Pinkney. This model card is based on the [Stable Diffusion model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) and the model is hosted by Lambda Labs on Hugging Face ([lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers)). Lambda Labs also provided the GPU cloud for training ([Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud)).

### Model date:
Version 2 of the model was released after the initial version and after fixing blurriness issues. The model is now natively supported in 🤗 Diffusers (version >=0.8.0).

### Model version:
Version 2. This version of Stable Diffusion has been fine tuned from [CompVis/stable-diffusion-v1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) to accept CLIP image embedding rather than text embeddings. This version was trained in two stages and longer than the original variations model and gives better image quality and better CLIP rated similarity compared to the original version. Version 1 is also available by specifiying `revision="v1.0"`.

### Model type:
This model is a fine-tuned Stable Diffusion model. It is based on Stable Diffusion v1-3 where the text encoder has been replaced with an image encoder (ViT-L/14 image-encoder including the final projection layer to the CLIP shared embedding space). It is designed for image variation generation, similar to DALLE-2.

### Training details:
**Training Procedure**
This model is fine tuned from Stable Diffusion v1-3 where the text encoder has been replaced with an image encoder. The training procedure is the same as for Stable Diffusion except for the fact that images are encoded through a ViT-L/14 image-encoder including the final projection layer to the CLIP shared embedding space. The model was trained on LAION improved aesthetics 6plus.

- **Hardware:** 8 x A100-40GB GPUs (provided by [Lambda GPU Cloud](https://lambdalabs.com/service/gpu-cloud))
- **Optimizer:** AdamW

- **Stage 1** - Fine tune only CrossAttention layer weights from Stable Diffusion v1.4 model
  - **Steps**: 46,000
  - **Batch:** batch size=4, GPUs=8, Gradient Accumulations=4. Total batch size=128
  - **Learning rate:** warmup to 1e-5 for 10,000 steps and then kept constant

- **Stage 2** - Resume from Stage 1 training the whole unet
  - **Steps**: 50,000
  - **Batch:** batch size=4, GPUs=8, Gradient Accumulations=5. Total batch size=160
  - **Learning rate:** warmup to 1e-5 for 5,000 steps and then kept constant


Training was done using a [modified version of the original Stable Diffusion training code](https://github.com/justinpinkney/stable-diffusion).

### Paper or other resource for more information:
- Original Stable Diffusion model: [CompVis/stable-diffusion-v1-4-original](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original)
- Diffusers library: 🤗 Diffusers (natively supported)
- Lambda Diffusers repo: [Lambda Diffusers repo](https://github.com/LambdaLabsML/lambda-diffusers)
- Modified Stable Diffusion training code: [modified version of the original Stable Diffusion training code](https://github.com/justinpinkney/stable-diffusion)
- Stable Diffusion model card: [Stable Diffusion model card](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- Training dataset: [LAION-5B](https://laion.ai/blog/laion-5b/)
- Training data search: [https://rom1504.github.io/clip-retrieval/](https://rom1504.github.io/clip-retrieval/)

### Citation details:
Not available.

### License:
Not available.

### Contact:
This model card was written by Justin Pinkney.

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
- Creation of "image variations" similar to DALLE-2 using Stable Diffusion.

### Primary intended users:
The primary intended users are researchers interested in generative models.

### Out-of-scope uses:
**Out-of-Scope Use**
The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

**Misuse and Malicious Use**
Using the model to generate content that is cruel to individuals is a misuse of this model. This includes, but is not limited to:

- Generating demeaning, dehumanizing, or otherwise harmful representations of people or their environments, cultures, religions, etc.
- Intentionally promoting or propagating discriminatory content or harmful stereotypes.
- Impersonating individuals without their consent.
- Sexual content without consent of the people who might see it.
- Mis- and disinformation
- Representations of egregious violence and gore
- Sharing of copyrighted or licensed material in violation of its terms of use.
- Sharing content that is an alteration of copyrighted or licensed material in violation of its terms of use.

The model should not be used to intentionally create or disseminate images that create hostile or alienating environments for people. This includes generating images that people would foreseeably find disturbing, distressing, or offensive; or content that propagates historical or current stereotypes.

---

## How to Use
This section outlines how to use the model.

Make sure you are using a version of Diffusers >=0.8.0 (for older version see the old instructions in the "Additional Information" section).

```python
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
from torchvision import transforms

device = "cuda:0"
sd_pipe = StableDiffusionImageVariationPipeline.from_pretrained(
  "lambdalabs/sd-image-variations-diffusers",
  revision="v2.0",
  )
sd_pipe = sd_pipe.to(device)

im = Image.open("path/to/image.jpg")
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(
        (224, 224),
        interpolation=transforms.InterpolationMode.BICUBIC,
        antialias=False,
        ),
    transforms.Normalize(
      [0.48145466, 0.4578275, 0.40821073],
      [0.26862954, 0.26130258, 0.27577711]),
])
inp = tform(im).to(device).unsqueeze(0)

out = sd_pipe(inp, guidance_scale=3)
out["images"][0].save("result.jpg")
```

**Example Outputs:**

Here are examples of images generated without (top) and with (bottom) anti-aliasing during resize. (Input is [this image](https://github.com/SHI-Labs/Versatile-Diffusion/blob/master/assets/ghibli.jpg))

![](alias-montage.jpg)

![](default-montage.jpg)

Here's an example of V1 vs V2, version two was trained more carefully and for longer, see the details below. V2-top vs V1-bottom

![](v2-montage.jpg)

![](v1-montage.jpg)

Input images:

![](inputs.jpg)

v2 output for girl with a pearl earing as input (guidance scale=3)

![](earring.jpg)

---

## Factors
This section addresses variables that may impact the model's performance.

### Relevant factors:
The use of anti-aliasing during image resizing significantly impacts the model's output. The model expects resized images without anti-aliasing due to an oversight during training.

### Evaluation factors:
The author evaluates the impact of anti-aliasing during resizing and compares the output quality of Version 1 and Version 2 of the model.

---

## Metrics
This section describes how the model's performance is evaluated.

### Model performance measures:
"CLIP rated similarity" is a metric that is improved in Version 2 compared to Version 1. Visual comparison of generated images is also used to assess the quality and variations produced by the model.

### Decision thresholds:
Not available.

### Variation approaches:
The author demonstrates variations through visual comparisons of outputs from different model versions (V1 vs V2) and different preprocessing steps (with and without anti-aliasing).

---

## Evaluation Data
This section provides details about the datasets used to evaluate the model.

### Datasets:
The evaluation uses example images to visually compare the outputs of different model versions and preprocessing methods. Specific datasets used for quantitative evaluation are not explicitly mentioned. Example input images are provided in the model card (e.g., [this image](https://github.com/SHI-Labs/Versatile-Diffusion/blob/master/assets/ghibli.jpg), Girl with a Pearl Earring, and others shown in "inputs.jpg").

### Motivation:
The evaluation aims to demonstrate the improvements in Version 2, highlight the importance of correct resizing (without anti-aliasing), and showcase the image variation capabilities of the model.

### Preprocessing:
The evaluation specifically tests the impact of anti-aliasing during resizing. Images are resized with and without anti-aliasing to demonstrate the difference in model output. The provided code snippet shows the recommended preprocessing steps including resizing without anti-aliasing and normalization.

---

## Training Data
This section provides details about the datasets used to train the model.

### Datasets:
The model was trained on LAION improved aesthetics 6plus. The base Stable Diffusion model was trained on subsets of [LAION-2B(en)](https://laion.ai/blog/laion-5b/) and [LAION-5B](https://laion.ai/blog/laion-5b/).

### Motivation:
LAION datasets were chosen for training Stable Diffusion models due to their large scale and diversity, enabling the model to learn general image generation capabilities. LAION improved aesthetics 6plus was used for fine-tuning this specific image variation model.

### Preprocessing:
Images were encoded through a ViT-L/14 image-encoder including the final projection layer to the CLIP shared embedding space before being used for training.

---

## Quantitative Analyses
This section presents disaggregated evaluation results.

### Unitary results:
Visual comparisons of image outputs for V1 vs V2 and with/without anti-aliasing can be considered as unitary results, demonstrating the impact of model version and preprocessing choices on image generation quality.

### Intersectional results:
Not available.

---

## Memory or Hardware Requirements
This section outlines the memory or hardware requirements for loading, deploying, and training the model.

### Loading Requirements:
Loading and running the model requires a GPU. Example code specifies "device = "cuda:0"".

### Deploying Requirements:
Deploying the model requires a GPU to run efficiently. Example code specifies "device = "cuda"".

### Training or Fine-tuning Requirements:
Training the model was performed using 8 x A100-40GB GPUs.

---

## Ethical Considerations
This section discusses the ethical considerations in model development, including challenges, risks, and solutions.

The model was trained on [LAION-5B](https://laion.ai/blog/laion-5b/) which contains adult material and may exhibit biases present in the training data. There are potential risks of misuse, including generating harmful, offensive, or stereotypical content.

The model is intended to be used with the [Safety Checker](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) in Diffusers to mitigate the risk of generating NSFW content.

### Caveats and Recommendations
This section lists unresolved issues and provides guidance for users.

### Caveats:
**Limitations**

- The model does not achieve perfect photorealism
- The model cannot render legible text
- The model does not perform well on more difficult tasks which involve compositionality, such as rendering an image corresponding to “A red cube on top of a blue sphere”
- Faces and people in general may not be generated properly.
- The model was trained mainly with English captions and will not work as well in other languages.
- The autoencoding part of the model is lossy
- The model was trained on a large-scale dataset
  [LAION-5B](https://laion.ai/blog/laion-5b/) which contains adult material
  and is not fit for product use without additional safety mechanisms and
  considerations.
- No additional measures were used to deduplicate the dataset. As a result, we observe some degree of memorization for images that are duplicated in the training data.
  The training data can be searched at [https://rom1504.github.io/clip-retrieval/](https://rom1504.github.io/clip-retrieval/) to possibly assist in the detection of memorized images.

**Bias**

While the capabilities of image generation models are impressive, they can also reinforce or exacerbate social biases.
Stable Diffusion v1 was trained on subsets of [LAION-2B(en)](https://laion.ai/blog/laion-5b/),
which consists of images that are primarily limited to English descriptions.
Texts and images from communities and cultures that use other languages are likely to be insufficiently accounted for.
This affects the overall output of the model, as white and western cultures are often set as the default. Further, the
ability of the model to generate content with non-English prompts is significantly worse than with English-language prompts.

### Recommendations:
It is recommended to use this model with the [Safety Checker](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) provided in Diffusers. Users should be aware of the limitations and biases of the model and use it responsibly, avoiding the out-of-scope and misuse cases outlined in the "Intended Use" section. Pay attention to resizing images correctly without anti-aliasing for optimal results.

---

## Additional Information

📣 V2 model released, and blurriness issues fixed! 📣

🧨🎉 Image Variations is now natively supported in 🤗 Diffusers! 🎉🧨

**Old instructions**

If you are using a diffusers version <0.8.0 there is no `StableDiffusionImageVariationPipeline`,
in this case you need to use an older revision (`2ddbd90b14bc5892c19925b15185e561bc8e5d0a`) in conjunction with the lambda-diffusers repo:


First clone [Lambda Diffusers](https://github.com/LambdaLabsML/lambda-diffusers) and install any requirements (in a virtual environment in the example below):

```bash
git clone https://github.com/LambdaLabsML/lambda-diffusers.git
cd lambda-diffusers
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run the following python code:

```python
from pathlib import Path
from lambda_diffusers import StableDiffusionImageEmbedPipeline
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionImageEmbedPipeline.from_pretrained(
"lambdalabs/sd-image-variations-diffusers",
revision="2ddbd90b14bc5892c19925b15185e561bc8e5d0a",
)
pipe = pipe.to(device)

im = Image.open("your/input/image/here.jpg")
num_samples = 4
image = pipe(num_samples*[im], guidance_scale=3.0)
image = image["sample"]

base_path = Path("outputs/im2im")
base_path.mkdir(exist_ok=True, parents=True)
for idx, im in enumerate(image):
    im.save(base_path/f"{idx:06}.jpg")
```


*This model card was written by: Justin Pinkney and is based on the [Stable Diffusion model card](https://huggingface.co/CompVis/stable-diffusion-v1-4).*