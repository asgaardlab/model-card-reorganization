Extract and list information from each sentence of a model card given in markdown format.

List extracted pieces of information, breaking down complex sentences into simpler, distinct pieces of information wherever possible. Treat special markdown elements like tables, code snippets, images, and citations as single list items.

- Break down individual sentences into simpler assertions.
- Include every detail from the model card, preserving format and content.
- Tables, code snippets, images, and citations should be treated as one information entry in their entirety.

# Steps

1. Parse each sentence in the model card, breaking down complex sentences into simpler individual assertions.
2. Extract special elements:
   - **Tables:** Consider each table as a single piece of information and keep it intact.
   - **Code Snippets:** Consider each code snippet as a single piece of information without modification.
   - **Images:** Consider each image as a single piece of information and keep it intact.
   - **Citations:** Consider each citation as a single piece of information and keep it intact.
   - **Hyperlinks:** Keep every hyperlink with associated text. Don't remove the hyperlinks even if they are repeatetive.
   - **Headings:** Include every headings in the list as a heading of a list of items of that section.
3. List all the extracted information without losing any details.

# Output Format

- Provide each piece of information as a separate bullet point.
- Maintain sequential order, matching the model card’s arrangement.
  
# Examples

**Input Example:**
## Model
![mmdit](mmdit.png)

Stable Diffusion is a latent text-to-image diffusion model capable of generating photo-realistic images given any text input. For more information about how Stable Diffusion functions, please have a look at [🤗's Stable Diffusion with 🧨Diffusers blog](https://huggingface.co/blog/stable_diffusion).

## Training
Carbon footprint
||Time (GPU hours)|Power Consumption (W)|Carbon Emitted(tCO<sub>2</sub>eq)|
|---|---|---|---|
|Llama 2 7B|184320|400|31.22|
|Llama 2 13B|368640|400|62.44|
|Llama 2 70B|1720320|400|291.42|
|Total|3311616||539.00|
**CO<sub>2</sub> emissions during pretraining.**

## How to Run
To use the model with Diffusers, you can run:

```
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
```

**Cite as**:
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

**Output Example:**

## Model
- ![mmdit](mmdit.png)
- Stable Diffusion is a latent text-to-image diffusion model.
- Stable Diffusion is capable of generating photo-realistic images.
- Stable Diffusion works with any text input.
- For more information about how Stable Diffusion functions, please have a look at [🤗's Stable Diffusion with 🧨Diffusers blog](https://huggingface.co/blog/stable_diffusion).

## Training
- Table titled: Carbon Emissions During Pretraining
   ||Time (GPU hours)|Power Consumption (W)|Carbon Emitted(tCO<sub>2</sub>eq)|
   |---|---|---|---|
   |Llama 2 7B|184320|400|31.22|
   |Llama 2 13B|368640|400|62.44|
   |Llama 2 70B|1720320|400|291.42|
   |Total|3311616||539.00|

## How to Run
- Code snippet to use the model with Diffusers:
   ```
   import torch
   from diffusers import StableDiffusion3Pipeline
   
   pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
   pipe = pipe.to("cuda")
   ```
- **Cite as**:
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

# Notes

- For complex or compounded sentences, break them down into simpler assertions to have a more granular representation. Don't leave anything out.
- Maintain the content’s integrity — do not summarize or add additional inferences that are not present in the original text.
