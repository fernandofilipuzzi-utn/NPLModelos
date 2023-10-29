import warnings
warnings.filterwarnings("ignore")
import os,json 

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache")
os.environ["HF_HOME"] = relative_path

##-----------------------

from diffusers import StableDiffusionPipeline
import torch

model_id = "CompVis/stable-diffusion-v1-4"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
#pipe = pipe.to("cuda")

# prompt = "retro serie of different cars with different colors and shapes, mdjrny-v4 style"
# prompt = "duckling on the water, comic style"
# prompt = "light watercolor, interior of a cozy cafe, bright, white background, few details, dreamy, Studio Ghibli --seed 3817455947 --no people person"
# prompt = "train station, watercolor painting --no people, clocks, flowers"

prompt = "a close up of a person with pink hair, inspired by Lois van Baarle, trending on Artstation, digital art, girl with short white hair, in style of cyril rolando, night color, rob rey and kentarõ miura style, concept art, mdjrny-v4 style"
image = pipe(prompt).images[0]

##---------------------------

from PIL import Image
image.save("imagen.png")
