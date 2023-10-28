import warnings
warnings.filterwarnings("ignore")
import os,json 

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/google")
os.environ["HF_HOME"] = relative_path

##--------------------
## carga el modelo
from diffusers import DDPMPipeline

prompt = "cat running"
ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True, from_tf=True,
                                     prompt=prompt)
image = ddpm(num_inference_steps=1000).images[0]


##--------------------
from PIL import Image
image.save("imagen.png")
