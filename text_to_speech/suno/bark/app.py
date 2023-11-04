#app_sentimental.py

import warnings
warnings.filterwarnings("ignore")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/suno")
os.environ["HF_HOME"] = relative_path

##----------

from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("suno/bark")
model = AutoModel.from_pretrained("suno/bark")

inputs = processor(
    text=["hola, ¡soy la bomba! ... ¡tucumana!, ja ja ja!"],
    return_tensors="pt",
)

speech_values = model.generate(**inputs, do_sample=True)


import soundfile as sf
sf.write("salida.wav", speech_values[0].numpy(), 22050)