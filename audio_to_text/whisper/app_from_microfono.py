import warnings
warnings.filterwarnings("ignore")
import os,json 

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/huggingface")
os.environ["HF_HOME"] = relative_path

##-----------------

# captura del microfono 
import sounddevice as sd
samplerate = 16000
tiempo=10 #segundos
audio = sd.rec(int(tiempo * samplerate), samplerate=samplerate, channels=1, dtype="int16")
sd.wait()

##-----------------
# carga el modelo

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.config.forced_decoder_ids = None

##-----------------
# procesando el audio
input_values = processor(audio, return_tensors="pt").input_values
predicted_ids = model.generate(input_values)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

##-----------------
print(transcription)
