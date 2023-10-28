import warnings
warnings.filterwarnings("ignore")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/huggingface")
os.environ["HF_HOME"] = relative_path

##------

# captura de audio desde el microfono

import sounddevice as sd
import numpy as np

duracion = 10 # en segundos
sampling_rate=16000

print("Iniciando la grabación...")
audio = sd.rec(int(duracion * sampling_rate), samplerate=sampling_rate, channels=1, dtype="int16")
sd.wait()
print("Grabación finalizada")

audio = audio.astype(np.float64)/1024.0/32.0

##-----------------

# carga el modelo

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.config.forced_decoder_ids = None

##-----------------

# procesando el audio

input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
predicted_ids = model.generate(input_features, temperature=0.7)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

##-----------------

print(transcription)
