import warnings
warnings.filterwarnings("ignore")
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/huggingface")
os.environ["HF_HOME"] = relative_path

##-----------------
# captura de audio desde el microfono

import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.config.forced_decoder_ids = None

##------
import sounddevice as sd


print("Iniciando la grabación...")
duracion = 5  # en segundos
sampling_rate=16000
audio = sd.rec(int(duracion * sampling_rate), channels=1, dtype="int16")
sd.wait()
print("Grabación finalizada")

##----

input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

##-----------------

print(transcription)
