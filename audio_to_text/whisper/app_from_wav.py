import warnings
warnings.filterwarnings("ignore")
import os,json 

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/huggingface")
os.environ["HF_HOME"] = relative_path

##------------------
# 

print("Iniciando  lectura fichero de audio ...")
import soundfile as sf
audio_path = "salida1.wav"
audio, sampling_rate = sf.read(audio_path, dtype='int16')
print("Lectura del archivo finalizada")

print(F"{audio.shape}-{audio.dtype}-{sampling_rate }")

##-----------------

# carga el modelo

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.config.forced_decoder_ids = None

##-----------------

# Procesando el audio

input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
predicted_ids = model.generate(input_features)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print(transcription)