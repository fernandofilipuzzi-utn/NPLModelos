import warnings
warnings.filterwarnings("ignore")
import os,json 

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/huggingface")
os.environ["HF_HOME"] = relative_path

##-----------------

import moviepy.editor as mp
import wave
import numpy as np

sampling_rate = 16000

# Nombre del archivo MKV de entrada

input_file = "entrada.mkv"
output_file = "salida.wav"

print("Iniciando  lectura y extracción del audio del fichero de video ...")
clip = mp.VideoFileClip(input_file)
audio = clip.audio
audio.write_audiofile(output_file, codec='pcm_s16le', ffmpeg_params=["-ac", "1", "-ar", "16000"])
print("Lectura del archivo finalizada")


##------------------

# carga del fichero de audio

print("Iniciando  lectura fichero de audio ...")
import soundfile as sf
audio_path = "salida.wav"
audio, sampling_rate = sf.read(audio_path, dtype='int16')
print("Lectura del archivo finalizada")

print(F"{audio.shape}-{audio.dtype}")

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

##-----------------

print(transcription)
output_file = "salida.txt"

transcription_text = '\n'.join(transcription)

with open(output_file, 'w', encoding='utf-8') as file:
    file.write(transcription_text)