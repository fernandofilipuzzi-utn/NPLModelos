import warnings
warnings.filterwarnings("ignore")
import os,json 

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/huggingface")
os.environ["HF_HOME"] = relative_path

##-----------------

sampling_rate=16000

# import moviepy.editor as mp

# input_file = "entrada.mkv"
# output_file = "salida.wav"

# clip = mp.VideoFileClip(input_file)
# audio = clip.audio
# audio.write_audiofile(output_file, codec='pcm_s16le', verbose=False)

# print("Extracción del audio finalizada")

##-----------------

# from pydub import AudioSegment

# input_file = "salida1.wav"
# output_file = "salida.wav"

# audio = AudioSegment.from_wav(input_file)
# audio = audio.set_channels(1)
# audio = audio.set_frame_rate(sampling_rate)
# audio.export(output_file, format="wav")

# print("Conversión completada")

##------------------

print("Iniciando  lectura fichero de audio ...")

import soundfile as sf
output_file = "salida.wav"
audio, sampling_rate = sf.read(output_file, dtype='int16')

print("Lectura del archivo finalizada")

print(F"{audio.shape}-{audio.dtype}-{sampling_rate }")

##-----------------

# carga el modelo

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

model_name='openai/whisper-large'
#model_name='openai/whisper-huge'

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
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