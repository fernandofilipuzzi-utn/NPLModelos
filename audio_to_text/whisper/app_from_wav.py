import warnings
warnings.filterwarnings("ignore")
import os,json 

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path = os.path.join(script_dir, ".cache/huggingface")
os.environ["HF_HOME"] = relative_path

##------------------

# Extrae el audio

from moviepy.editor import VideoFileClip
import soundfile as sf

input_file = "entrada.mkv"
output_file = "salida.wav"

video_clip = VideoFileClip(input_file)
audio = video_clip.audio
target_sample_rate = 16000
audio.write_audiofile(output_file,  fps=target_sample_rate,codec=None)

print("Audio extraction and conversion completed.")

##-------------------

# Convierte el fichero de audio a un fichero wav pcm monocanal

from pydub import AudioSegment

audio = AudioSegment.from_wav(output_file)
audio = audio.set_channels(1)
audio = audio.set_frame_rate(target_sample_rate)
audio.export(output_file, format="wav")

##-------------------

print("Iniciando  lectura fichero de audio ...")
import soundfile as sf
audio_path = "salida.wav"
audio, sampling_rate = sf.read(audio_path)
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

##-----------------

print(transcription)

transcription_text = '\n'.join(transcription)

with open(output_file, 'w', encoding='utf-8') as file:
    file.write(transcription_text)