import sounddevice as sd
import wave
import numpy as np

duracion = 10  # segundos
sampling_rate = 16000  # Frecuencia de muestreo (ejemplo: 16000 Hz)

print("Iniciando grabación...")
audio = sd.rec(int(duracion * sampling_rate), samplerate=sampling_rate,
                        channels=1, dtype="int16",
                        blocking=True)
#sd.wait()
print("Grabación finalizada")

audio16=audio;
float64_array = audio.astype(np.float64)/1024.0/32

tipo_de_dato = type(audio)
print(tipo_de_dato)

#audio_float64 = audio.astype(np.float64)

print(F"{audio.shape}-{audio.dtype}")

filename = "salida.wav"  # Nombre del archivo de audio de salida
print("Audio grabado. Guardando en", filename)
with wave.open(filename, 'wb') as wf:
     wf.setnchannels(1)
     wf.setsampwidth(2) # la cantidad de byte de 
     wf.setframerate(sampling_rate)
     wf.writeframes(audio.tobytes())

print("Iniciando  lectura fichero de audio ...")
import soundfile as sf
audio_path = "salida.wav"
audio, sampling_rate = sf.read(audio_path)
print("Lectura del archivo finalizada")

print(F"{audio.shape}-{audio.dtype}")
 
tipo_de_dato = type(audio)
print(tipo_de_dato)