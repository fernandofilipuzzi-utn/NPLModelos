from moviepy.editor import VideoFileClip
import soundfile as sf

input_file = "entrada.mkv"
output_file = "salida.wav"

video_clip = VideoFileClip(input_file)
audio = video_clip.audio
target_sample_rate = 16000
audio.write_audiofile(output_file,  fps=target_sample_rate,codec=None)

print("Audio extraction and conversion completed.")

from pydub import AudioSegment

audio = AudioSegment.from_wav(output_file)
audio = audio.set_channels(1)
audio = audio.set_frame_rate(16000)
audio.export(output_file, format="wav")