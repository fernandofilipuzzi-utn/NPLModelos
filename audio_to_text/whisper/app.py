from moviepy.editor import VideoFileClip
import soundfile as sf

# Input MKV video file and output WAV audio file
input_file = "entrada.mkv"
output_file = "salida.wav"

# Load the video clip
video_clip = VideoFileClip(input_file)

# Extract the audio from the video
audio = video_clip.audio

# Set the desired sample rate (16000 Hz)
target_sample_rate = 16000

# Export the audio as a WAV file with the desired sample rate
#audio.write_audiofile(output_file, codec="pcm_s16le", ffmpeg_params=["-ac", "1", "-ar", str(target_sample_rate)])
audio.write_audiofile(output_file, codec=None)

print("Audio extraction and conversion completed.")

from pydub import AudioSegment

audio = AudioSegment.from_wav(output_file)
audio = audio.set_channels(1)
audio = audio.set_frame_rate(16000)
audio.export(output_file, format="wav")