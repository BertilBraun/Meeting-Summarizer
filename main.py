import os
from moviepy.editor import VideoFileClip
import whisper


def transcribe_video(video_path: str) -> str:
    # Convert video to audio using moviepy
    audio_path = 'audiofile.wav'

    print(f'Converting {video_path} to {audio_path}')
    video = VideoFileClip(video_path)
    print(f'Loaded video {video_path}')
    if video.audio is None:
        raise ValueError('Video has no audio')

    video.audio.write_audiofile(audio_path, codec='pcm_s16le', fps=44100, nbytes=2)
    print(f'Converted video to audio')

    # Load the Whisper model (automatically downloads the model if it's the first time)
    print(f'Loading model')
    model = whisper.load_model('medium')
    print(f'Loaded model')

    # Transcribe the audio to German
    print(f'Transcribing audio')
    result = model.transcribe(audio_path, language='de', verbose=True)
    print(f'Transcribed audio')

    os.remove(audio_path)

    return result['text']


if __name__ == '__main__':
    video_path = R'C:\Path\To\Video.mp4'
    transcription = transcribe_video(video_path)
    print(transcription)
    with open('transcription.txt', 'w', encoding='utf-8') as file:
        file.write(transcription)
