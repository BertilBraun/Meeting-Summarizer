# Taken from: https://huggingface.co/spaces/therealcyberlord/whisper-diarization/blob/main/app.py

import numpy as np
from torch import tensor
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline


# Model and pipeline setup
model_id = 'openai/whisper-medium'
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, low_cpu_mem_usage=True, use_safetensors=True)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    'automatic-speech-recognition',
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
)

diarization_pipeline = Pipeline.from_pretrained(
    'pyannote/speaker-diarization-3.1',
    use_auth_token='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',  # TODO hide token
)


# returns diarization info such as segment start and end times, and speaker id
def diarization_info(res):
    starts = []
    ends = []
    speakers = []

    for segment, _, speaker in res.itertracks(yield_label=True):
        starts.append(segment.start)
        ends.append(segment.end)
        speakers.append(speaker)

    return starts, ends, speakers


def transcribe(sr, data):
    processed_data = np.array(data).astype(np.float32) / 32767.0

    # results from the pipeline
    transcription_res = pipe({'sampling_rate': sr, 'raw': processed_data}, generate_kwargs={'language': 'de'})

    transcription_text: str = transcription_res['text']  # type: ignore

    return transcription_text.strip()


def transcribe_diarize(sr, audio_data):
    # Check if the audio is multi-channel and convert it to mono by averaging the channels
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        # Averaging the channels if the audio is stereo or has more channels
        audio_data = np.mean(audio_data, axis=1)

    processed_data = np.array(audio_data).astype(np.float32) / 32767.0
    waveform_tensor = tensor(processed_data[np.newaxis, :])

    print('Diarizing...')
    # results from the diarization pipeline
    diarization_res = diarization_pipeline({'waveform': waveform_tensor, 'sample_rate': sr})
    print('Diarization:')
    print(diarization_res)

    # Get diarization information
    starts, ends, speakers = diarization_info(diarization_res)

    # results from the transcription pipeline
    diarized_transcription = ''
    transcriptions = []

    # Get transcription results for each speaker segment
    for start_time, end_time, speaker_id in tqdm(zip(starts, ends, speakers), total=len(starts), desc='Transcribing'):
        segment = audio_data[int(start_time * sr) : int(end_time * sr)]
        transcription = transcribe(sr, segment)
        diarized_transcription += f'{speaker_id} {round(start_time, 2)}:{round(end_time, 2)} \t {transcription}\n'

        if transcriptions and transcriptions[-1][0] == speaker_id:
            transcriptions[-1] = (speaker_id, transcriptions[-1][1] + '\n' + transcription)
        else:
            transcriptions.append((speaker_id, transcription))

    return transcriptions


if __name__ == '__main__':
    audio_file = 'audiofile.wav'
    from scipy.io import wavfile

    samplerate, audio_data = wavfile.read(audio_file)

    diarized_transcription = transcribe_diarize(samplerate, audio_data)
    print(diarized_transcription)
