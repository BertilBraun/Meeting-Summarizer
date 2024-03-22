import os
from moviepy.editor import VideoFileClip
from pyAudioAnalysis import audioSegmentation as aS
import whisper


def transcribe_video(video_path: str) -> str:
    # Convert video to audio using moviepy
    audio_path = 'audiofile.wav'

    print(f'Converting {video_path} to {audio_path}')
    video = VideoFileClip(video_path)
    print(f'Loaded video {video_path}')
    if video.audio is None:
        raise ValueError('Video has no audio')

    # trim audio to 1 minute
    # TODO remove this
    # video = video.subclip(0, 60)

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

    for segment in result['segments']:
        del segment['id']
        del segment['tokens']
        del segment['avg_logprob']
        del segment['compression_ratio']
        del segment['temperature']
        del segment['no_speech_prob']
        del segment['seek']

    print(result)

    audio = 0  # TODO parse wav file into numpy array

    starts, ends, speakers = diarize_audio(audio_path)

    for start, end, speaker in zip(starts, ends, speakers):
        segment = audio[start:end]
        transcription = model.transcribe(segment, language='de', verbose=True)
        print(f'start={start:.1f}s stop={end:.1f}s speaker={speaker}')
        print(transcription)

    print(flags)

    flags = [int(flag) for flag in flags]

    # Map diarization speaker flags to Whisper transcription segments
    transcription_segments = map_sentences_to_speakers(result['segments'], flags)

    print(transcription_segments)

    # Organize the transcription by speaker
    organized_segments = organize_by_speaker(transcription_segments)
    print(organized_segments)

    # os.remove(audio_path)

    return result['text']


def diarize_audio(audio_path):
    # Perform speaker diarization
    # 'smoothing' and 'weight' are parameters that might need tuning based on your audio
    # The number of speakers is set to 0, which means the algorithm tries to estimate it
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        'pyannote/speaker-diarization-3.1',
        use_auth_token='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',  # TODO hide token
    )

    # apply the pipeline to an audio file
    diarization = pipeline(audio_path)  # , min_speakers=1, max_speakers=3)  # TODO add parameter for number of speakers

    tracks = list(diarization.itertracks(yield_label=True))
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f'start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}')

    print(diarization)
    print(type(diarization))
    print(tracks)
    print(type(tracks[0]))
    print(tracks[0][0].start)
    print(tracks[0][0].end)
    print(tracks[-1][0].start)
    print(tracks[-1][0].end)
    print(type(tracks[-1][0].end))
    print(int(tracks[-1][0].end) * 10)

    starts, ends, speakers = [], [], []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speakers and speakers[-1] == speaker:
            ends[-1] = turn.end
        else:
            starts.append(turn.start)
            ends.append(turn.end)
            speakers.append(speaker)

    print(starts)
    print(ends)
    print(speakers)

    return starts, ends, speakers

    end = tracks[-1][0].end
    flags = [-1 for _ in range(int(end) * 10)]
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        for i in range(int(turn.start) * 10, int(turn.end) * 10):
            flags[i] = speaker
    return diarization

    [flags, classes, accuracy] = aS.speaker_diarization(audio_path, 0, plot_res=False)
    return flags, classes


def split_text_into_sentences(text):
    """
    Simple function to split text into sentences based on punctuation.
    This is a naive implementation and can be replaced with more sophisticated NLP tools.
    """
    import re

    sentences = re.split(r'[.!?]\s*', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences


def map_sentences_to_speakers(transcription_segments, flags, segment_duration=0.1):
    """
    Enhance mapping by splitting text into sentences for segments with mixed speakers.
    :param transcription_segments: List of Whisper segments with 'start', 'end', 'text'
    :param flags: List of speaker IDs from diarization for fixed-duration segments
    :param segment_duration: Duration of each segment in the diarization flags, in seconds
    :return: A list of dicts with 'start', 'end', 'text', 'speaker_id' for each sentence
    """
    result_segments = []
    for segment in transcription_segments:
        start_index = int(segment['start'] / segment_duration)
        end_index = int(segment['end'] / segment_duration) + 1
        segment_flags = flags[start_index:end_index]

        # Filter out pause/silence speaker IDs if defined (e.g., ID 0)
        # actual_speaker_flags = [flag for flag in segment_flags if flag != 0]

        if not segment_flags or len(set(segment_flags)) == 1:
            # Segment has a unanimous speaker (or silence), keep it as is
            speaker_id = segment_flags[0] if segment_flags else None
            segment['speaker_id'] = speaker_id
            segment['flags'] = segment_flags
            result_segments.append(segment)
        else:
            # Segment has mixed speakers, split into sentences
            sentences = split_text_into_sentences(segment['text'])
            sentence_start_time = segment['start']
            for sentence in sentences:
                # Estimate sentence duration and speaker by splitting segment evenly among sentences
                sentence_duration = (segment['end'] - segment['start']) / len(sentences)
                sentence_end_time = sentence_start_time + sentence_duration
                sentence_start_index = int(sentence_start_time / segment_duration)
                sentence_end_index = int(sentence_end_time / segment_duration) + 1

                sentence_flags = flags[sentence_start_index:sentence_end_index]
                most_common_speaker = max(set(sentence_flags), key=sentence_flags.count) if sentence_flags else None

                result_segments.append(
                    {
                        'start': sentence_start_time,
                        'end': sentence_end_time,
                        'text': sentence,
                        'flags': flags[start_index:end_index],
                        'assigned_flags': sentence_flags,
                        'speaker_id': most_common_speaker,
                        'mixed_speakers': True,
                    }
                )

                sentence_start_time = sentence_end_time  # Update start time for the next sentence

    end_index = int(segment['end'] / segment_duration) + 1
    print('We are at the end of the function')
    print(end_index, len(flags), 'diff', len(flags) - end_index)

    return result_segments


def organize_by_speaker(transcription_segments):
    organized_segments = []
    current_speaker = None
    current_segment = None

    for segment in transcription_segments:
        # Check if the current segment is continuing
        if segment['speaker_id'] == current_speaker:
            # Append text to the current segment
            current_segment['text'] += ' ' + segment['text']
            current_segment['end'] = segment['end']
        else:
            # Finish the current segment and start a new one
            if current_segment:
                organized_segments.append(current_segment)

            current_speaker = segment['speaker_id']
            current_segment = {
                'speaker_id': current_speaker,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
            }

    # Don't forget to add the last segment
    if current_segment:
        organized_segments.append(current_segment)

    return organized_segments


if __name__ == '__main__':
    segments = [
        {
            'start': 0.0,
            'end': 4.36,
            'text': ' Weg, also ich fahre jetzt nicht, aber ich stehe hier im Auto quasi zur Besprechung,',
        },
        {
            'start': 5.48,
            'end': 10.040000000000001,
            'text': ' weil ich gerade aus einem anderen Termin raus bin und es nicht rechtzeitig geschafft habe.',
        },
        {'start': 10.64, 'end': 13.68, 'text': ' Aber das macht nichts, ich habe meinen Laptop hier bei mir.'},
        {'start': 13.68, 'end': 16.2, 'text': ' Wir können gerne durchstarten.'},
        {'start': 16.2, 'end': 18.12, 'text': ' Kein Problem. Super, dass es geklappt hat.'},
        {'start': 18.12, 'end': 21.48, 'text': ' Und danke, dass ihr heute auch die Zeit habt.'},
        {'start': 21.48, 'end': 34.36, 'text': ' Also mit Bertil hatte ich ja schon getagt.'},
        {'start': 34.36, 'end': 37.64, 'text': ' Arthur, ist es okay, wenn wir uns gut sind?'},
        {'start': 37.64, 'end': 39.120000000000005, 'text': ' Ja, sehr gerne sogar.'},
        {'start': 39.120000000000005, 'end': 43.04, 'text': ' Okay, nur ab und zu muss man fragen.'},
        {'start': 43.04, 'end': 45.88, 'text': ' Ich würde vielleicht mal kurz noch was zu mir sagen.'},
        {'start': 45.88, 'end': 48.6, 'text': ' Dann vielleicht noch, dass du mir mal ganz kurz was zu dir sagst,'},
        {'start': 48.6, 'end': 50.440000000000005, 'text': ' einfach ein bisschen kennenlernen.'},
        {
            'start': 50.440000000000005,
            'end': 56.96,
            'text': ' Genau, ich bin akademischer Mitarbeiter, Doktorand bei Herrn Oberweiß',
        },
        {'start': 56.96, 'end': 59.88, 'text': ' in der Forschungsgruppe Betriebliche Informationssysteme.'},
    ]

    flags, classes = diarize_audio('audiofile.wav')
    print(flags)

    flags = [int(flag) for flag in flags]

    # Map diarization speaker flags to Whisper transcription segments
    transcription_segments = map_sentences_to_speakers(segments, flags)

    print(transcription_segments)

    # Organize the transcription by speaker
    organized_segments = organize_by_speaker(transcription_segments)
    print(organized_segments)

else:
    video_path = R'C:\Users\berti\OneDrive\Documents\Studium\Semester 8\Masterarbeit\Auftakt.wmv'
    transcription = transcribe_video(video_path)
    print(transcription)
    with open('transcription.txt', 'w', encoding='utf-8') as file:
        file.write(transcription)
