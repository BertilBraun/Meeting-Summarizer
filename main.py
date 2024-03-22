import os
import whisper
import numpy as np
from dataclasses import dataclass
from moviepy.editor import VideoFileClip
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import ShortTermFeatures as sF


AS_SEGMENT_DURATION = 0.1  # Duration of each segment in the speaker diarization, in seconds
TRANSCRIPT_PATH = 'transcript.txt'  # Path to the transcript file


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker_id: int | None  # -1 for silence


def transcribe_video(video_path: str, num_speakers: int, language: str) -> list[Segment]:
    """
    Transcribe the audio of a video file to text. The function performs speaker diarization on the audio file and transcribes the audio to text using the Whisper model. The function returns a list of Segments with 'start', 'end', 'text', 'speaker_id'. The 'speaker_id' is an integer representing the speaker ID for the segment. These segments should represent the dialogue as accurately as possible, with each segment containing the text spoken by a single speaker and the next segment containing the text spoken by another speaker. The speaker ID should be unique for each speaker and should be consistent throughout the transcription.
    :param video_path: Path to the video file
    :param num_speakers: Number of speakers in the audio file
    :param language: Language of the audio file
    :return: List of Segments with 'start', 'end', 'text', 'speaker_id'
    """
    tmp_audio_path = 'audiofile.wav'  # Path to the temporary audio file

    # Convert video to audio using moviepy
    print(f'Converting {video_path} to {tmp_audio_path}')
    video = VideoFileClip(video_path)
    print(f'Loaded video {video_path}')
    if video.audio is None:
        raise ValueError('Video has no audio')

    video.audio.write_audiofile(tmp_audio_path, codec='pcm_s16le', fps=44100, nbytes=2)
    print('Converted video to audio')

    segments = transcribe_audio(tmp_audio_path, num_speakers, language)

    os.remove(tmp_audio_path)

    return segments


def transcribe_audio(audio_path: str, num_speakers: int, language: str) -> list[Segment]:
    """
    Transcribe the audio file to text. The function performs speaker diarization on the audio file and transcribes the audio to text using the Whisper model. The function returns a list of Segments with 'start', 'end', 'text', 'speaker_id'. The 'speaker_id' is an integer representing the speaker ID for the segment. These segments should represent the dialogue as accurately as possible, with each segment containing the text spoken by a single speaker and the next segment containing the text spoken by another speaker. The speaker ID should be unique for each speaker and should be consistent throughout the transcription.
    :param audio_path: Path to the audio file
    :param num_speakers: Number of speakers in the audio file
    :param language: Language of the audio file
    :return: List of Segments with 'start', 'end', 'text', 'speaker_id'
    """
    # Load the Whisper model (automatically downloads the model if it's the first time)
    print('Loading model')
    model = whisper.load_model('medium')
    print('Loaded model')

    # Transcribe the audio to German
    print('Transcribing audio')
    result = model.transcribe(audio_path, language=language, verbose=True)
    print('Transcribed audio')

    segments = []
    for segment in result['segments']:
        segments.append(Segment(segment['start'], segment['end'], segment['text'].strip(), None))  # type: ignore

    print(segments)

    flags = diarize_audio(audio_path, num_speakers)

    print('Flags')
    print(flags)

    # Map diarization speaker flags to Whisper transcription segments
    transcription_segments = map_sentences_to_speakers(segments, flags)

    print(transcription_segments)

    # Organize the transcription by speaker
    organized_segments = organize_by_speaker(transcription_segments)
    print(organized_segments)

    return organized_segments


def diarize_audio(audio_path: str, num_speakers: int) -> list[int]:
    """
    Perform speaker diarization on the audio file. The function returns a list of speaker IDs for fixed-duration segments. The speaker IDs are integers starting from 0. The function also detects silence periods in the audio file which are represented by a speaker ID of -1.
    """

    # Detect silence periods
    silence_periods = detect_silence(audio_path)

    [flags, classes, accuracy] = aS.speaker_diarization(audio_path, num_speakers, plot_res=False)
    flags = [int(flag) for flag in flags]

    # Adjust flags based on silence
    adjusted_flags = adjust_flags_for_silence(flags, silence_periods)

    return adjusted_flags


def detect_silence(audio_path: str, smoothing_filter_size: int = 100) -> list[tuple[float, float]]:
    """
    Detects silence periods in an audio file.

    :param audio_path: Path to the audio file.
    :param smoothing_filter_size: Size of the smoothing filter applied to energy signal.
    :return: A list of tuples representing silent periods (start_time, end_time).
    """
    # Extract short-term features
    [fs, x] = audioBasicIO.read_audio_file(audio_path)
    x = audioBasicIO.stereo_to_mono(x)  # Convert to mono if stereo

    # Calculate frame length and step size in samples
    frame_length_samples = int(0.050 * fs)  # 50 ms frame
    frame_step_samples = int(0.025 * fs)  # 25 ms step

    features, f_names = sF.feature_extraction(x, fs, frame_length_samples, frame_step_samples)

    # Find the index of the energy feature
    energy_index = f_names.index('energy')
    energy = features[energy_index, :]

    # Smooth the energy signal
    if smoothing_filter_size > 1:
        energy = np.convolve(energy, np.ones(smoothing_filter_size) / smoothing_filter_size, mode='same')

    # Using the energy signal calculated above
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)

    # Example heuristic: set threshold as mean minus half standard deviation
    energy_threshold = max(mean_energy - 1.2 * std_energy, 0.001)  # Ensure it doesn't go negative

    # Identify frames below the energy threshold
    silent_frames = energy < energy_threshold

    # Group silent frames into continuous silent periods
    silent_periods = []
    start_time = None
    for i, is_silent in enumerate(silent_frames):
        if is_silent and start_time is None:
            start_time = i * 0.025  # Start time of the silent period
        elif not is_silent and start_time is not None:
            end_time = i * 0.025  # End time of the silent period
            silent_periods.append((start_time, end_time))
            start_time = None
    # Handle case where the last frame is silent
    if start_time is not None:
        silent_periods.append((start_time, len(silent_frames) * 0.025))

    return silent_periods


def adjust_flags_for_silence(flags: list[int], silence_periods: list[tuple[float, float]]) -> list[int]:
    """
    Insert silence periods into the speaker flags array. This function adjusts the speaker flags array based on the identified silence periods. The speaker flags array is a list of speaker IDs for fixed-duration segments. The silence periods are represented by a speaker ID of -1.
    :param flags: List of speaker IDs from diarization for fixed-duration segments
    :param silence_periods: List of tuples representing silent periods (start_time, end_time)
    :return: Adjusted list of speaker IDs with silence periods inserted
    """
    # Adjust the flags array based on identified silence periods
    adjusted_flags = flags.copy()

    for silence_start, silence_end in silence_periods:
        start_index = int(silence_start / AS_SEGMENT_DURATION)
        end_index = int(silence_end / AS_SEGMENT_DURATION) + 1
        adjusted_flags[start_index:end_index] = [-1] * (end_index - start_index)

    return adjusted_flags


def split_text_into_sentences(text: str) -> list[str]:
    """
    Simple function to split text into sentences based on punctuation.
    This is a naive implementation and can be replaced with more sophisticated NLP tools.
    """
    import re

    # TODO replace with a more sophisticated NLP tool?
    sentences = re.split(r'[.!?]\s*', text)
    sentences = [sentence.strip() for sentence in sentences if sentence]
    return sentences


def get_segment_flags(flags: list[int], start_time: float, end_time: float) -> list[int]:
    """
    Get speaker flags for a segment based on the start and end times. Silences (represented by -1) are filtered out.
    :param flags: List of speaker IDs from diarization for fixed-duration segments
    :param start_time: Start time of the segment
    :param end_time: End time of the segment
    :return: List of speaker IDs for the segment
    """
    start_index = int(start_time / AS_SEGMENT_DURATION)
    end_index = int(end_time / AS_SEGMENT_DURATION) + 1
    segment_flags = flags[start_index:end_index]

    # Filter out pause/silence speaker IDs if defined (e.g., ID -1)
    return [flag for flag in segment_flags if flag != -1]


def map_sentences_to_speakers(transcription_segments: list[Segment], flags: list[int]) -> list[Segment]:
    """
    Map sentences to speakers based on the speaker diarization. This function splits segments with mixed speakers into sentences and assigns the most common speaker ID to each sentence. The function returns a list of Segments with proper speaker IDs for each sentence.
    :param transcription_segments: List of Segments with 'start', 'end', 'text'
    :param flags: List of speaker IDs from diarization for fixed-duration segments
    :return: A list of Segments for each sentence
    """
    result_segments: list[Segment] = []
    for segment in transcription_segments:
        segment_flags = get_segment_flags(flags, segment.start, segment.end)

        if not segment_flags or len(set(segment_flags)) == 1:
            # Segment has a unanimous speaker (or silence), keep it as is
            speaker_id = segment_flags[0] if segment_flags else None
            segment.speaker_id = speaker_id
            result_segments.append(segment)
        else:
            # Segment has mixed speakers, split into sentences
            sentences = split_text_into_sentences(segment.text)

            sentence_length_prefix_sum = sum(len(sentence) for sentence in sentences)

            for i, sentence in enumerate(sentences):
                # Estimate sentence duration based on the number of characters in the sentence compared to the total with respect to the segment duration
                sentence_duration = len(sentence) / sentence_length_prefix_sum * (segment.end - segment.start)

                sentence_start_time = segment.start + i * sentence_duration
                sentence_end_time = sentence_start_time + sentence_duration

                sentence_flags = get_segment_flags(flags, sentence_start_time, sentence_end_time)
                most_common_speaker = max(set(sentence_flags), key=sentence_flags.count) if sentence_flags else None

                result_segments.append(
                    Segment(
                        start=sentence_start_time,
                        end=sentence_end_time,
                        text=sentence,
                        speaker_id=most_common_speaker,
                    )
                )

                sentence_start_time = sentence_end_time  # Update start time for the next sentence

    return result_segments


def organize_by_speaker(transcription_segments: list[Segment]) -> list[Segment]:
    """
    Organize the transcription segments by speaker. This function groups consecutive segments by the same speaker into a single segment. This means that consecutive segments by the same speaker are merged into a single segment and the text is concatenated. The output will therefore contain fewer segments than the input and never have consecutive segments by the same speaker.
    :param transcription_segments: List of Segments
    :return: List of Segments organized by speaker
    """
    organized_segments: list[Segment] = []
    current_speaker: int | None = None
    current_segment: Segment = None  # type: ignore

    for segment in transcription_segments:
        # Check if the current segment is continuing
        if segment.speaker_id == current_speaker:
            # Append text to the current segment
            current_segment.text += ' ' + segment.text
            current_segment.end = segment.end
        else:
            # Finish the current segment and start a new one
            if current_segment:
                organized_segments.append(current_segment)

            current_speaker = segment.speaker_id
            # Make a copy of the segment to avoid modifying the original segment
            current_segment = Segment(
                start=segment.start,
                end=segment.end,
                text=segment.text,
                speaker_id=segment.speaker_id,
            )

    # Don't forget to add the last segment
    if current_segment:
        organized_segments.append(current_segment)

    return organized_segments


def write_transcript(segments: list[Segment], output_path: str) -> None:
    """
    Write the transcription segments to the console and a text file. Each segment is printed with the speaker ID and text. The output file will contain the same information. The output file is opened in the default text editor after writing.
    """

    for segment in segments:
        print(f'Speaker {segment.speaker_id} : {segment.text}')

    unique_speakers = len(set([segment.speaker_id for segment in segments]))
    print(f'In total we have {len(segments)} segments and {unique_speakers} unique speakers')

    with open(output_path, 'w', encoding='utf-8') as file:
        for segment in segments:
            file.write(f'Speaker {segment.speaker_id} : {segment.text}\n\n')

    os.startfile(output_path)  # open the file in the default file editor


if __name__ == '__main__':
    video_path = input('Enter the path to the video file: ')
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'File not found: {video_path}')

    num_speakers = int(input('Enter the number of speakers: '))
    language = input('Enter the language of the audio file: ')

    transcription_segments = transcribe_video(video_path, num_speakers, language)

    write_transcript(transcription_segments, TRANSCRIPT_PATH)

    unique_speakers = list(
        sorted(set(segment.speaker_id for segment in transcription_segments if segment.speaker_id is not None))
    )
    speaker_name_mapping = {}
    for speaker in unique_speakers:
        name = input(f'Enter name for speaker {speaker}: ')
        speaker_name_mapping[speaker] = name

    for segment in transcription_segments:
        segment.speaker_id = speaker_name_mapping[segment.speaker_id]

    write_transcript(transcription_segments, TRANSCRIPT_PATH)
