# Meeting Summarizer

The Meeting Summarizer project leverages the power of OpenAI's Whisper model for transcription and `pyAudioAnalysis` for speaker diarization to analyze video recordings, particularly useful in meeting settings. It transcribes spoken content into text and identifies distinct speakers, organizing the transcript accordingly for easier review and analysis.

## Features

- **Transcription**: Converts spoken language in videos into written text, supporting a wide range of languages as offered by the Whisper model.
- **Speaker Diarization**: Differentiates between speakers in the recording, assigning unique IDs to segments of speech by different individuals.
- **Segment Organization**: Compiles the transcription into a structured format by grouping consecutive speech segments by the same speaker. This organization allows for a clear, readable transcript that indicates not only what was said but also by whom, without interruptions for brief pauses or silence.

## Prerequisites

- **Python 3.8+**: Ensure Python is installed on your system. You can verify this by running `python --version` in your terminal.
- **FFmpeg**: This project requires FFmpeg for processing audio from video files. Install FFmpeg using `sudo apt-get install ffmpeg` on Ubuntu or `choco install ffmpeg` on Windows.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/BertilBraun/Meeting-Summarizer
    ```

2. **Navigate to the project directory**:

    ```bash
    cd Meeting-Summarizer
    ```

3. **Install required Python packages**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare your video file**: Have the video you wish to analyze ready. Ensure clear audio for best results.

2. **Run the script**: Execute the main script via the terminal or command prompt:

    ```bash
    python main.py
    ```

3. **Follow prompts**: Input the path to your video file, the number of speakers present, and the language code (e.g., "en" for English) when prompted by the script.

The script processes the video to generate a transcript that organizes speech segments by speaker, facilitating a coherent read-through. The outcome is displayed in the terminal and also saved to `transcript.txt` within the project directory.

## Customizing Speaker Names

After processing, you'll have the opportunity to assign names to the identified speakers, replacing generic IDs with meaningful labels in the final transcript for enhanced clarity.

## Contributing

Contributions are welcome! If you have ideas for improvements or encounter any issues, please feel free to create an issue or submit a pull request on GitHub.
