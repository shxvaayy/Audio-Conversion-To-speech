# Audio & Video Analyzer with Speaker Diarization

A powerful web application that analyzes audio and video files to provide:
- **Speaker Diarization** - Identifies who spoke when
- **Speech Transcription** - Converts speech to text with speaker labels
- **Audio Analysis** - Speech rate, pitch variation, energy levels, pauses
- **Video Analysis** - Eye contact, facial expressions, body language (for video files)

## Features

### Audio Analysis
- Voice Activity Detection (VAD)
- Speech rate (words per minute)
- Pitch analysis and variation
- Energy levels and dynamic range
- Pause detection

### Speaker Diarization
- Automatic speaker detection (2-10 speakers)
- Speaker timeline visualization
- Speaking time percentages per speaker
- Powered by PyAnnote Audio

### Speech Transcription
- Accurate speech-to-text using OpenAI Whisper
- Speaker-aligned transcript
- Word-level timestamps
- Multi-language support

### Video Analysis (for video files only)
- Blink detection
- Smile detection
- Head nod/shake tracking
- Gaze stability analysis
- Forward/backward lean detection
- Overall engagement score

## Supported File Formats

### Audio Files
- MP3, WAV, M4A, OGG, FLAC, AAC

### Video Files
- MP4, MOV, AVI, WebM

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/shxvaayy/Audio-Conversion-To-speech.git
cd Audio-Conversion-To-speech
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up HuggingFace Token (IMPORTANT!)

Speaker diarization requires a HuggingFace token to download the PyAnnote models.

#### Step 1: Create HuggingFace Account
Go to https://huggingface.co/join and create a free account.

#### Step 2: Accept PyAnnote Model Terms
You MUST accept the terms for these two models (click each link, login, and click "Agree"):
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

#### Step 3: Get Your Access Token
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name (e.g., "audio-analyzer")
4. Select "Read" permission
5. Click "Generate token"
6. Copy the token (starts with `hf_...`)

#### Step 4: Create .env file
```bash
# Copy the example file
cp .env.example .env

# Edit the .env file and add your token
nano .env  # or use any text editor
```

Your `.env` file should look like this:
```
HF_TOKEN=hf_your_token_here
HUGGINGFACE_TOKEN=hf_your_token_here
```

### 5. Run the application
```bash
python app.py
```

Open http://localhost:5001 in your browser.

## Troubleshooting

### "Access denied" or "401 Unauthorized" error
- Make sure you accepted the PyAnnote model terms (Step 2)
- Check that your token is correct in the `.env` file
- Token must start with `hf_`

### "No module named..." error
```bash
pip install -r requirements.txt
```

### FFmpeg not found
Install FFmpeg:
- **Mac**: `brew install ffmpeg`
- **Ubuntu**: `sudo apt install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html

## Usage

1. Open the web interface at http://localhost:5001
2. Upload an audio or video file
3. Wait for processing (progress shown in real-time)
4. View results:
   - For audio files: Audio metrics, speaker diarization, transcript
   - For video files: All of the above + video engagement metrics

## API Endpoints

- `GET /` - Web interface
- `POST /upload` - Upload and analyze file
- `GET /progress/<task_id>` - Get analysis progress
- `GET /result/<task_id>` - Get analysis results

## Tech Stack

- **Backend:** Python, Flask
- **Speech Recognition:** OpenAI Whisper
- **Speaker Diarization:** PyAnnote Audio
- **Video Analysis:** OpenCV, MediaPipe
- **Audio Processing:** librosa, webrtcvad

## Requirements

- Python 3.9+
- FFmpeg (for audio extraction)
- HuggingFace account (for PyAnnote models)

## License

MIT License
