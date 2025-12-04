"""
Arali.ai Video Engagement Analyzer
Production-grade video analysis with OpenCV and advanced ML

ROBUST DETECTION FEATURES:
- Per-person baseline calibration (first 3-5 seconds)
- EWMA smoothing for all metrics
- Hysteresis thresholds to prevent flickering
- Percentage-based change detection (not absolute values)
- Adaptive thresholds based on individual variance
- Minimum face size validation
- Rolling window comparisons (1 second lookback)
"""

from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import mediapipe as mp
from scipy.spatial import distance
from collections import deque
import statistics
import subprocess
import tempfile
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()
print(f"🔑 HF_TOKEN loaded: {'Yes' if os.environ.get('HF_TOKEN') else 'No'}")

# Audio processing imports
try:
    import librosa
    import webrtcvad
    import parselmouth
    from parselmouth.praat import call
    AUDIO_AVAILABLE = True
    print("✅ Audio processing libraries loaded successfully!")
except ImportError as e:
    AUDIO_AVAILABLE = False
    print(f"⚠️ Audio processing libraries not available: {e}")

# Speaker diarization imports
try:
    from pyannote.audio import Pipeline
    import torch
    DIARIZATION_AVAILABLE = True
    print("✅ Speaker diarization libraries loaded successfully!")
except ImportError as e:
    DIARIZATION_AVAILABLE = False
    print(f"⚠️ Speaker diarization libraries not available: {e}")

# Speech-to-text (Whisper) imports
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper speech-to-text library loaded successfully!")
except ImportError as e:
    WHISPER_AVAILABLE = False
    print(f"⚠️ Whisper library not available: {e}")

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max for long meetings

# Configure screenshots folder
SCREENSHOTS_FOLDER = Path('static/screenshots')
SCREENSHOTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Global progress tracking
analysis_progress = {}

# MediaPipe Face Mesh for facial landmark detection
# max_num_faces=3 to detect main speaker + thumbnails in video calls
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=3,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


class EWMASmooth:
    """Exponentially Weighted Moving Average for smoothing noisy signals"""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def get(self):
        return self.value


class RollingWindow:
    """Rolling window for storing historical values"""

    def __init__(self, max_size=30):
        self.data = deque(maxlen=max_size)

    def add(self, value):
        self.data.append(value)

    def get_median(self):
        if not self.data:
            return None
        return statistics.median(self.data)

    def get_mean(self):
        if not self.data:
            return None
        return statistics.mean(self.data)

    def get_std(self):
        if len(self.data) < 2:
            return 0
        return statistics.stdev(self.data)

    def is_full(self):
        return len(self.data) == self.data.maxlen

    def get_value_n_ago(self, n):
        """Get value from n frames ago"""
        if len(self.data) > n:
            return self.data[-(n+1)]
        return None


class HysteresisState:
    """Hysteresis state machine to prevent flickering between states"""

    def __init__(self, enter_threshold, exit_threshold, initial_state=False):
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.state = initial_state

    def update(self, value):
        if not self.state and value >= self.enter_threshold:
            self.state = True
        elif self.state and value <= self.exit_threshold:
            self.state = False
        return self.state


class AudioAnalyzer:
    """
    Production-grade audio analyzer for engagement metrics.

    Features:
    - Voice Activity Detection (VAD) using WebRTC VAD
    - Speech Rate estimation (syllables/words per minute)
    - Prosody/Pitch analysis (F0 mean, std, range)
    - Energy/Loudness metrics (RMS, dynamic range)
    - Pause duration analysis
    """

    # Configuration
    SAMPLE_RATE = 16000  # 16kHz for speech processing
    VAD_MODE = 3  # 0-3, 3 is most aggressive
    VAD_FRAME_MS = 30  # Frame duration in ms (10, 20, or 30)
    MIN_SPEECH_DURATION_MS = 150  # Minimum speech segment duration
    MIN_PAUSE_DURATION_MS = 200  # Minimum pause to count
    MERGE_GAP_MS = 200  # Merge speech segments with gaps smaller than this

    def __init__(self, video_path, task_id=None):
        self.video_path = video_path
        self.task_id = task_id
        self.audio_path = None
        self.audio_data = None
        self.sr = self.SAMPLE_RATE

        # Results storage - keys match frontend expectations
        self.results = {
            'success': False,  # Set to True after successful analysis
            'audio_available': False,
            'duration_seconds': 0,
            'speech_segments': [],
            'speech_rate': {
                'syllables_per_minute': 0,
                'wpm': 0,  # Frontend expects 'wpm'
                'speaking_time_seconds': 0,
                'speaking_percentage': 0
            },
            'voice_activity': {  # Frontend expects 'voice_activity'
                'speaking_percent': 0,
                'total_speaking_seconds': 0,
                'segments_count': 0
            },
            'prosody': {
                'mean_pitch_hz': 0,  # Frontend expects 'mean_pitch_hz'
                'min_pitch_hz': 0,
                'max_pitch_hz': 0,
                'pitch_std_hz': 0,
                'variation_score': 0,  # Frontend expects 'variation_score' (0-100)
                'pattern': 'unknown'  # Frontend expects 'pattern'
            },
            'energy': {
                'mean_db': 0,
                'max_db': 0,
                'dynamic_range_db': 0,
                'energy_variation': 0,
                'loud_moments': []
            },
            'pauses': {
                'count': 0,  # Frontend expects 'count'
                'avg_duration_ms': 0,
                'long_pause_count': 0,
                'pause_ratio': 0,
                'pause_segments': []
            }
        }

    def extract_audio(self):
        """Extract audio from video using ffmpeg"""
        try:
            # Create temp file for audio
            fd, self.audio_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)

            # Use ffmpeg to extract audio
            cmd = [
                'ffmpeg', '-y', '-i', str(self.video_path),
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM format
                '-ar', str(self.SAMPLE_RATE),  # Sample rate
                '-ac', '1',  # Mono
                self.audio_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                print(f"⚠️ FFmpeg error: {result.stderr[:200]}")
                return False

            # Load audio with librosa
            self.audio_data, self.sr = librosa.load(self.audio_path, sr=self.SAMPLE_RATE, mono=True)
            self.results['duration_seconds'] = len(self.audio_data) / self.sr
            self.results['audio_available'] = True

            print(f"🔊 Audio extracted: {self.results['duration_seconds']:.2f}s, {self.sr}Hz")
            return True

        except Exception as e:
            print(f"⚠️ Audio extraction failed: {e}")
            return False

    def detect_voice_activity(self):
        """Detect speech segments using WebRTC VAD"""
        if self.audio_data is None:
            return []

        try:
            vad = webrtcvad.Vad(self.VAD_MODE)

            # Convert to 16-bit PCM
            audio_int16 = (self.audio_data * 32767).astype(np.int16)

            # Frame parameters
            frame_samples = int(self.sr * self.VAD_FRAME_MS / 1000)
            frame_bytes = frame_samples * 2  # 16-bit = 2 bytes

            speech_frames = []

            # Process each frame
            for i in range(0, len(audio_int16) - frame_samples, frame_samples):
                frame = audio_int16[i:i + frame_samples].tobytes()
                is_speech = vad.is_speech(frame, self.sr)
                timestamp_ms = (i / self.sr) * 1000
                speech_frames.append((timestamp_ms, is_speech))

            # Merge consecutive speech frames into segments
            segments = []
            segment_start = None

            for timestamp_ms, is_speech in speech_frames:
                if is_speech and segment_start is None:
                    segment_start = timestamp_ms
                elif not is_speech and segment_start is not None:
                    # Check if gap is small enough to merge
                    if segments and (timestamp_ms - segments[-1]['end']) < self.MERGE_GAP_MS:
                        segments[-1]['end'] = timestamp_ms
                    else:
                        if (timestamp_ms - segment_start) >= self.MIN_SPEECH_DURATION_MS:
                            segments.append({
                                'start': segment_start,
                                'end': timestamp_ms,
                                'duration': timestamp_ms - segment_start
                            })
                    segment_start = None

            # Handle last segment
            if segment_start is not None:
                end_time = len(self.audio_data) / self.sr * 1000
                if (end_time - segment_start) >= self.MIN_SPEECH_DURATION_MS:
                    segments.append({
                        'start': segment_start,
                        'end': end_time,
                        'duration': end_time - segment_start
                    })

            self.results['speech_segments'] = segments

            # Calculate speaking time
            total_speech_ms = sum(s['duration'] for s in segments)
            total_speech_sec = total_speech_ms / 1000
            speaking_pct = round(
                (total_speech_ms / (self.results['duration_seconds'] * 1000)) * 100, 1
            ) if self.results['duration_seconds'] > 0 else 0

            # Update both speech_rate and voice_activity for frontend compatibility
            self.results['speech_rate']['speaking_time_seconds'] = total_speech_sec
            self.results['speech_rate']['speaking_percentage'] = speaking_pct
            self.results['voice_activity']['speaking_percent'] = speaking_pct
            self.results['voice_activity']['total_speaking_seconds'] = total_speech_sec
            self.results['voice_activity']['segments_count'] = len(segments)

            print(f"🎙️ VAD: {len(segments)} speech segments, {total_speech_sec:.1f}s speaking time")
            return segments

        except Exception as e:
            print(f"⚠️ VAD failed: {e}")
            return []

    def analyze_speech_rate(self):
        """Estimate speech rate using energy envelope peaks (syllable proxy)"""
        if self.audio_data is None or not self.results['speech_segments']:
            return

        try:
            # Use onset detection as syllable proxy
            onset_env = librosa.onset.onset_strength(y=self.audio_data, sr=self.sr)
            onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr, units='time')

            # Count onsets only within speech segments
            syllable_count = 0
            for onset_time in onsets:
                onset_ms = onset_time * 1000
                for seg in self.results['speech_segments']:
                    if seg['start'] <= onset_ms <= seg['end']:
                        syllable_count += 1
                        break

            # Calculate rates
            speaking_minutes = self.results['speech_rate']['speaking_time_seconds'] / 60
            if speaking_minutes > 0:
                syllables_per_min = syllable_count / speaking_minutes
                # Approximate WPM (average 1.5 syllables per word)
                estimated_wpm = syllables_per_min / 1.5

                self.results['speech_rate']['syllables_per_minute'] = round(syllables_per_min, 1)
                self.results['speech_rate']['wpm'] = round(estimated_wpm, 1)  # Frontend expects 'wpm'

                print(f"📊 Speech Rate: ~{estimated_wpm:.0f} WPM ({syllables_per_min:.0f} syllables/min)")

        except Exception as e:
            print(f"⚠️ Speech rate analysis failed: {e}")

    def analyze_prosody(self):
        """Analyze pitch/prosody using Parselmouth (Praat)"""
        if self.audio_data is None:
            return

        try:
            # Create Parselmouth Sound object
            snd = parselmouth.Sound(self.audio_data, sampling_frequency=self.sr)

            # Extract pitch
            pitch = call(snd, "To Pitch", 0.0, 75, 600)  # time_step, min_pitch, max_pitch

            # Get pitch values (excluding unvoiced frames)
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced

            if len(pitch_values) > 10:
                # Calculate statistics
                pitch_mean = np.mean(pitch_values)
                pitch_std = np.std(pitch_values)
                pitch_5th = np.percentile(pitch_values, 5)
                pitch_95th = np.percentile(pitch_values, 95)
                pitch_range = pitch_95th - pitch_5th

                # Update prosody with frontend-compatible keys
                self.results['prosody']['mean_pitch_hz'] = round(pitch_mean, 1)
                self.results['prosody']['min_pitch_hz'] = round(pitch_5th, 1)
                self.results['prosody']['max_pitch_hz'] = round(pitch_95th, 1)
                self.results['prosody']['pitch_std_hz'] = round(pitch_std, 1)

                # Calculate variation score (0-100)
                # Convert to semitones for better comparison
                pitch_std_semitones = 12 * np.log2((pitch_mean + pitch_std) / pitch_mean) if pitch_mean > 0 else 0

                # Score: <0.5 semitones = monotone (0), >2.0 semitones = very expressive (100)
                variation_score = min(100, max(0, (pitch_std_semitones / 2.0) * 100))
                self.results['prosody']['variation_score'] = round(variation_score, 1)  # Frontend expects 'variation_score'

                # Classify intonation pattern - frontend expects 'pattern'
                if variation_score < 30:
                    pattern = 'monotone'
                elif variation_score < 60:
                    pattern = 'varied'
                else:
                    pattern = 'expressive'
                self.results['prosody']['pattern'] = pattern

                print(f"🎵 Prosody: Mean pitch {pitch_mean:.0f}Hz, Variation score {variation_score:.0f}/100 ({pattern})")

        except Exception as e:
            print(f"⚠️ Prosody analysis failed: {e}")

    def analyze_energy(self):
        """Analyze energy/loudness levels"""
        if self.audio_data is None:
            return

        try:
            # Calculate RMS energy
            frame_length = int(self.sr * 0.025)  # 25ms frames
            hop_length = int(self.sr * 0.010)  # 10ms hop

            rms = librosa.feature.rms(y=self.audio_data, frame_length=frame_length, hop_length=hop_length)[0]

            # Convert to dB (avoid log(0))
            rms_db = 20 * np.log10(rms + 1e-10)

            # Filter to only speech segments
            speech_rms = []
            times = librosa.frames_to_time(np.arange(len(rms)), sr=self.sr, hop_length=hop_length)

            for i, t in enumerate(times):
                t_ms = t * 1000
                for seg in self.results['speech_segments']:
                    if seg['start'] <= t_ms <= seg['end']:
                        speech_rms.append(rms_db[i])
                        break

            if speech_rms:
                speech_rms = np.array(speech_rms)

                self.results['energy']['mean_db'] = round(np.mean(speech_rms), 1)
                self.results['energy']['max_db'] = round(np.max(speech_rms), 1)
                self.results['energy']['dynamic_range_db'] = round(np.max(speech_rms) - np.mean(speech_rms), 1)
                self.results['energy']['energy_variation'] = round(np.std(speech_rms), 1)

                # Find loud moments (>6dB above mean)
                mean_energy = np.mean(speech_rms)
                loud_threshold = mean_energy + 6

                loud_moments = []
                for i, (t, db) in enumerate(zip(times, rms_db)):
                    if db > loud_threshold:
                        loud_moments.append(round(t, 2))

                # Limit to 20 loud moments
                self.results['energy']['loud_moments'] = loud_moments[:20]

                print(f"🔊 Energy: Mean {self.results['energy']['mean_db']:.1f}dB, Dynamic range {self.results['energy']['dynamic_range_db']:.1f}dB")

        except Exception as e:
            print(f"⚠️ Energy analysis failed: {e}")

    def analyze_pauses(self):
        """Analyze pause durations between speech segments"""
        if not self.results['speech_segments']:
            return

        try:
            segments = sorted(self.results['speech_segments'], key=lambda x: x['start'])
            pauses = []

            # Find pauses between segments
            for i in range(1, len(segments)):
                pause_start = segments[i-1]['end']
                pause_end = segments[i]['start']
                pause_duration = pause_end - pause_start

                if pause_duration >= self.MIN_PAUSE_DURATION_MS:
                    pauses.append({
                        'start': pause_start,
                        'end': pause_end,
                        'duration': pause_duration
                    })

            if pauses:
                total_pause_ms = sum(p['duration'] for p in pauses)
                avg_pause_ms = total_pause_ms / len(pauses)
                long_pauses = [p for p in pauses if p['duration'] > 1000]

                self.results['pauses']['count'] = len(pauses)  # Frontend expects 'count'
                self.results['pauses']['avg_duration_ms'] = round(avg_pause_ms, 0)
                self.results['pauses']['long_pause_count'] = len(long_pauses)
                self.results['pauses']['pause_ratio'] = round(
                    (total_pause_ms / (self.results['duration_seconds'] * 1000)) * 100, 1
                ) if self.results['duration_seconds'] > 0 else 0
                self.results['pauses']['pause_segments'] = pauses[:50]  # Limit stored pauses

                print(f"⏸️ Pauses: {len(pauses)} total, avg {avg_pause_ms:.0f}ms, {len(long_pauses)} long pauses")

        except Exception as e:
            print(f"⚠️ Pause analysis failed: {e}")

    def _convert_to_native(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def analyze(self):
        """Run full audio analysis pipeline"""
        if not AUDIO_AVAILABLE:
            print("⚠️ Audio libraries not available, skipping audio analysis")
            return self.results

        print("\n🎵 Starting audio analysis...")

        # Extract audio from video
        if not self.extract_audio():
            return self.results

        # Run analysis pipeline
        self.detect_voice_activity()
        self.analyze_speech_rate()
        self.analyze_prosody()
        self.analyze_energy()
        self.analyze_pauses()

        # NOTE: Don't cleanup temp file here - it's needed for diarization and transcription
        # Cleanup will be done by the calling code after all audio processing is complete

        # Mark as successful
        self.results['success'] = True
        print("✅ Audio analysis complete\n")
        # Convert numpy types to native Python types
        return self._convert_to_native(self.results)


class SpeakerDiarizer:
    """
    Speaker Diarization using pyannote.audio

    Identifies who spoke when in multi-speaker audio.
    Requires Hugging Face authentication token.

    Features:
    - Automatic speaker detection
    - Per-speaker speaking time
    - Speaker segments timeline
    - Speaker overlap detection
    """

    # Hugging Face token - set via environment variable
    HF_TOKEN = os.environ.get('HF_TOKEN', None)

    def __init__(self, audio_path, task_id=None):
        self.audio_path = audio_path
        self.task_id = task_id
        self.pipeline = None
        self.diarization = None

        # Results storage
        self.results = {
            'success': False,
            'num_speakers': 0,
            'speakers': {},  # speaker_id -> {speaking_time, percent, segments}
            'timeline': [],  # [{start, end, speaker, duration}]
            'overlaps': [],  # [{start, end, speakers}]
            'total_duration': 0,
            'error': None
        }

    def load_pipeline(self):
        """Load pyannote diarization pipeline"""
        if not DIARIZATION_AVAILABLE:
            self.results['error'] = "Diarization libraries not available"
            return False

        if not self.HF_TOKEN:
            self.results['error'] = "Hugging Face token not set. Set HF_TOKEN environment variable."
            print("⚠️ HF_TOKEN not set. Get token from https://huggingface.co/settings/tokens")
            return False

        try:
            print("🔄 Loading speaker diarization model...")
            # Use pyannote/speaker-diarization-3.1 (latest stable)
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.HF_TOKEN
            )

            # Use MPS (Metal) on Mac M1/M2/M3 if available, else CPU
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                print("🚀 Using Apple Metal (MPS) for acceleration")
            else:
                device = torch.device("cpu")
                print("💻 Using CPU for processing")

            self.pipeline.to(device)
            print("✅ Diarization model loaded successfully!")
            return True

        except Exception as e:
            self.results['error'] = f"Failed to load diarization model: {str(e)}"
            print(f"⚠️ {self.results['error']}")
            return False

    def run_diarization(self, min_speakers=1, max_speakers=10):
        """Run speaker diarization on audio file

        Args:
            min_speakers: Minimum expected speakers (default 1)
            max_speakers: Maximum expected speakers (default 10 for large meetings)
        """
        if self.pipeline is None:
            return False

        try:
            print(f"🎙️ Running speaker diarization on: {self.audio_path}")
            print(f"   Expected speakers: {min_speakers}-{max_speakers}")

            # Run diarization with speaker constraints
            # This helps avoid over-segmentation (detecting too many speakers)
            self.diarization = self.pipeline(
                self.audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )

            # Process results
            speakers = {}
            timeline = []
            total_duration = 0

            for turn, _, speaker in self.diarization.itertracks(yield_label=True):
                start = turn.start
                end = turn.end
                duration = end - start

                # Update speaker stats
                if speaker not in speakers:
                    speakers[speaker] = {
                        'speaking_time': 0,
                        'segments_count': 0,
                        'segments': []
                    }

                speakers[speaker]['speaking_time'] += duration
                speakers[speaker]['segments_count'] += 1
                speakers[speaker]['segments'].append({
                    'start': round(start, 2),
                    'end': round(end, 2),
                    'duration': round(duration, 2)
                })

                # Add to timeline
                timeline.append({
                    'start': round(start, 2),
                    'end': round(end, 2),
                    'speaker': speaker,
                    'duration': round(duration, 2)
                })

                if end > total_duration:
                    total_duration = end

            # Calculate percentages
            for speaker_id, data in speakers.items():
                data['percent'] = round((data['speaking_time'] / total_duration) * 100, 1) if total_duration > 0 else 0
                data['speaking_time'] = round(data['speaking_time'], 2)
                # Limit segments stored to prevent huge JSON
                data['segments'] = data['segments'][:50]

            # Post-process: Merge minor speakers (< 5% speaking time) into nearest major speaker
            # This handles pyannote over-segmentation in long audio
            speakers, timeline = self._merge_minor_speakers(speakers, timeline, threshold_percent=5.0)

            # Detect overlaps (when multiple speakers talk simultaneously)
            overlaps = self._detect_overlaps(timeline)

            # Update results
            self.results['success'] = True
            self.results['num_speakers'] = len(speakers)
            self.results['speakers'] = speakers
            self.results['timeline'] = timeline  # Keep all timeline entries for accurate alignment
            self.results['overlaps'] = overlaps[:50]
            self.results['total_duration'] = round(total_duration, 2)

            print(f"✅ Diarization complete!")
            print(f"   Speakers detected: {len(speakers)}")
            for sid, data in speakers.items():
                print(f"   - {sid}: {data['speaking_time']:.1f}s ({data['percent']}%)")

            return True

        except Exception as e:
            self.results['error'] = f"Diarization failed: {str(e)}"
            print(f"⚠️ {self.results['error']}")
            import traceback
            traceback.print_exc()
            return False

    def _merge_minor_speakers(self, speakers, timeline, threshold_percent=5.0):
        """
        Merge speakers with less than threshold_percent speaking time into the most similar major speaker.
        This handles pyannote's over-segmentation issue in long audio files.

        Logic: If someone speaks less than 5% of total time in a 30+ min meeting,
        they're likely a misdetection of one of the main speakers.
        """
        if len(speakers) <= 2:
            # Already at expected speaker count for typical meetings
            return speakers, timeline

        # Identify major speakers (> threshold%)
        major_speakers = {sid: data for sid, data in speakers.items() if data['percent'] >= threshold_percent}
        minor_speakers = {sid: data for sid, data in speakers.items() if data['percent'] < threshold_percent}

        if not minor_speakers:
            return speakers, timeline

        if not major_speakers:
            # Edge case: all speakers are minor, keep the top 2
            sorted_speakers = sorted(speakers.items(), key=lambda x: x[1]['speaking_time'], reverse=True)
            major_speakers = dict(sorted_speakers[:2])
            minor_speakers = dict(sorted_speakers[2:])

        print(f"🔄 Merging {len(minor_speakers)} minor speakers into {len(major_speakers)} major speakers")

        # Map each minor speaker to the major speaker they most likely belong to
        # Strategy: Assign to the major speaker with most segments nearby
        speaker_mapping = {}

        for minor_sid in minor_speakers:
            # Find which major speaker's segments are closest to this minor speaker's segments
            minor_segments = minor_speakers[minor_sid].get('segments', [])

            best_major = None
            best_score = -1

            for major_sid in major_speakers:
                major_segments = major_speakers[major_sid].get('segments', [])

                # Score based on temporal proximity
                proximity_score = 0
                for minor_seg in minor_segments:
                    for major_seg in major_segments:
                        # Check if segments are close in time (within 5 seconds)
                        time_gap = min(
                            abs(minor_seg['start'] - major_seg['end']),
                            abs(minor_seg['end'] - major_seg['start'])
                        )
                        if time_gap < 5:
                            proximity_score += 1

                if proximity_score > best_score:
                    best_score = proximity_score
                    best_major = major_sid

            # If no proximity found, assign to the speaker with most speaking time
            if best_major is None:
                best_major = max(major_speakers.keys(), key=lambda x: major_speakers[x]['speaking_time'])

            speaker_mapping[minor_sid] = best_major
            print(f"   {minor_sid} ({minor_speakers[minor_sid]['percent']}%) → {best_major}")

        # Update timeline with merged speakers
        new_timeline = []
        for seg in timeline:
            new_seg = seg.copy()
            if seg['speaker'] in speaker_mapping:
                new_seg['speaker'] = speaker_mapping[seg['speaker']]
            new_timeline.append(new_seg)

        # Rebuild speaker stats from merged timeline
        merged_speakers = {}
        total_duration = max(seg['end'] for seg in new_timeline) if new_timeline else 0

        for seg in new_timeline:
            sid = seg['speaker']
            if sid not in merged_speakers:
                merged_speakers[sid] = {
                    'speaking_time': 0,
                    'segments_count': 0,
                    'segments': []
                }
            merged_speakers[sid]['speaking_time'] += seg['duration']
            merged_speakers[sid]['segments_count'] += 1
            if len(merged_speakers[sid]['segments']) < 50:
                merged_speakers[sid]['segments'].append(seg)

        # Recalculate percentages
        for sid, data in merged_speakers.items():
            data['percent'] = round((data['speaking_time'] / total_duration) * 100, 1) if total_duration > 0 else 0
            data['speaking_time'] = round(data['speaking_time'], 2)

        # Rename speakers to be sequential (SPEAKER_00, SPEAKER_01, etc.)
        final_speakers = {}
        final_timeline = []
        speaker_rename = {}

        for i, (old_sid, data) in enumerate(sorted(merged_speakers.items(), key=lambda x: x[1]['speaking_time'], reverse=True)):
            new_sid = f"SPEAKER_{i:02d}"
            speaker_rename[old_sid] = new_sid
            final_speakers[new_sid] = data

        for seg in new_timeline:
            new_seg = seg.copy()
            new_seg['speaker'] = speaker_rename.get(seg['speaker'], seg['speaker'])
            final_timeline.append(new_seg)

        print(f"✅ Merged to {len(final_speakers)} speakers")

        return final_speakers, final_timeline

    def _detect_overlaps(self, timeline):
        """Detect overlapping speech segments"""
        overlaps = []

        # Sort by start time
        sorted_segments = sorted(timeline, key=lambda x: x['start'])

        for i, seg1 in enumerate(sorted_segments):
            for seg2 in sorted_segments[i+1:]:
                # Check if segments overlap
                if seg2['start'] < seg1['end']:
                    overlap_start = seg2['start']
                    overlap_end = min(seg1['end'], seg2['end'])

                    if overlap_end > overlap_start:
                        overlaps.append({
                            'start': round(overlap_start, 2),
                            'end': round(overlap_end, 2),
                            'duration': round(overlap_end - overlap_start, 2),
                            'speakers': [seg1['speaker'], seg2['speaker']]
                        })
                else:
                    # No more overlaps possible with seg1
                    break

        return overlaps

    def analyze(self, min_speakers=1, max_speakers=10):
        """Run full speaker diarization pipeline

        Args:
            min_speakers: Minimum expected speakers (default 1)
            max_speakers: Maximum expected speakers (default 10)
                         Supports up to 10 speakers for large meetings/conferences
        """
        print("\n🎭 Starting speaker diarization...")

        if not self.load_pipeline():
            return self.results

        self.run_diarization(min_speakers=min_speakers, max_speakers=max_speakers)

        return self.results


class SpeechTranscriber:
    """
    Speech-to-Text using OpenAI Whisper

    Transcribes audio to text with timestamps.
    Can align transcription with speaker diarization.
    """

    def __init__(self, audio_path, task_id=None, model_size="base"):
        """
        Initialize transcriber

        model_size options:
        - "tiny": Fastest, lowest quality (~1GB VRAM)
        - "base": Good balance of speed/quality (~1GB VRAM)
        - "small": Better quality (~2GB VRAM)
        - "medium": High quality (~5GB VRAM)
        - "large": Best quality (~10GB VRAM)
        """
        self.audio_path = audio_path
        self.task_id = task_id
        self.model_size = model_size
        self.model = None

        self.results = {
            'success': False,
            'text': '',
            'segments': [],  # [{start, end, text}]
            'language': None,
            'error': None
        }

    def load_model(self):
        """Load Whisper model"""
        if not WHISPER_AVAILABLE:
            self.results['error'] = "Whisper library not available"
            return False

        try:
            print(f"🔄 Loading Whisper model ({self.model_size})...")
            self.model = whisper.load_model(self.model_size)
            print("✅ Whisper model loaded successfully!")
            return True
        except Exception as e:
            self.results['error'] = f"Failed to load Whisper model: {str(e)}"
            print(f"⚠️ {self.results['error']}")
            return False

    def transcribe(self):
        """Transcribe audio to text with high accuracy"""
        if self.model is None:
            return False

        try:
            print(f"🎤 Transcribing audio: {self.audio_path}")

            # Transcribe with HIGH ACCURACY settings
            # - word_timestamps=True for precise word-level timing
            # - condition_on_previous_text=True for better context understanding
            # - temperature=0 for more deterministic output
            result = self.model.transcribe(
                self.audio_path,
                language=None,  # Auto-detect language
                verbose=False,
                task="transcribe",
                fp16=False,  # Use fp32 for CPU compatibility
                word_timestamps=True,  # ACCURATE - word-level timestamps for better speaker alignment
                condition_on_previous_text=False,  # DISABLE - prevents "Thank you" hallucination loops
                temperature=0,  # More deterministic, less hallucination
                compression_ratio_threshold=2.4,  # Filter out bad segments
                logprob_threshold=-1.0,  # Filter low confidence
                no_speech_threshold=0.6,  # Better silence detection
                hallucination_silence_threshold=0.5,  # Skip segments with >0.5s silence (anti-hallucination)
                # Multi-language support prompt - avoid repetitive phrases
                initial_prompt=None  # No prompt - avoids hallucination bias
            )

            # Log detected language
            detected_lang = result.get('language', 'unknown')
            print(f"   Detected language: {detected_lang}")

            # Extract results
            self.results['success'] = True
            self.results['text'] = result['text'].strip()
            self.results['language'] = result.get('language', 'unknown')

            # Extract segments with timestamps and filter hallucinations
            segments = []
            # Common hallucination phrases to filter
            hallucination_phrases = [
                "thank you", "thanks for watching", "please subscribe",
                "like and subscribe", "see you next time", "bye",
                "music", "applause", "[music]", "[applause]"
            ]

            for seg in result['segments']:
                text = seg['text'].strip()

                # Skip empty segments
                if not text:
                    continue

                # Skip very short repeated hallucinations
                text_lower = text.lower().strip()
                if text_lower in hallucination_phrases:
                    continue

                # Skip segments that are too short (less than 0.1s duration)
                duration = seg['end'] - seg['start']
                if duration < 0.1:
                    continue

                segments.append({
                    'start': round(seg['start'], 2),
                    'end': round(seg['end'], 2),
                    'text': text
                })

            self.results['segments'] = segments
            print(f"   Filtered segments: {len(result['segments'])} -> {len(segments)}")

            print(f"✅ Transcription complete!")
            print(f"   Language: {self.results['language']}")
            print(f"   Segments: {len(segments)}")
            print(f"   Text length: {len(self.results['text'])} chars")

            return True

        except Exception as e:
            self.results['error'] = f"Transcription failed: {str(e)}"
            print(f"⚠️ {self.results['error']}")
            import traceback
            traceback.print_exc()
            return False

    def align_with_diarization(self, diarization_results):
        """
        Align transcription segments with speaker diarization

        Uses improved alignment:
        1. Find speaker with maximum overlap for each segment
        2. If multiple speakers, check which speaker covers segment start
        3. Use binary search for faster lookup in long timelines
        """
        if not self.results['success'] or not diarization_results.get('success'):
            return self.results['segments']

        aligned_segments = []
        diar_timeline = diarization_results.get('timeline', [])

        if not diar_timeline:
            return self.results['segments']

        # Sort timeline by start time for efficient lookup
        diar_timeline = sorted(diar_timeline, key=lambda x: x['start'])

        # Get list of unique speakers for fallback
        unique_speakers = list(set(seg['speaker'] for seg in diar_timeline))
        last_known_speaker = unique_speakers[0] if unique_speakers else "Speaker 00"

        def find_speaker_at_time(t):
            """Find which speaker is talking at time t"""
            for diar_seg in diar_timeline:
                if diar_seg['start'] <= t <= diar_seg['end']:
                    return diar_seg['speaker']
            return None

        def find_dominant_speaker(seg_start, seg_end):
            """Find the speaker with most overlap in this segment"""
            speaker_overlap = {}

            for diar_seg in diar_timeline:
                # Skip if completely outside our segment
                if diar_seg['end'] < seg_start or diar_seg['start'] > seg_end:
                    continue

                # Calculate overlap
                overlap_start = max(seg_start, diar_seg['start'])
                overlap_end = min(seg_end, diar_seg['end'])
                overlap = max(0, overlap_end - overlap_start)

                if overlap > 0:
                    spk = diar_seg['speaker']
                    speaker_overlap[spk] = speaker_overlap.get(spk, 0) + overlap

            if speaker_overlap:
                # Return speaker with maximum overlap
                return max(speaker_overlap.items(), key=lambda x: x[1])[0]
            return None

        for seg in self.results['segments']:
            seg_start = seg['start']
            seg_end = seg['end']

            # Method 1: Find dominant speaker by overlap
            speaker = find_dominant_speaker(seg_start, seg_end)

            # Method 2: If no overlap, find speaker at segment start
            if speaker is None:
                speaker = find_speaker_at_time(seg_start)

            # Method 3: Find speaker at segment midpoint
            if speaker is None:
                seg_mid = (seg_start + seg_end) / 2
                speaker = find_speaker_at_time(seg_mid)

            # Method 4: Find nearest diarization segment
            if speaker is None:
                min_distance = float('inf')
                seg_mid = (seg_start + seg_end) / 2
                for diar_seg in diar_timeline:
                    diar_mid = (diar_seg['start'] + diar_seg['end']) / 2
                    distance = abs(seg_mid - diar_mid)
                    if distance < min_distance:
                        min_distance = distance
                        speaker = diar_seg['speaker']

            # Method 5: Use last known speaker
            if speaker is None:
                speaker = last_known_speaker
            else:
                last_known_speaker = speaker

            aligned_segments.append({
                'start': seg_start,
                'end': seg_end,
                'text': seg['text'],
                'speaker': speaker
            })

        return aligned_segments

    def analyze(self):
        """Run full transcription pipeline"""
        print("\n🎤 Starting speech transcription...")

        if not self.load_model():
            return self.results

        self.transcribe()

        return self.results


class VideoAnalyzer:
    """
    Production-grade video analyzer with robust engagement metrics.

    Features:
    - Calibration phase (first 3-5 seconds) to establish per-person baselines
    - EWMA smoothing on all metrics to reduce noise
    - Hysteresis thresholds to prevent state flickering
    - Rolling window comparisons (compare to 1 second ago, not previous frame)
    - Adaptive thresholds based on individual variance
    - Minimum face size validation (reject unreliable detections)
    - Optimized for long videos (50+ minutes)
    - Focus on largest face (main speaker in video calls)
    """

    # Configuration constants
    CALIBRATION_SECONDS = 4  # Seconds for baseline calibration
    MIN_FACE_HEIGHT_PX = 50  # Minimum face height in pixels for reliable detection
    EWMA_ALPHA = 0.3  # Smoothing factor (0.2-0.4 recommended)

    # Long video optimization
    MAX_SCREENSHOTS = 200  # Limit screenshots to prevent memory issues
    MAX_TIMELINE_EVENTS = 500  # Limit timeline events
    PROGRESS_UPDATE_INTERVAL = 50  # Update progress every N frames

    # Lean detection thresholds (percentage change from baseline)
    LEAN_FORWARD_ENTER = 0.15  # +15% to trigger forward
    LEAN_FORWARD_EXIT = 0.10   # +10% to clear forward
    LEAN_BACKWARD_ENTER = -0.15  # -15% to trigger backward
    LEAN_BACKWARD_EXIT = -0.10   # -10% to clear backward

    # Blink detection - more sensitive for real-world videos
    EAR_BLINK_THRESHOLD = 0.25  # Eye Aspect Ratio threshold (higher = more sensitive)
    BLINK_CONSECUTIVE_FRAMES = 1  # Minimum frames for valid blink (lower for 3rd frame sampling)

    # Smile detection thresholds (AU-based Duchenne smile detection)
    SMILE_CONFIDENCE_THRESHOLD = 0.65  # Combined AU score threshold
    SMILE_EXIT_THRESHOLD = 0.50  # Hysteresis exit threshold
    SMILE_MIN_DURATION_FRAMES = 2  # Minimum frames (at 3rd frame sampling = ~200ms at 30fps)

    # AU12 (Lip Corner Puller) weights
    AU12_WEIGHT = 0.5
    # AU6 (Eye Crinkle / Cheek Raiser) weight
    AU6_WEIGHT = 0.3
    # Mouth aspect ratio weight
    MOUTH_RATIO_WEIGHT = 0.2

    # Head gesture thresholds (relative to baseline movement)
    NOD_THRESHOLD_MULTIPLIER = 3.0  # Movement must be 3x baseline std
    SHAKE_THRESHOLD_MULTIPLIER = 3.0

    # Gaze stability
    GAZE_STABLE_MIN = 0.30  # Iris position ratio range for stable gaze
    GAZE_STABLE_MAX = 0.70

    def __init__(self, video_path, task_id=None):
        self.video_path = video_path
        self.task_id = task_id
        self.cap = cv2.VideoCapture(str(video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 if not detected
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Dynamic frame skip based on video duration for optimization
        # Short videos (<5 min): every 3rd frame
        # Medium videos (5-20 min): every 5th frame
        # Long videos (20-60 min): every 6th frame
        # Very long videos (>60 min): every 8th frame
        if self.duration < 300:  # < 5 min
            self.frame_skip = 3
        elif self.duration < 1200:  # 5-20 min
            self.frame_skip = 5
        elif self.duration < 3600:  # 20-60 min
            self.frame_skip = 6
        else:  # > 60 min
            self.frame_skip = 8

        print(f"📊 Video duration: {self.duration/60:.1f} min, using frame_skip={self.frame_skip}")

        # Calculate frames for 1 second lookback
        self.one_second_frames = int(self.fps / self.frame_skip)

        # Engagement data
        self.results = {
            'video_info': {
                'duration': self.duration,
                'fps': self.fps,
                'total_frames': self.total_frames,
                'resolution': f"{self.frame_width}x{self.frame_height}"
            },
            'visual_metrics': {
                'blinks': {'total': 0, 'per_minute': 0, 'timestamps': []},
                'smiles': {'total': 0, 'duration_seconds': 0, 'timestamps': []},
                'lean': {
                    'forward_percent': 0,
                    'backward_percent': 0,
                    'neutral_percent': 0,
                    'forward_events': 0,
                    'backward_events': 0,
                    'timeline': []
                },
                'head_gestures': {'nods': 0, 'shakes': 0, 'timestamps': []},
                'gaze': {'stable_percent': 0, 'avg_stability': 0, 'wandering_events': []},
                'tension': {'avg_level': 0, 'max_level': 0, 'timeline': []},
                'attention_score': 0
            },
            'calibration_info': {
                'baseline_face_area': 0,
                'baseline_ear': 0,
                'baseline_smile_ratio': 0,
                'calibration_frames': 0,
                'calibration_quality': 'unknown'
            },
            'overall_score': 0
        }

        # ========== SMOOTHING & ROLLING WINDOWS ==========
        # Face area for lean detection
        self.face_area_smoother = EWMASmooth(alpha=self.EWMA_ALPHA)
        self.face_area_window = RollingWindow(max_size=self.one_second_frames * 2)
        self.face_area_baseline_window = RollingWindow(max_size=100)  # For calibration

        # Eye Aspect Ratio for blink detection
        self.ear_smoother = EWMASmooth(alpha=0.6)  # Fast response for blinks (less smoothing)
        self.ear_baseline_window = RollingWindow(max_size=100)

        # Smile baseline (for mouth ratio comparison)
        self.smile_baseline_window = RollingWindow(max_size=100)

        # Head position for gesture detection
        self.nose_y_smoother = EWMASmooth(alpha=self.EWMA_ALPHA)
        self.nose_x_smoother = EWMASmooth(alpha=self.EWMA_ALPHA)
        self.nose_y_window = RollingWindow(max_size=self.one_second_frames)
        self.nose_x_window = RollingWindow(max_size=self.one_second_frames)
        self.vertical_movement_baseline = RollingWindow(max_size=100)
        self.horizontal_movement_baseline = RollingWindow(max_size=100)

        # Gaze tracking
        self.gaze_smoother = EWMASmooth(alpha=self.EWMA_ALPHA)
        self.gaze_stability_window = RollingWindow(max_size=self.one_second_frames * 2)

        # Tension tracking
        self.tension_smoother = EWMASmooth(alpha=self.EWMA_ALPHA)
        self.tension_baseline_window = RollingWindow(max_size=100)

        # ========== HYSTERESIS STATES ==========
        self.lean_forward_state = HysteresisState(
            enter_threshold=self.LEAN_FORWARD_ENTER,
            exit_threshold=self.LEAN_FORWARD_EXIT
        )
        self.lean_backward_state = HysteresisState(
            enter_threshold=abs(self.LEAN_BACKWARD_ENTER),
            exit_threshold=abs(self.LEAN_BACKWARD_EXIT)
        )
        # Note: Smile detection now uses manual hysteresis with AU-based confidence

        # ========== STATE TRACKING ==========
        self.frame_count = 0
        self.processed_frame_count = 0
        self.calibration_complete = False
        self.calibration_frames_needed = int(self.CALIBRATION_SECONDS * self.fps / self.frame_skip)

        # Baselines (set after calibration)
        self.baseline_face_area = None
        self.baseline_ear = None
        self.baseline_smile_ratio = None
        self.baseline_vertical_std = None
        self.baseline_horizontal_std = None
        self.baseline_tension = None

        # Blink detection state
        self.blink_frame_counter = 0
        self.in_blink = False

        # Previous states for event detection
        self.previous_lean_state = 'neutral'
        self.previous_smile_state = False
        self.previous_nose_y = None
        self.previous_nose_x = None

        # Counters for lean percentage calculation
        self.lean_forward_frames = 0
        self.lean_backward_frames = 0
        self.lean_neutral_frames = 0

        # Smile detection state (AU-based)
        self.smile_start_time = None
        self.total_smile_duration = 0
        self.smile_confidence_smoother = EWMASmooth(alpha=0.4)
        self.smile_frame_counter = 0  # Consecutive frames with smile detected
        self.current_smile_state = False
        self.baseline_lip_corner_y = None  # For AU12 calibration
        self.baseline_eye_cheek_dist = None  # For AU6 calibration

        # Gaze tracking
        self.gaze_stable_frames = 0
        self.gaze_total_frames = 0

        # Screenshot capture
        self.current_frame = None  # Store current frame for screenshot capture
        self.screenshot_count = 0
        self.screenshots_dir = SCREENSHOTS_FOLDER / (task_id or datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)

        # Detection events with screenshots
        self.detection_events = []

    def capture_screenshot(self, event_type, timestamp, details=""):
        """Capture current frame as screenshot and record the detection event"""
        if self.current_frame is None:
            return None

        # Limit screenshots to prevent memory issues on long videos
        if self.screenshot_count >= self.MAX_SCREENSHOTS:
            # Still record the event but without screenshot
            event = {
                'type': event_type,
                'timestamp': round(timestamp, 2),
                'details': details,
                'screenshot': None
            }
            if len(self.detection_events) < self.MAX_TIMELINE_EVENTS:
                self.detection_events.append(event)
            return None

        self.screenshot_count += 1
        filename = f"{event_type}_{timestamp:.2f}s_{self.screenshot_count}.jpg"
        filepath = self.screenshots_dir / filename

        # Resize for smaller file size (max width 400px)
        height, width = self.current_frame.shape[:2]
        if width > 400:
            scale = 400 / width
            new_width = 400
            new_height = int(height * scale)
            resized = cv2.resize(self.current_frame, (new_width, new_height))
        else:
            resized = self.current_frame

        # Save with compression
        cv2.imwrite(str(filepath), resized, [cv2.IMWRITE_JPEG_QUALITY, 70])

        # Create event record
        event = {
            'type': event_type,
            'timestamp': round(timestamp, 2),
            'details': details,
            'screenshot': f"/static/screenshots/{self.screenshots_dir.name}/{filename}"
        }
        self.detection_events.append(event)

        return event['screenshot']

    def get_face_bounding_box(self, landmarks):
        """Calculate face bounding box from landmarks"""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = (max_x - min_x) * self.frame_width
        height = (max_y - min_y) * self.frame_height
        area = width * height

        return {
            'width': width,
            'height': height,
            'area': area,
            'center_x': (min_x + max_x) / 2,
            'center_y': (min_y + max_y) / 2
        }

    def is_face_valid(self, face_bbox):
        """Check if face detection is reliable (minimum size)"""
        return face_bbox['height'] >= self.MIN_FACE_HEIGHT_PX

    def calculate_ear(self, eye_points):
        """Calculate Eye Aspect Ratio for blink detection"""
        # Vertical distances
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        # Horizontal distance
        C = distance.euclidean(eye_points[0], eye_points[3])

        if C == 0:
            return 0.3  # Return normal value if can't calculate

        ear = (A + B) / (2.0 * C)
        return ear

    def collect_calibration_data(self, landmarks, face_bbox):
        """Collect data during calibration phase"""
        # Face area for lean baseline
        self.face_area_baseline_window.add(face_bbox['area'])

        # EAR for blink baseline
        left_eye = [[landmarks[33].x, landmarks[33].y],
                    [landmarks[160].x, landmarks[160].y],
                    [landmarks[158].x, landmarks[158].y],
                    [landmarks[133].x, landmarks[133].y],
                    [landmarks[153].x, landmarks[153].y],
                    [landmarks[144].x, landmarks[144].y]]
        right_eye = [[landmarks[362].x, landmarks[362].y],
                     [landmarks[385].x, landmarks[385].y],
                     [landmarks[387].x, landmarks[387].y],
                     [landmarks[263].x, landmarks[263].y],
                     [landmarks[373].x, landmarks[373].y],
                     [landmarks[380].x, landmarks[380].y]]

        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        ear = (left_ear + right_ear) / 2.0
        self.ear_baseline_window.add(ear)

        # Smile AU baselines
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]
        nose_tip = landmarks[1]

        # Mouth aspect ratio baseline
        mouth_width = distance.euclidean(
            [left_mouth.x, left_mouth.y],
            [right_mouth.x, right_mouth.y]
        )
        mouth_height = distance.euclidean(
            [upper_lip.x, upper_lip.y],
            [lower_lip.x, lower_lip.y]
        )
        smile_ratio = mouth_width / (mouth_height + 1e-6)
        self.smile_baseline_window.add(smile_ratio)

        # AU12 baseline: lip corner position relative to nose
        lip_corner_avg_y = (left_mouth.y + right_mouth.y) / 2
        lip_corner_relative = nose_tip.y - lip_corner_avg_y  # Higher = raised corners
        if self.baseline_lip_corner_y is None:
            self.baseline_lip_corner_y = lip_corner_relative
        else:
            self.baseline_lip_corner_y = 0.9 * self.baseline_lip_corner_y + 0.1 * lip_corner_relative

        # AU6 baseline: eye-cheek distance (lower eyelid to cheek)
        # Left eye lower lid and left cheek
        left_lower_lid = landmarks[145]
        left_cheek = landmarks[117]
        right_lower_lid = landmarks[374]
        right_cheek = landmarks[346]

        eye_cheek_dist = (
            distance.euclidean([left_lower_lid.x, left_lower_lid.y], [left_cheek.x, left_cheek.y]) +
            distance.euclidean([right_lower_lid.x, right_lower_lid.y], [right_cheek.x, right_cheek.y])
        ) / 2
        if self.baseline_eye_cheek_dist is None:
            self.baseline_eye_cheek_dist = eye_cheek_dist
        else:
            self.baseline_eye_cheek_dist = 0.9 * self.baseline_eye_cheek_dist + 0.1 * eye_cheek_dist

        # Head position for gesture baseline
        nose = landmarks[1]
        if self.previous_nose_y is not None:
            vertical_movement = abs(nose.y - self.previous_nose_y)
            horizontal_movement = abs(nose.x - self.previous_nose_x)
            self.vertical_movement_baseline.add(vertical_movement)
            self.horizontal_movement_baseline.add(horizontal_movement)

        self.previous_nose_y = nose.y
        self.previous_nose_x = nose.x

        # Tension baseline
        left_brow = landmarks[70]
        right_brow = landmarks[300]
        left_eye_pt = landmarks[33]
        right_eye_pt = landmarks[263]

        brow_eye_dist = (
            distance.euclidean([left_brow.x, left_brow.y], [left_eye_pt.x, left_eye_pt.y]) +
            distance.euclidean([right_brow.x, right_brow.y], [right_eye_pt.x, right_eye_pt.y])
        ) / 2
        self.tension_baseline_window.add(brow_eye_dist)

    def finalize_calibration(self):
        """Set baselines after calibration phase"""
        self.baseline_face_area = self.face_area_baseline_window.get_median()
        self.baseline_ear = self.ear_baseline_window.get_median()
        self.baseline_smile_ratio = self.smile_baseline_window.get_median()
        self.baseline_tension = self.tension_baseline_window.get_median()

        # Calculate baseline standard deviations for gesture detection
        self.baseline_vertical_std = self.vertical_movement_baseline.get_std() or 0.005
        self.baseline_horizontal_std = self.horizontal_movement_baseline.get_std() or 0.005

        # Ensure minimum thresholds
        self.baseline_vertical_std = max(self.baseline_vertical_std, 0.003)
        self.baseline_horizontal_std = max(self.baseline_horizontal_std, 0.003)

        # Store calibration info in results
        self.results['calibration_info'] = {
            'baseline_face_area': round(self.baseline_face_area, 2) if self.baseline_face_area else 0,
            'baseline_ear': round(self.baseline_ear, 3) if self.baseline_ear else 0,
            'baseline_smile_ratio': round(self.baseline_smile_ratio, 2) if self.baseline_smile_ratio else 0,
            'calibration_frames': self.processed_frame_count,
            'calibration_quality': self._assess_calibration_quality()
        }

        self.calibration_complete = True
        print(f"\n✅ Calibration complete!")
        print(f"   Baseline face area: {self.baseline_face_area:.0f} px²" if self.baseline_face_area else "   Baseline face area: N/A (no face detected)")
        print(f"   Baseline EAR: {self.baseline_ear:.3f}" if self.baseline_ear else "   Baseline EAR: N/A")
        print(f"   Baseline smile ratio: {self.baseline_smile_ratio:.2f}" if self.baseline_smile_ratio else "   Baseline smile ratio: N/A")
        print(f"   Vertical movement std: {self.baseline_vertical_std:.4f}" if self.baseline_vertical_std else "   Vertical movement std: N/A")
        print(f"   Horizontal movement std: {self.baseline_horizontal_std:.4f}" if self.baseline_horizontal_std else "   Horizontal movement std: N/A")

    def _assess_calibration_quality(self):
        """Assess the quality of calibration data"""
        if self.baseline_face_area is None or self.baseline_face_area < 1000:
            return 'poor'
        if self.processed_frame_count < self.calibration_frames_needed * 0.5:
            return 'fair'
        if self.face_area_baseline_window.get_std() / self.baseline_face_area > 0.3:
            return 'fair'  # High variance during calibration
        return 'good'

    def detect_blink(self, landmarks):
        """
        Robust blink detection using Eye Aspect Ratio with:
        - EWMA smoothing
        - Consecutive frame validation
        - Adaptive threshold based on baseline
        """
        # Calculate EAR
        left_eye = [[landmarks[33].x, landmarks[33].y],
                    [landmarks[160].x, landmarks[160].y],
                    [landmarks[158].x, landmarks[158].y],
                    [landmarks[133].x, landmarks[133].y],
                    [landmarks[153].x, landmarks[153].y],
                    [landmarks[144].x, landmarks[144].y]]
        right_eye = [[landmarks[362].x, landmarks[362].y],
                     [landmarks[385].x, landmarks[385].y],
                     [landmarks[387].x, landmarks[387].y],
                     [landmarks[263].x, landmarks[263].y],
                     [landmarks[373].x, landmarks[373].y],
                     [landmarks[380].x, landmarks[380].y]]

        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        raw_ear = (left_ear + right_ear) / 2.0

        # Smooth the EAR value
        smoothed_ear = self.ear_smoother.update(raw_ear)

        # Calculate adaptive threshold (percentage of baseline)
        # Use 85% of baseline - more sensitive to catch real blinks
        if self.baseline_ear:
            adaptive_threshold = self.baseline_ear * 0.85  # 85% of baseline
            threshold = max(adaptive_threshold, 0.18)  # Don't go below 0.18
        else:
            threshold = self.EAR_BLINK_THRESHOLD

        # Blink detection with consecutive frame validation
        if smoothed_ear < threshold:
            self.blink_frame_counter += 1
        else:
            # Check if we had a valid blink (enough consecutive low-EAR frames)
            if self.blink_frame_counter >= self.BLINK_CONSECUTIVE_FRAMES and not self.in_blink:
                timestamp = self.frame_count / self.fps
                self.results['visual_metrics']['blinks']['total'] += 1
                self.results['visual_metrics']['blinks']['timestamps'].append(round(timestamp, 2))
                self.in_blink = True
                # Capture screenshot
                self.capture_screenshot('blink', timestamp, f"EAR: {smoothed_ear:.3f}")
                print(f"👁️ Blink detected at {timestamp:.2f}s (EAR: {smoothed_ear:.3f})")

            self.blink_frame_counter = 0
            self.in_blink = False

    def detect_smile(self, landmarks):
        """
        Smile detection using mouth aspect ratio with:
        - Baseline comparison (percentage change from neutral)
        - EWMA smoothing
        - Hysteresis thresholds
        - Temporal debounce (minimum duration)

        Simpler but more reliable than full AU detection for video calls.
        """
        # Get landmark points
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        upper_lip = landmarks[13]
        lower_lip = landmarks[14]

        # Calculate mouth aspect ratio (width / height)
        mouth_width = distance.euclidean(
            [left_mouth.x, left_mouth.y],
            [right_mouth.x, right_mouth.y]
        )
        mouth_height = distance.euclidean(
            [upper_lip.x, upper_lip.y],
            [lower_lip.x, lower_lip.y]
        )
        mouth_ratio = mouth_width / (mouth_height + 1e-6)

        # Also check lip corner raise (simple AU12 proxy)
        # When smiling, lip corners move up (lower y value)
        nose_tip = landmarks[1]
        lip_corner_avg_y = (left_mouth.y + right_mouth.y) / 2
        corner_raise = nose_tip.y - lip_corner_avg_y  # Higher value = more raised

        # Get baseline - must be in reasonable range (1.5 to 6.0 for normal faces)
        baseline_ratio = self.smile_baseline_window.get_median()

        if baseline_ratio is None or baseline_ratio < 1.5 or baseline_ratio > 6.0:
            baseline_ratio = 2.8  # Default neutral ratio

        # Calculate smile score based on ratio increase from baseline
        ratio_increase = (mouth_ratio - baseline_ratio) / baseline_ratio

        # Also factor in corner raise if we have baseline
        corner_score = 0
        if self.baseline_lip_corner_y and self.baseline_lip_corner_y > 0:
            corner_increase = (corner_raise - self.baseline_lip_corner_y) / abs(self.baseline_lip_corner_y)
            corner_score = max(0, corner_increase * 2)

        # Combined score (70% ratio, 30% corner raise)
        raw_score = (ratio_increase * 0.7 + corner_score * 0.3)

        # Map to 0-1 range: 15% increase in ratio = smile threshold
        # This is more sensitive than before
        smile_confidence = min(1.0, max(0.0, raw_score / 0.15))

        # Smooth the confidence
        smoothed_confidence = self.smile_confidence_smoother.update(smile_confidence)

        timestamp = self.frame_count / self.fps

        # Hysteresis thresholds (lowered for better detection)
        SMILE_ON_THRESHOLD = 0.55   # Enter smile state
        SMILE_OFF_THRESHOLD = 0.35  # Exit smile state

        is_above_threshold = smoothed_confidence >= SMILE_ON_THRESHOLD
        is_below_exit = smoothed_confidence < SMILE_OFF_THRESHOLD

        if is_above_threshold:
            self.smile_frame_counter += 1
        elif is_below_exit:
            self.smile_frame_counter = 0

        # Check if smile should be confirmed (meets duration requirement - 2 frames)
        smile_confirmed = self.smile_frame_counter >= 2

        # State machine with debounce
        if smile_confirmed and not self.current_smile_state:
            # Smile started
            self.current_smile_state = True
            self.results['visual_metrics']['smiles']['total'] += 1
            self.results['visual_metrics']['smiles']['timestamps'].append(round(timestamp, 2))
            self.smile_start_time = timestamp
            # Capture screenshot
            self.capture_screenshot('smile', timestamp, f"Confidence: {smoothed_confidence:.0%}")
            print(f"😊 Smile detected at {timestamp:.2f}s (conf: {smoothed_confidence:.2f}, ratio: {mouth_ratio:.2f}, baseline: {baseline_ratio:.2f})")

        elif is_below_exit and self.current_smile_state:
            # Smile ended
            self.current_smile_state = False
            if self.smile_start_time:
                duration = timestamp - self.smile_start_time
                self.total_smile_duration += duration
                print(f"   Smile ended at {timestamp:.2f}s (duration: {duration:.2f}s)")
            self.smile_frame_counter = 0

        self.previous_smile_state = self.current_smile_state

    def detect_lean(self, landmarks, face_bbox):
        """
        Robust lean detection with:
        - Face area comparison to baseline (not absolute values)
        - EWMA smoothing
        - Hysteresis to prevent flickering
        - Rolling window comparison (vs 1 second ago)
        """
        if self.baseline_face_area is None or self.baseline_face_area == 0:
            return

        # Smooth current face area
        smoothed_area = self.face_area_smoother.update(face_bbox['area'])
        self.face_area_window.add(smoothed_area)

        # Calculate percentage change from baseline
        percent_change = (smoothed_area - self.baseline_face_area) / self.baseline_face_area

        # Also compare to 1 second ago for trend detection
        area_1sec_ago = self.face_area_window.get_value_n_ago(self.one_second_frames)
        if area_1sec_ago:
            recent_trend = (smoothed_area - area_1sec_ago) / area_1sec_ago
        else:
            recent_trend = 0

        # Determine lean state using hysteresis
        is_forward = self.lean_forward_state.update(percent_change)
        is_backward = self.lean_backward_state.update(abs(percent_change) if percent_change < 0 else 0)

        # Determine final lean state
        if is_forward and percent_change > 0:
            current_lean = 'forward'
            self.lean_forward_frames += 1
        elif is_backward and percent_change < 0:
            current_lean = 'backward'
            self.lean_backward_frames += 1
        else:
            current_lean = 'neutral'
            self.lean_neutral_frames += 1

        timestamp = self.frame_count / self.fps

        # Track lean change events
        if current_lean != self.previous_lean_state:
            if current_lean == 'forward':
                self.results['visual_metrics']['lean']['forward_events'] += 1
                # Capture screenshot
                self.capture_screenshot('lean_forward', timestamp, f"Area change: +{percent_change*100:.1f}%")
                print(f"⬆️ Lean FORWARD at {timestamp:.2f}s (area: +{percent_change*100:.1f}%)")
            elif current_lean == 'backward':
                self.results['visual_metrics']['lean']['backward_events'] += 1
                # Capture screenshot
                self.capture_screenshot('lean_backward', timestamp, f"Area change: {percent_change*100:.1f}%")
                print(f"⬇️ Lean BACKWARD at {timestamp:.2f}s (area: {percent_change*100:.1f}%)")

            # Add to timeline only on state changes
            self.results['visual_metrics']['lean']['timeline'].append({
                'timestamp': round(timestamp, 2),
                'direction': current_lean,
                'percent_change': round(percent_change * 100, 1)
            })

        self.previous_lean_state = current_lean

    def detect_head_gestures(self, landmarks):
        """
        Robust head gesture detection with:
        - EWMA smoothing
        - Adaptive thresholds based on baseline variance
        - Rolling window for trend detection
        """
        nose = landmarks[1]

        # Smooth nose position
        smoothed_y = self.nose_y_smoother.update(nose.y)
        smoothed_x = self.nose_x_smoother.update(nose.x)

        self.nose_y_window.add(smoothed_y)
        self.nose_x_window.add(smoothed_x)

        if self.baseline_vertical_std is None:
            return

        # Get position from several frames ago for gesture detection
        y_prev = self.nose_y_window.get_value_n_ago(3)
        x_prev = self.nose_x_window.get_value_n_ago(3)

        if y_prev is None or x_prev is None:
            return

        # Calculate movement
        vertical_movement = abs(smoothed_y - y_prev)
        horizontal_movement = abs(smoothed_x - x_prev)

        # Adaptive thresholds based on baseline
        nod_threshold = self.baseline_vertical_std * self.NOD_THRESHOLD_MULTIPLIER
        shake_threshold = self.baseline_horizontal_std * self.SHAKE_THRESHOLD_MULTIPLIER

        # Ensure minimum thresholds
        nod_threshold = max(nod_threshold, 0.015)
        shake_threshold = max(shake_threshold, 0.02)

        timestamp = self.frame_count / self.fps

        # Nod detection (significant vertical movement, minimal horizontal)
        if vertical_movement > nod_threshold and horizontal_movement < shake_threshold * 0.5:
            self.results['visual_metrics']['head_gestures']['nods'] += 1
            self.results['visual_metrics']['head_gestures']['timestamps'].append({
                'time': round(timestamp, 2),
                'type': 'nod'
            })
            # Capture screenshot
            self.capture_screenshot('nod', timestamp, f"Movement: {vertical_movement:.4f}")
            print(f"👍 Nod detected at {timestamp:.2f}s (movement: {vertical_movement:.4f})")

        # Shake detection (significant horizontal movement, minimal vertical)
        elif horizontal_movement > shake_threshold and vertical_movement < nod_threshold * 0.5:
            self.results['visual_metrics']['head_gestures']['shakes'] += 1
            self.results['visual_metrics']['head_gestures']['timestamps'].append({
                'time': round(timestamp, 2),
                'type': 'shake'
            })
            # Capture screenshot
            self.capture_screenshot('shake', timestamp, f"Movement: {horizontal_movement:.4f}")
            print(f"👎 Shake detected at {timestamp:.2f}s (movement: {horizontal_movement:.4f})")

    def detect_gaze(self, landmarks):
        """
        Robust gaze detection with:
        - EWMA smoothing
        - Rolling window stability tracking
        """
        # Get iris position (if available with refined landmarks)
        try:
            left_iris = landmarks[468]
        except (IndexError, AttributeError):
            left_iris = landmarks[133]

        left_outer = landmarks[33]
        left_inner = landmarks[133]

        # Calculate gaze ratio
        eye_width = left_inner.x - left_outer.x
        if abs(eye_width) < 1e-6:
            return

        raw_ratio = (left_iris.x - left_outer.x) / eye_width

        # Smooth the ratio
        smoothed_ratio = self.gaze_smoother.update(raw_ratio)

        # Check if gaze is stable (looking at screen)
        is_stable = self.GAZE_STABLE_MIN < smoothed_ratio < self.GAZE_STABLE_MAX

        self.gaze_total_frames += 1
        if is_stable:
            self.gaze_stable_frames += 1
        else:
            timestamp = self.frame_count / self.fps
            self.results['visual_metrics']['gaze']['wandering_events'].append(round(timestamp, 2))

        # Track stability in rolling window
        self.gaze_stability_window.add(1 if is_stable else 0)

    def calculate_tension(self, landmarks):
        """
        Robust tension detection with:
        - EWMA smoothing
        - Baseline normalization
        - Percentage-based scoring
        """
        left_brow = landmarks[70]
        right_brow = landmarks[300]
        left_eye = landmarks[33]
        right_eye = landmarks[263]

        brow_eye_dist = (
            distance.euclidean([left_brow.x, left_brow.y], [left_eye.x, left_eye.y]) +
            distance.euclidean([right_brow.x, right_brow.y], [right_eye.x, right_eye.y])
        ) / 2

        # Smooth the value
        smoothed_dist = self.tension_smoother.update(brow_eye_dist)

        # Calculate tension relative to baseline
        if self.baseline_tension and self.baseline_tension > 0:
            # Higher distance = raised eyebrows = less tension
            # Lower distance = furrowed brows = more tension
            relative_tension = (self.baseline_tension - smoothed_dist) / self.baseline_tension
            tension_level = 50 + (relative_tension * 100)  # Center at 50
            tension_level = max(0, min(100, tension_level))
        else:
            tension_level = 50  # Default to neutral

        timestamp = self.frame_count / self.fps
        self.results['visual_metrics']['tension']['timeline'].append({
            'timestamp': round(timestamp, 2),
            'level': round(tension_level, 1)
        })

    def analyze_video(self):
        """
        Main video analysis loop with:
        - Calibration phase (first few seconds)
        - Face validation (reject small/unreliable detections)
        - All robust detection methods
        - Optimized for long videos with dynamic frame skipping
        - Focus on largest face (main speaker in video calls)
        """
        print(f"\n🎬 Analyzing video: {self.duration/60:.1f} min ({self.total_frames} frames)")
        print(f"   Resolution: {self.frame_width}x{self.frame_height}, FPS: {self.fps:.1f}")
        print(f"   Frame skip: {self.frame_skip} (processing ~{self.total_frames//self.frame_skip} frames)")
        print(f"   Calibration period: {self.CALIBRATION_SECONDS} seconds")
        print("\n📊 Starting calibration phase...")

        frames_without_face = 0
        max_frames_without_face = int(self.fps * 2)  # 2 seconds without face = warning
        last_progress_time = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1

            # Dynamic frame skip for performance (varies by video length)
            if self.frame_count % self.frame_skip != 0:
                continue

            self.processed_frame_count += 1

            # Store current frame for screenshot capture (don't copy on every frame for memory)
            # Only copy when we might need screenshot
            if self.screenshot_count < self.MAX_SCREENSHOTS:
                self.current_frame = frame.copy()

            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face landmarks
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                # If multiple faces detected, select the LARGEST one (main speaker)
                if len(results.multi_face_landmarks) > 1:
                    largest_face = None
                    largest_area = 0
                    for face_landmarks in results.multi_face_landmarks:
                        bbox = self.get_face_bounding_box(face_landmarks.landmark)
                        if bbox['area'] > largest_area:
                            largest_area = bbox['area']
                            largest_face = face_landmarks.landmark
                    landmarks = largest_face
                else:
                    landmarks = results.multi_face_landmarks[0].landmark

                frames_without_face = 0

                # Calculate face bounding box
                face_bbox = self.get_face_bounding_box(landmarks)

                # Validate face size
                if not self.is_face_valid(face_bbox):
                    continue  # Skip unreliable detection

                # CALIBRATION PHASE
                if not self.calibration_complete:
                    self.collect_calibration_data(landmarks, face_bbox)

                    if self.processed_frame_count >= self.calibration_frames_needed:
                        self.finalize_calibration()
                        print("\n📈 Starting engagement analysis...")
                    continue

                # ANALYSIS PHASE (after calibration)
                self.detect_blink(landmarks)
                self.detect_smile(landmarks)
                self.detect_lean(landmarks, face_bbox)
                self.detect_head_gestures(landmarks)
                self.detect_gaze(landmarks)
                self.calculate_tension(landmarks)

            else:
                frames_without_face += 1
                if frames_without_face == max_frames_without_face:
                    timestamp = self.frame_count / self.fps
                    print(f"⚠️ No face detected for 2+ seconds at {timestamp:.2f}s")

            # Progress indicator - more frequent for long videos
            current_time = self.frame_count / self.fps
            if self.processed_frame_count % self.PROGRESS_UPDATE_INTERVAL == 0 or current_time - last_progress_time >= 30:
                progress = (self.frame_count / self.total_frames) * 100
                status = "Calibrating..." if not self.calibration_complete else f"Analyzing... ({current_time/60:.1f} min)"

                # Only print every 5% or 30 seconds for long videos
                if self.duration > 600:  # > 10 min
                    if int(progress) % 5 == 0 or current_time - last_progress_time >= 30:
                        print(f"   Progress: {progress:.1f}% ({current_time/60:.1f}/{self.duration/60:.1f} min)")
                        last_progress_time = current_time
                else:
                    print(f"   Progress: {progress:.1f}%")

                # Update global progress
                if self.task_id:
                    analysis_progress[self.task_id] = {
                        'progress': round(progress * 0.85, 1),  # Reserve 15% for audio analysis
                        'status': status
                    }

        self.cap.release()

        # Free memory
        self.current_frame = None

        # Handle case where calibration never completed
        if not self.calibration_complete:
            print("\n⚠️ Warning: Calibration incomplete - using default baselines")
            self.finalize_calibration()

        self.finalize_results()

        # Add detection events with screenshots to results
        self.results['detection_events'] = sorted(self.detection_events, key=lambda x: x['timestamp'])

        return self.results

    def finalize_results(self):
        """Calculate final metrics with robust averaging"""
        # Effective analysis duration (excluding calibration)
        analysis_start_time = self.CALIBRATION_SECONDS
        effective_duration = max(0, self.duration - analysis_start_time)

        # Blinks per minute
        if effective_duration > 0:
            self.results['visual_metrics']['blinks']['per_minute'] = round(
                self.results['visual_metrics']['blinks']['total'] / (effective_duration / 60), 1
            )

        # Smile duration
        self.results['visual_metrics']['smiles']['duration_seconds'] = round(self.total_smile_duration, 1)

        # Lean percentages
        total_lean_frames = self.lean_forward_frames + self.lean_backward_frames + self.lean_neutral_frames
        if total_lean_frames > 0:
            self.results['visual_metrics']['lean']['forward_percent'] = round(
                (self.lean_forward_frames / total_lean_frames) * 100, 1
            )
            self.results['visual_metrics']['lean']['backward_percent'] = round(
                (self.lean_backward_frames / total_lean_frames) * 100, 1
            )
            self.results['visual_metrics']['lean']['neutral_percent'] = round(
                (self.lean_neutral_frames / total_lean_frames) * 100, 1
            )

        # Gaze stability percentage
        if self.gaze_total_frames > 0:
            self.results['visual_metrics']['gaze']['stable_percent'] = round(
                (self.gaze_stable_frames / self.gaze_total_frames) * 100, 1
            )
            self.results['visual_metrics']['gaze']['avg_stability'] = round(
                self.gaze_stability_window.get_mean() * 100 if self.gaze_stability_window.get_mean() else 0, 1
            )

        # Tension statistics
        if self.results['visual_metrics']['tension']['timeline']:
            tension_values = [t['level'] for t in self.results['visual_metrics']['tension']['timeline']]
            self.results['visual_metrics']['tension']['avg_level'] = round(np.mean(tension_values), 1)
            self.results['visual_metrics']['tension']['max_level'] = round(max(tension_values), 1)

        # Limit timeline entries to prevent huge JSON
        max_timeline_entries = 100
        if len(self.results['visual_metrics']['lean']['timeline']) > max_timeline_entries:
            # Keep only significant events (state changes already filtered)
            self.results['visual_metrics']['lean']['timeline'] = \
                self.results['visual_metrics']['lean']['timeline'][:max_timeline_entries]

        if len(self.results['visual_metrics']['tension']['timeline']) > max_timeline_entries:
            # Sample tension timeline
            step = len(self.results['visual_metrics']['tension']['timeline']) // max_timeline_entries
            self.results['visual_metrics']['tension']['timeline'] = \
                self.results['visual_metrics']['tension']['timeline'][::step][:max_timeline_entries]

        # Limit wandering events
        if len(self.results['visual_metrics']['gaze']['wandering_events']) > 50:
            self.results['visual_metrics']['gaze']['wandering_events'] = \
                self.results['visual_metrics']['gaze']['wandering_events'][:50]

        # Calculate overall attention score
        self.calculate_attention_score()

        # Print summary
        print("\n" + "="*50)
        print("✅ ANALYSIS COMPLETE")
        print("="*50)
        print(f"\n📊 ENGAGEMENT METRICS:")
        print(f"   Overall Score: {self.results['overall_score']}/100")
        print(f"\n   👁️ Blinks: {self.results['visual_metrics']['blinks']['total']} "
              f"({self.results['visual_metrics']['blinks']['per_minute']}/min)")
        print(f"   😊 Smiles: {self.results['visual_metrics']['smiles']['total']} "
              f"({self.results['visual_metrics']['smiles']['duration_seconds']}s total)")
        print(f"   👍 Nods: {self.results['visual_metrics']['head_gestures']['nods']}")
        print(f"   👎 Shakes: {self.results['visual_metrics']['head_gestures']['shakes']}")
        print(f"   👀 Gaze Stability: {self.results['visual_metrics']['gaze']['stable_percent']}%")
        print(f"   ⬆️ Forward Lean: {self.results['visual_metrics']['lean']['forward_percent']}%")
        print(f"   ⬇️ Backward Lean: {self.results['visual_metrics']['lean']['backward_percent']}%")
        print(f"   😐 Tension: {self.results['visual_metrics']['tension']['avg_level']}/100 avg")
        print("="*50)

    def calculate_attention_score(self):
        """
        Calculate overall attention score (0-100) using weighted metrics.

        Weights:
        - Gaze stability: 35% (most important for video calls)
        - Forward lean: 20% (shows interest)
        - Blink rate: 15% (normal = engaged)
        - Positive signals: 15% (smiles + nods)
        - Low tension: 15% (relaxed = comfortable)
        """
        score = 0

        # 1. Gaze stability contribution (35%)
        gaze_score = self.results['visual_metrics']['gaze']['stable_percent']
        score += gaze_score * 0.35

        # 2. Forward lean contribution (20%)
        forward_percent = self.results['visual_metrics']['lean']['forward_percent']
        # More forward lean = more engaged (cap at 100% contribution)
        lean_contribution = min(100, forward_percent * 2)  # 50% forward = full score
        score += lean_contribution * 0.20

        # 3. Blink rate contribution (15%) - normal is 15-20/min
        blink_rate = self.results['visual_metrics']['blinks']['per_minute']
        if 12 <= blink_rate <= 25:
            blink_score = 100  # Optimal range
        elif 8 <= blink_rate <= 30:
            blink_score = 75  # Acceptable range
        elif blink_rate > 30:
            blink_score = 50  # High blink rate (might indicate discomfort)
        else:
            blink_score = 60  # Low blink rate
        score += blink_score * 0.15

        # 4. Positive engagement signals (15%) - smiles and nods
        smiles = self.results['visual_metrics']['smiles']['total']
        nods = self.results['visual_metrics']['head_gestures']['nods']
        # Normalize: expect ~1-2 smiles and ~5-10 nods per minute
        effective_duration = max(1, self.duration - self.CALIBRATION_SECONDS) / 60  # in minutes
        expected_smiles = effective_duration * 2
        expected_nods = effective_duration * 8

        smile_ratio = min(1, smiles / max(1, expected_smiles))
        nod_ratio = min(1, nods / max(1, expected_nods))
        engagement_score = ((smile_ratio + nod_ratio) / 2) * 100
        score += engagement_score * 0.15

        # 5. Low tension contribution (15%) - lower tension = more comfortable
        tension_avg = self.results['visual_metrics']['tension']['avg_level']
        # Tension of 50 = neutral, lower = relaxed, higher = stressed
        if tension_avg <= 50:
            tension_score = 100  # Relaxed or neutral
        elif tension_avg <= 65:
            tension_score = 80  # Slightly tense
        elif tension_avg <= 80:
            tension_score = 60  # Moderately tense
        else:
            tension_score = 40  # High tension
        score += tension_score * 0.15

        # Round and store
        self.results['visual_metrics']['attention_score'] = round(score, 1)
        self.results['overall_score'] = round(score, 1)


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/progress/<task_id>')
def get_progress(task_id):
    """Get analysis progress for a task"""
    if task_id in analysis_progress:
        return jsonify(analysis_progress[task_id])
    return jsonify({'progress': 0, 'status': 'Starting...'})


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video/audio upload and analysis"""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Get task ID from frontend (if provided) or generate one
    # This ensures frontend and backend use the same ID for progress tracking
    task_id = request.form.get('task_id')
    if not task_id:
        task_id = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Initialize progress BEFORE saving (so frontend can poll during upload)
    analysis_progress[task_id] = {'progress': 0, 'status': 'Uploading file...'}

    # Save uploaded file with the task_id
    filename = f"{task_id}_{file.filename}"
    filepath = UPLOAD_FOLDER / filename
    file.save(filepath)

    print(f"File uploaded: {filepath} (task_id: {task_id})")

    # Detect if file is audio-only or has video stream
    is_audio_only = False
    file_ext = filepath.suffix.lower()

    # Check file extension first
    audio_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac', '.wma']
    if file_ext in audio_extensions:
        is_audio_only = True
        print(f"📢 Audio-only file detected: {file_ext}")
    else:
        # For video files (mp4, webm, etc.), check if they have a valid video stream
        try:
            cap = cv2.VideoCapture(str(filepath))
            if cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None:
                    is_audio_only = True
                    print(f"📢 Video file has no valid video stream - treating as audio-only")
                else:
                    # Check if video has actual content (not just black frames)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames < 10:
                        is_audio_only = True
                        print(f"📢 Video has only {total_frames} frames - treating as audio-only")
            else:
                is_audio_only = True
                print(f"📢 Cannot open video stream - treating as audio-only")
            cap.release()
        except Exception as e:
            print(f"⚠️ Error checking video stream: {e} - treating as audio-only")
            is_audio_only = True

    # Update progress after upload complete
    analysis_progress[task_id] = {'progress': 2, 'status': 'Starting analysis...'}

    try:
        results = {
            'success': True,
            'is_audio_only': is_audio_only,
            'filename': file.filename,
            'task_id': task_id
        }

        # Only run video analysis if file has video stream
        if not is_audio_only:
            # Analyze video with task_id for progress tracking
            analyzer = VideoAnalyzer(filepath, task_id=task_id)
            video_results = analyzer.analyze_video()
            results.update(video_results)
        else:
            print(f"⏭️ Skipping video analysis for audio-only file")
            analysis_progress[task_id] = {'progress': 50, 'status': 'Processing audio...'}

        # Update progress for audio analysis
        analysis_progress[task_id] = {'progress': 90, 'status': 'Analyzing audio...'}

        # Run audio analysis
        if AUDIO_AVAILABLE:
            audio_analyzer = AudioAnalyzer(filepath, task_id=task_id)
            audio_results = audio_analyzer.analyze()
            results['audio_metrics'] = audio_results

            # Run speaker diarization if HF_TOKEN is set
            if DIARIZATION_AVAILABLE and os.environ.get('HF_TOKEN'):
                analysis_progress[task_id] = {'progress': 95, 'status': 'Running speaker diarization...'}
                # Use the extracted audio path from AudioAnalyzer or extract again
                audio_path = audio_analyzer.audio_path
                if audio_path and os.path.exists(audio_path):
                    diarizer = SpeakerDiarizer(audio_path, task_id=task_id)
                    # min_speakers=2 helps with 2-person conversations, max_speakers=10 for flexibility
                    diarization_results = diarizer.analyze(min_speakers=2, max_speakers=10)
                    results['speaker_diarization'] = diarization_results
                else:
                    # Need to extract audio again for diarization
                    import tempfile
                    fd, temp_audio = tempfile.mkstemp(suffix='.wav')
                    os.close(fd)
                    # Extract audio
                    cmd = ['ffmpeg', '-y', '-i', str(filepath), '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', temp_audio]
                    subprocess.run(cmd, capture_output=True, timeout=120)
                    if os.path.exists(temp_audio):
                        diarizer = SpeakerDiarizer(temp_audio, task_id=task_id)
                        diarization_results = diarizer.analyze(min_speakers=2, max_speakers=10)
                        results['speaker_diarization'] = diarization_results
                        os.remove(temp_audio)
                    else:
                        results['speaker_diarization'] = {'success': False, 'error': 'Could not extract audio'}
            else:
                results['speaker_diarization'] = {
                    'success': False,
                    'error': 'HF_TOKEN not set. Set environment variable to enable speaker diarization.'
                }

            # Run speech transcription with Whisper
            print(f"🔍 DEBUG: WHISPER_AVAILABLE = {WHISPER_AVAILABLE}")
            if WHISPER_AVAILABLE:
                analysis_progress[task_id] = {'progress': 97, 'status': 'Transcribing speech...'}
                audio_path = audio_analyzer.audio_path
                print(f"🔍 DEBUG: audio_path = {audio_path}")
                print(f"🔍 DEBUG: audio_path exists = {os.path.exists(audio_path) if audio_path else 'None'}")
                if audio_path and os.path.exists(audio_path):
                    # Use "base" model - fast + decent accuracy
                    # tiny=39MB (fastest), base=74MB (fast+good), small=244MB, medium=769MB (slow), large=1550MB (very slow)
                    # For 50min video: tiny ~1min, base ~2-3min, medium ~15-20min, large ~1hr+
                    transcriber = SpeechTranscriber(audio_path, task_id=task_id, model_size="small")
                    transcription_results = transcriber.analyze()

                    # Align transcription with diarization if both are available
                    if transcription_results.get('success') and results.get('speaker_diarization', {}).get('success'):
                        aligned_segments = transcriber.align_with_diarization(results['speaker_diarization'])
                        transcription_results['aligned_segments'] = aligned_segments

                    results['transcription'] = transcription_results
                else:
                    results['transcription'] = {'success': False, 'error': 'Audio path not available'}
            else:
                results['transcription'] = {'success': False, 'error': 'Whisper library not available'}

            # Cleanup temp audio file after all audio processing is done
            if audio_analyzer.audio_path and os.path.exists(audio_analyzer.audio_path):
                try:
                    os.remove(audio_analyzer.audio_path)
                    print(f"🗑️ Cleaned up temp audio file")
                except:
                    pass
        else:
            results['audio_metrics'] = {'audio_available': False}
            results['speaker_diarization'] = {'success': False, 'error': 'Audio libraries not available'}
            results['transcription'] = {'success': False, 'error': 'Audio libraries not available'}

        # Mark complete
        analysis_progress[task_id] = {'progress': 100, 'status': 'Complete!'}

        # Save results
        results_file = UPLOAD_FOLDER / f"{task_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Clean up progress after a delay (keep for 60 seconds)
        # In production, use a proper cleanup mechanism

        return jsonify({
            'success': True,
            'results': results,
            'video_id': task_id
        })

    except Exception as e:
        print(f"Error analyzing video: {e}")
        import traceback
        traceback.print_exc()
        analysis_progress[task_id] = {'progress': 0, 'status': f'Error: {str(e)}'}
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Arali.ai Video Analyzer Starting...")
    print("📊 Advanced engagement analysis with OpenCV + MediaPipe")
    print("🌐 Open http://localhost:5001 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5001)
