"""
Audio Splitter & Transcriber for TTS Fine-tuning
Splits MP3 files into 7-10 second WAV segments and generates transcripts
"""

import os
import json
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import whisper
import torch
from tqdm import tqdm

class AudioProcessor:
    def __init__(self, input_dir, output_dir, min_duration=7, max_duration=10):
        """
        Initialize the audio processor
        
        Args:
            input_dir: Directory containing MP3 files
            output_dir: Directory to save WAV segments and transcripts
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.min_duration = min_duration * 1000  # Convert to milliseconds
        self.max_duration = max_duration * 1000
        
        # Create output directories
        self.wav_dir = self.output_dir / "wavs"
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Whisper model
        print("Loading Whisper model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model("base", device=device)
        print(f"Using device: {device}")
        
    def split_audio(self, audio_file):
        """
        Split audio file into segments of 7-10 seconds
        """
        print(f"\nProcessing: {audio_file.name}")
        audio = AudioSegment.from_file(str(audio_file))
        
        segments = []
        file_base_name = audio_file.stem
        
        # Detect non-silent chunks
        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=500,  # 500ms of silence
            silence_thresh=-40    # dB threshold
        )
        
        if not nonsilent_ranges:
            # If no silence detected, split by fixed duration
            nonsilent_ranges = [(i, min(i + self.max_duration, len(audio))) 
                               for i in range(0, len(audio), self.max_duration)]
        
        # Merge small chunks into optimal segments
        current_start = nonsilent_ranges[0][0]
        
        for i, (start, end) in enumerate(nonsilent_ranges):
            duration = end - current_start
            
            # If we've reached optimal duration or last chunk
            if duration >= self.min_duration or i == len(nonsilent_ranges) - 1:
                # Cap at max duration
                if duration > self.max_duration:
                    end = current_start + self.max_duration
                
                segments.append((current_start, end))
                
                # Start next segment
                if i < len(nonsilent_ranges) - 1:
                    current_start = nonsilent_ranges[i + 1][0]
        
        # Export segments
        segment_files = []
        for idx, (start, end) in enumerate(segments):
            segment = audio[start:end]
            
            # Convert to mono, 22050 Hz (common for TTS)
            segment = segment.set_channels(1)
            segment = segment.set_frame_rate(22050)
            
            # Generate filename
            filename = f"{file_base_name}_seg{idx:03d}.wav"
            filepath = self.wav_dir / filename
            
            # Export as WAV
            segment.export(str(filepath), format="wav")
            segment_files.append(filepath)
            
        return segment_files
    
    def transcribe_segments(self, segment_files):
        """
        Transcribe audio segments using Whisper
        """
        transcriptions = []
        
        print(f"\nTranscribing {len(segment_files)} segments...")
        for filepath in tqdm(segment_files):
            result = self.model.transcribe(
                str(filepath),
                language="en",  # Change if needed
                fp16=torch.cuda.is_available()
            )
            
            transcriptions.append({
                "audio_file": filepath.name,
                "text": result["text"].strip(),
                "duration": result.get("duration", 0)
            })
        
        return transcriptions
    
    def process_all(self):
        """
        Process all audio files in input directory
        """
        # Support multiple audio formats
        audio_files = []
        for ext in ['*.m4a', '*.mp4a', '*.mp4', '*.mp3', '*.wav', '*.aac']:
            audio_files.extend(list(self.input_dir.glob(ext)))
        
        if not audio_files:
            print(f"No audio files found in {self.input_dir}")
            return
        
        print(f"Found {len(audio_files)} audio files")
        
        all_transcriptions = []
        
        for audio_file in audio_files:
            # Split audio into segments
            segment_files = self.split_audio(audio_file)
            
            # Transcribe segments
            transcriptions = self.transcribe_segments(segment_files)
            all_transcriptions.extend(transcriptions)
        
        # Save transcriptions
        self.save_transcriptions(all_transcriptions)
        
        print(f"\n✓ Processing complete!")
        print(f"  Total segments: {len(all_transcriptions)}")
        print(f"  WAV files: {self.wav_dir}")
        print(f"  Transcripts: {self.output_dir}")
    
    def save_transcriptions(self, transcriptions):
        """
        Save transcriptions in multiple formats for TTS training
        """
        # Format 1: JSON (detailed)
        json_path = self.output_dir / "transcriptions.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(transcriptions, f, indent=2, ensure_ascii=False)
        
        # Format 2: CSV-like format (LJSpeech style)
        metadata_path = self.output_dir / "metadata.csv"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for item in transcriptions:
                # Format: filename|text
                f.write(f"{item['audio_file']}|{item['text']}\n")
        
        # Format 3: Separate text files
        txt_dir = self.output_dir / "transcripts"
        txt_dir.mkdir(exist_ok=True)
        
        for item in transcriptions:
            txt_file = txt_dir / f"{Path(item['audio_file']).stem}.txt"
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(item['text'])
        
        print(f"\nSaved transcriptions:")
        print(f"  - JSON: {json_path}")
        print(f"  - Metadata CSV: {metadata_path}")
        print(f"  - Individual txt: {txt_dir}")


def main():
    """
    Main execution function
    """
    # Configuration
    INPUT_DIR = "./audiofiles"     # Directory with your audio files
    OUTPUT_DIR = "./tts_dataset"   # Output directory
    MIN_DURATION = 7               # Minimum segment duration (seconds)
    MAX_DURATION = 10              # Maximum segment duration (seconds)
    
    # Create processor and run
    processor = AudioProcessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        min_duration=MIN_DURATION,
        max_duration=MAX_DURATION
    )
    
    processor.process_all()


if __name__ == "__main__":
    main()


"""
INSTALLATION INSTRUCTIONS:
--------------------------
pip install pydub whisper torch torchaudio tqdm

# For pydub to work, you also need ffmpeg:
# Ubuntu/Debian: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: Download from https://ffmpeg.org/

USAGE:
------
1. Create a folder called 'input_audio' and put your .m4a files there
2. Run: python audio_processor.py
3. Output will be in 'tts_dataset' folder:
   - tts_dataset/wavs/ : All WAV segments
   - tts_dataset/metadata.csv : Training metadata
   - tts_dataset/transcriptions.json : Detailed info
   - tts_dataset/transcripts/ : Individual text files

OUTPUT FORMAT:
--------------
The metadata.csv follows LJSpeech format:
filename.wav|Transcribed text here

This is compatible with most TTS training frameworks like:
- Coqui TTS
- NVIDIA NeMo
- ESPnet
- Tacotron2
- VITS
"""