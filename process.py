import os
import torch
import subprocess
import shutil
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import whisper
import noisereduce as nr
import librosa
import soundfile as sf
from tqdm import tqdm

class NaturalAIProcessor:
    def __init__(self, input_file, output_dir, min_dur=2, max_dur=10):
        self.input_path = Path(input_file)
        self.output_dir = Path(output_dir)
        self.wav_dir = self.output_dir / "wavs1"
        self.min_dur = min_dur * 1000  # ms
        self.max_dur = max_dur * 1000  # ms
        self.padding = 300  # 300ms buffer to prevent word clipping
        
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f" Using Device: {self.device}")
        # Using 'medium' or 'large' is better for ensuring transcripts don't cut off words
        self.transcribe_model = whisper.load_model("medium", device=self.device)

    def separate_vocals(self):
        print("🛠 Stage 1: AI Vocal Isolation...")
        command = ["demucs", "--two-stems", "vocals", "-n", "htdemucs", str(self.input_path), "-o", str(self.output_dir)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        return self.output_dir / "htdemucs" / self.input_path.stem / "vocals.wav"

    def process(self):
        vocal_file = self.separate_vocals()
        
        print("🛠 Stage 2: Natural Segmentation...")
        audio = AudioSegment.from_wav(vocal_file)
        
        # Detect chunks with a threshold that allows for natural speech breath
        # min_silence_len=500 means the speaker must stop for 0.5s to create a cut
        ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)
        
        results = []
        print(f"🛠 Stage 3: Exporting {len(ranges)} segments...")
        
        for idx, (start, end) in enumerate(tqdm(ranges)):
            # Apply Padding: Start 300ms earlier and end 300ms later
            start = max(0, start - self.padding)
            end = min(len(audio), end + self.padding)
            
            duration = end - start
            if duration < self.min_dur: 
                continue # Skip fragments shorter than 2 seconds

            # Real-world logic: if a segment is too long (e.g. > 10s), 
            # don't just hard cut it. We keep it as is or handle it.
            # For TTS, 2-10 seconds is the "sweet spot".
            if duration > self.max_dur:
                end = start + self.max_dur

            seg_name = f"{idx:04d}.wav"
            seg_path = self.wav_dir / seg_name
            
            # Export with a slight fade to make it sound "real" and avoid pops
            segment = audio[start:end].fade_in(50).fade_out(50)
            segment.export(seg_path, format="wav")
            
            # Transcription
            txt = self.transcribe_model.transcribe(str(seg_path))["text"].strip()
            
            # Final Safety: If Whisper returns an empty or too short string, skip it
            if len(txt) > 5:
                results.append(f"{seg_name}|{txt}")

        # Cleanup
        shutil.rmtree(self.output_dir / "htdemucs")
        
        with open(self.output_dir / "metadata1.csv", "w", encoding="utf-8") as f:
            f.write("\n".join(results))
            
        print(f"\n✅ Finished! Natural segments in: {self.output_dir}")

if __name__ == "__main__":
    FILE = r"D:\audioanalysis\yotubue_voices\oshinsitaula.wav"
    OUT = r"D:\audioanalysis\output_natural"
    processor = NaturalAIProcessor(FILE, OUT)
    processor.process()
