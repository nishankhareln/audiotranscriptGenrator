import os
import torch
import subprocess
import shutil
import multiprocessing
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import whisper
from tqdm import tqdm

def process_chunk_on_gpu(task):
    """Function to be run in parallel across GPUs"""
    chunk_path, output_dir, gpu_id = task
    chunk_stem = Path(chunk_path).stem
    
    # Using 'htdemucs_ft' for better music removal and 'shifts' for accuracy
    cmd = [
        "demucs", 
        "--two-stems", "vocals", 
        "-n", "htdemucs_ft", 
        "--device", f"cuda:{gpu_id}", 
        "--segment", "10",
        "--shifts", "5",  # Increases quality significantly
        str(chunk_path), 
        "-o", str(output_dir)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return output_dir / "htdemucs_ft" / chunk_stem / "vocals.wav"

class NaturalAIProcessor:
    def __init__(self, input_file, output_dir, min_dur=2, max_dur=10):
        self.input_path = Path(input_file)
        self.output_dir = Path(output_dir)
        self.wav_dir = self.output_dir / "wavs1"
        self.min_dur = min_dur * 1000 
        self.max_dur = max_dur * 1000 
        self.padding = 300 
        
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.wav_dir.mkdir(parents=True, exist_ok=True)
        
        self.num_gpus = torch.cuda.device_count()
        print(f"🚀 Detected {self.num_gpus} GPUs. Using all for parallel processing.")
        
        # Whisper still loads on primary GPU (cuda:0)
        self.transcribe_model = whisper.load_model("medium", device="cuda:0")

    def separate_vocals(self):
        print("🛠 Stage 1: Parallel AI Vocal Isolation (Dual-GPU & High Quality)...")
        temp_chunks_dir = self.output_dir / "temp_chunks"
        temp_out_dir = self.output_dir / "temp_out"
        temp_chunks_dir.mkdir(parents=True, exist_ok=True)

        # Split into smaller 20-minute chunks to distribute work better
        subprocess.run([
            'ffmpeg', '-i', str(self.input_path), 
            '-f', 'segment', '-segment_time', '1200', 
            '-c', 'copy', str(temp_chunks_dir / "chunk_%03d.wav")
        ], check=True, capture_output=True)

        chunks = sorted(list(temp_chunks_dir.glob("*.wav")))
        tasks = []
        
        # Distribute chunks between GPU 0 and GPU 1
        for i, chunk in enumerate(chunks):
            gpu_id = i % self.num_gpus
            tasks.append((chunk, temp_out_dir, gpu_id))

        # Run parallel processing
        with multiprocessing.Pool(processes=self.num_gpus) as pool:
            vocal_paths = list(tqdm(pool.imap(process_chunk_on_gpu, tasks), total=len(tasks), desc="Processing Chunks"))

        print("🔗 Merging high-quality vocal chunks...")
        final_vocal = self.output_dir / "vocals_final.wav"
        list_file = self.output_dir / "concat_list.txt"
        
        with open(list_file, 'w') as f:
            for p in vocal_paths:
                f.write(f"file '{p.absolute()}'\n")

        subprocess.run([
            'ffmpeg', '-f', 'concat', '-safe', '0', 
            '-i', str(list_file), '-c', 'copy', str(final_vocal)
        ], check=True, capture_output=True)

        shutil.rmtree(temp_chunks_dir)
        return final_vocal

    def process(self):
        vocal_file = self.separate_vocals()
        
        print("🛠 Stage 2: Natural Segmentation...")
        audio = AudioSegment.from_wav(vocal_file)
        ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-45) # Lowered thresh for cleaner cuts
        
        results = []
        print(f"🛠 Stage 3: Transcribing {len(ranges)} segments...")
        
        for idx, (start, end) in enumerate(tqdm(ranges)):
            start = max(0, start - self.padding)
            end = min(len(audio), end + self.padding)
            
            duration = end - start
            if duration < self.min_dur: continue

            if duration > self.max_dur:
                end = start + self.max_dur

            seg_name = f"{idx:04d}.wav"
            seg_path = self.wav_dir / seg_name
            
            segment = audio[start:end].fade_in(50).fade_out(50)
            segment.export(seg_path, format="wav")
            
            txt = self.transcribe_model.transcribe(str(seg_path))["text"].strip()
            if len(txt) > 5:
                results.append(f"{seg_name}|{txt}")

        with open(self.output_dir / "metadata1.csv", "w", encoding="utf-8") as f:
            f.write("\n".join(results))
            
        print(f"\n✅ Finished! Multi-GPU processing complete.")

if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    
    FILE = r"D:\audioanalysis\audio_partsaigerace2.wav"
    OUT = r"D:\audioanalysis\output_saigrace1"
    processor = NaturalAIProcessor(FILE, OUT)
    processor.process()