import whisper
import os
import pandas as pd
from tqdm import tqdm

# ========== CONFIG ==========
AUDIO_FOLDER = "tts_dataset\wavs"
OUTPUT_CSV = "transcripts.csv"
OUTPUT_TXT = "nepali_transcripts.txt"
MODEL_SIZE = "medium"   # tiny, base, small, medium, large
# ============================

print("Loading Whisper model...")
model = whisper.load_model(MODEL_SIZE)

results = []

print("Transcribing Nepali audio files...\n")

for file in tqdm(os.listdir(AUDIO_FOLDER)):
    if file.lower().endswith(".wav"):
        path = os.path.join(AUDIO_FOLDER, file)

        try:
            result = model.transcribe(
                path,
                language="ne",        # Force Nepali
                fp16=False
            )

            text = result["text"].strip()

            print(f"\n{file} → {text}")

            results.append({
                "filename": file,
                "transcript_nepali": text
            })

        except Exception as e:
            print(f"❌ Error in {file}: {e}")

# ========== Save to CSV ==========
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# ========== Save to TXT ==========
with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    for r in results:
        f.write(f"{r['filename']}:\n{r['transcript_nepali']}\n\n")

print("\n==============================")
print("All files transcribed!")
print("Saved as:")
print(" → nepali_transcripts.csv")
print(" → nepali_transcripts.txt")
print("==============================")
