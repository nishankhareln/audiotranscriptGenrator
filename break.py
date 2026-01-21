import subprocess
import os

def split_audio_in_half(input_file):
    # Midpoint for 05:25:06 is 02:42:33
    half_point = "02:42:33"
    
    file_path = os.path.abspath(input_file)
    folder = os.path.dirname(file_path)
    
    part1 = os.path.join(folder, "audio_partsaigrace1.wav")
    part2 = os.path.join(folder, "audio_partsaigerace2.wav")

    print("✂️ Cutting audio into two halves...")

    # Command for the first half
    cmd1 = ['ffmpeg', '-i', file_path, '-t', half_point, '-c', 'copy', part1]
    
    # Command for the second half (starting from the half point)
    cmd2 = ['ffmpeg', '-i', file_path, '-ss', half_point, '-c', 'copy', part2]

    try:
        subprocess.run(cmd1, check=True)
        print(f"✅ Part 1 saved: {part1}")
        
        subprocess.run(cmd2, check=True)
        print(f"✅ Part 2 saved: {part2}")
        
        print("\nSuccess! You now have two smaller files ready for AI processing.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Path to your 5.5 hour file
    INPUT = r"D:\audioanalysis\audio_output.wav"
    split_audio_in_half(INPUT)