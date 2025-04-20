import subprocess
import json
import os

def extract_audio_with_ffmpeg(video_path, audio_path="/home/ubuntu/output.wav"):
    print(f"[INFO] Extracting audio from {video_path} to {audio_path}...")
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
        "-y"
    ]
    subprocess.run(command, check=True)
    return audio_path

def transcribe_with_whisper_cpp(audio_path, model_path="models/ggml-base.en.bin", output_json="whisper_transcription.json"):
    command = [
        "../../whisper.cpp/./build/bin/whisper-cli",
        "-f", audio_path,
        "-m", model_path,
        "-otxt",  # output plain .txt
        "-oj",    # output .json
        "-ml", "100",
        "-of", output_json.replace(".json", "")
    ]

    print(f"[INFO] Running whisper.cpp transcription on {audio_path}...")
    subprocess.run(command, check=True)

    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            data = json.load(f)
        return data
    else:
        raise FileNotFoundError(f"Failed to find {output_json}")

# === Main ===
if __name__ == "__main__":
    video_path = "../video/sample.mp4"
    audio_path = "output.wav"
    output_json = "whisper_transcription.json"
    model_path = "../../whisper.cpp/models/ggml-base.en.bin"

    extract_audio_with_ffmpeg(video_path, audio_path)
    result = transcribe_with_whisper_cpp(audio_path, model_path, output_json)

    print("\n=== Transcript ===")
    # Safely access and print each text block
    for segment in result.get("transcription", []):
        print(segment.get("text", ""))
