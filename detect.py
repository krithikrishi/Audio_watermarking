import torch
import soundfile as sf
from audioseal import AudioSeal
from torchaudio.transforms import Resample
import sys 
import csv # <-- ADDED

# --- 1. Load the Model ---
print("Loading AudioSeal detector model...")
try:
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

# --- 2. Load Your Watermarked File ---
input_filename = "watermarked_song.wav"
model_sample_rate = 16000

try:
    wav, sample_rate = sf.read(input_filename)
    print(f"Loaded '{input_filename}', sample rate: {sample_rate}")
except Exception as e:
    print(f"Error loading audio file: {e}")
    sys.exit()

if wav.ndim == 2:
    print("Audio is stereo. Converting to mono by averaging channels...")
    wav = wav.mean(axis=1)

wav_tensor = torch.tensor(wav, dtype=torch.float32)
wav_tensor = wav_tensor.reshape(1, 1, -1)

if sample_rate != model_sample_rate:
    print(f"Resampling audio from {sample_rate}Hz to {model_sample_rate}kHz...")
    resampler = Resample(orig_freq=sample_rate, new_freq=model_sample_rate)
    wav_tensor = resampler(wav_tensor)


# --- 3. Detect the Watermark ---
print("Detecting watermark...")
result, decoded_message = detector.detect_watermark(wav_tensor, sample_rate=model_sample_rate)

print("\n--- DETECTION RESULTS ---")
print(f"Watermark Detected: {result > 0.5} ")

if result > 0.5:
    # --- 4. Convert Bits to ID Number (CHANGED) ---
    
    # 1. Get the list of bits: [[0, 0, 1, ...]] -> [0, 0, 1, ...]
    bit_list = decoded_message.numpy().squeeze().tolist()
    
    # 2. Convert list of bits to a string: [0, 0, 1] -> "001"
    bit_string = "".join(map(str, bit_list))
    
    # 3. Convert binary string to a number: "00101010" -> 42
    message_id = int(bit_string, 2)
    
    print(f"Decoded Message ID: {message_id} (Bits: {bit_string})")

    # --- 5. Look up ID in Metadata Log ---
    try:
        found_metadata = False
        with open('metadata_log.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                # row[0] is the ID, row[1] is the metadata string
                if row[0] == str(message_id):
                    print(f"Full Metadata: {row[1]}")
                    found_metadata = True
                    break
        
        if not found_metadata:
            print(f"Found ID {message_id}, but no metadata was found in 'metadata_log.csv'.")

    except FileNotFoundError:
        print("Could not find 'metadata_log.csv' to look up ID.")
        
else:
    print("No watermark detected. No message to decode.")
