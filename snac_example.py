import torch
from snac import SNAC
import torchaudio
import soundfile as sf

# --- REPLACEMENT CODE ---
filepath = "00001_001_sales_000_1.wav"
target_sample_rate = 24000

# Load the audio file
waveform, sample_rate = torchaudio.load(filepath)

# Resample if necessary
if sample_rate != target_sample_rate:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    waveform = resampler(waveform)

# Ensure it's mono
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)

# Add a batch dimension and move to GPU
# The final shape should be (1, 1, T)
audio = waveform.unsqueeze(0).cuda()


model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

with torch.inference_mode():
    codes = model.encode(audio)
    audio_hat = model.decode(codes)

output_filepath = "snac.wav"
# --- 4. Save the Reconstructed Audio ---
print(f"Saving reconstructed audio to: {output_filepath}")
audio_to_save = audio_hat.squeeze().cpu().numpy()
sf.write(output_filepath, audio_to_save, target_sample_rate)
print("Done.")  