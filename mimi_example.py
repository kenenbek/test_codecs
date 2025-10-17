from datasets import load_dataset, Audio
from transformers import MimiModel, AutoFeatureExtractor

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


model = MimiModel.from_pretrained("kyutai/mimi")
feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

# pre-process the inputs
inputs = feature_extractor(raw_audio=audio_sample, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

# explicitly encode then decode the audio inputs
encoder_outputs = model.encode(inputs["input_values"])
audio_values = model.decode(encoder_outputs.audio_codes)[0]

# or the equivalent with a forward pass
audio_values = model(inputs["input_values"]).audio_values
