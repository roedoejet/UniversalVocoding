import torch
import soundfile as sf
import os
from univoc import Vocoder
from preprocess import melspectrogram, process_wav
import numpy as np
import json


with open(os.path.join('moh-out', 'train.json')) as f:
    data = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# download pretrained weights (and optionally move to GPU)
vocoder = Vocoder.from_pretrained(
    "https://github.com/bshall/UniversalVocoding/releases/download/v0.2/univoc-ljspeech-7mtpaq.pt"
)

# load log-Mel spectrogram from file or from tts (see https://github.com/bshall/Tacotron for example)
for item in data:
    fn, ext = os.path.splitext(item[1])
    mel = np.load(os.path.join('moh-out', f"{fn}.mel.npy"))
    mel = torch.FloatTensor(mel.T).unsqueeze(0).to(device)
    # generate waveform
    with torch.no_grad():
        wav, sr = vocoder.generate(mel)

    # save output
    sf.write(f"{fn}.wav", wav, sr)