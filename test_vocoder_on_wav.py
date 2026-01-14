# synthesis/test_vocoder_on_wav.py
import argparse
import soundfile as sf
import librosa
import torch
from model.tokenizer import HuBERTTokenizer
from synthesis.src.hificar_dur import HifiCarDurationSynthesizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True, help="Directory containing best_mel_ckpt.pkl and config.yml (the out_dir from conversion)")
parser.add_argument("--audio", required=True, help="Input audio WAV (any sr will be resampled to 16k)")
parser.add_argument("--out", default="vocoder_out.wav")
parser.add_argument("--device", default="cuda" )
parser.add_argument("--km_n", type=int, default=100)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--min_dur", type=int, default=1)
args = parser.parse_args()

wav, sr = sf.read(args.audio)
if sr != 16000:
    wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
device = args.device

tokenizer = HuBERTTokenizer(pre_tokenized=False, km_n=args.km_n, device=device, collapse=True)
tokens = tokenizer.tokenize(wav, sr=16000)  # returns numpy array of token ids
units = torch.tensor(tokens).long()

synth = HifiCarDurationSynthesizer(model_ckpt=str(args.model_dir), device=device, output_sr=16000)
synth.model.eval()
wav_out = synth.synthesize_v2(units, alpha=args.alpha, min_dur=args.min_dur)
if wav_out is None:
    raise SystemError("synthesis returned None")
sf.write(args.out, wav_out.numpy() if hasattr(wav_out, "numpy") else wav_out, 16000)
print("Saved:", args.out)