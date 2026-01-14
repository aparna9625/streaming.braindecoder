# synthesis/convert_vocoder_ckpt.py

'''
Docstring for synthesis.convert_vocoder_ckpt

python convert_vocoder_ckpt.py --lightning_ckpt outputs/2026-01-05/13-30-34/lightning_logs/version_0/checkpoints/best-epoch\=249-val_mel_loss\=0.66.ckpt --cfg configs/lj_hificar_dur.yaml --out_dir outputs/2026-01-05/13-30-34/converted_vocoder/

'''
import argparse
from pathlib import Path
import torch
import os, shutil, yaml
from omegaconf import OmegaConf
from src.vocoder_trainer import VocoderTrainer

parser = argparse.ArgumentParser()
parser.add_argument("--lightning_ckpt", required=True)
parser.add_argument("--cfg", required=True, help="Path to the vocoder YAML config used during training (e.g. synthesis/configs/lj_hificar_dur.yaml)")
parser.add_argument("--out_dir", required=True, help="Output folder for converted model (will contain best_mel_ckpt.pkl and config.yml)")
args = parser.parse_args()

ckpt = Path(args.lightning_ckpt).resolve()
cfg_path = Path(args.cfg).resolve()
out_dir = Path(args.out_dir).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

cfg = OmegaConf.load(str(cfg_path))
model = VocoderTrainer.load_from_checkpoint(checkpoint_path=str(ckpt), config=cfg, map_location="cpu", weights_only=False, strict=False)
gen_sd = model.generator.state_dict()

# Update num_emb in config to match the checkpoint !!!!!! (somehow the saved ckpt has 101 embedding size instead of 100??)
num_emb_actual = model.generator.emb_mat.weight.shape[0]
cfg["generator_params"]["num_emb"] = num_emb_actual

torch.save({"model": {"generator": gen_sd}}, out_dir / "best_mel_ckpt.pkl")

# save config.yml
with open(out_dir / "config.yml", "w") as f:
    yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f)

# place center-{num_emb}.npy where hificar_dur.load_model_eval expects it
num_emb = cfg["generator_params"]["num_emb"]
emb_p = cfg["generator_params"]["emb_p"]
emb_src = Path(emb_p)
if not emb_src.is_absolute():
    emb_src = cfg_path.parent.joinpath(emb_p).resolve()
if not emb_src.exists():
    print(f"Warning: emb file {emb_src} not found; please supply the embedding file.")
else:
    dest_dir = Path("/data/vocoder_ckpts/center")
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(emb_src), dest_dir / f"center-{num_emb}.npy")

print("Conversion done. Model dir:", out_dir)