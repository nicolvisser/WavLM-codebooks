# $k$-means codebooks for WavLM

A quick way to access the pretrained [WavLM Large](https://github.com/microsoft/unilm/tree/master/wavlm) model and the pretrained $k$-means codebooks as used in https://github.com/nicolvisser/dp-slm.

## Usage

Install requirements. Use Python 3.10+ and PyTorch 2+.

```sh
pip install torch torchaudio torchvision
```

Load your audio:

```python
import torch
import torchaudio

wav, sr = torchaudio.load("your_audio.wav")
assert sr == 16000
```

Extract features:

```py
wavlm, extract_features = torch.hub.load(
    "nicolvisser/WavLM-codebooks",
    "wavlm_large",
    trust_repo=True,
)
wavlm.to("cuda")

features = extract_features(wavlm, wav, sr, layer=11) # [T, D]
```

Load codebook and quantize to nearest entry:

```py
codebook = torch.hub.load(
    "nicolvisser/WavLM-codebooks",
    "codebook",
    layer=11,
    k=500, # <- change k here
    progress=True,
    trust_repo=True,
).cuda() # [K, D]


distances = torch.cdist(features, codebook, p=2) # [T, K]
units = torch.argmin(distances, dim=1) # [T,]
```

## Pretrained codebooks available

| Layer | k    | Bit rate (bps) |
| ----- | ---- | -------------- |
| 11    | 100  | 192            |
| 11    | 200  | 243            |
| 11    | 500  | 320            |
| 11    | 1000 | 386            |
| 11    | 2000 | 414            |

## Training

The training script for the $K$-means codebooks can be found in `kmeans/train.py`.
You can find the dependencies for training in `pyproject.toml`.
```
