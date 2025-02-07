# $k$-means codebooks for WavLM

A quick way to access the pretrained [WavLM Large](https://github.com/microsoft/unilm/tree/master/wavlm) model and a few pretrained $k$-means codebooks on some of the layers.

## Requirements

Python 3.10+ and PyTorch 2+ are required.

For training, see the `dev` requirements in `pyproject.toml`.

## Usage

```python
import torch
import torchaudio

wav, sr = torchaudio.load(
    "/path/to/wav.wav"
)
assert sr == 16000

wavlm = torch.hub.load(
    "nicolvisser/WavLM-codebooks",
    "wavlm_large",
    progress=True,
    trust_repo=True,
).cuda()
wavlm.eval()

codebook = torch.hub.load(
    "nicolvisser/WavLM-codebooks",
    "codebook",
    layer=11,
    k=500, # <- change k here
    progress=True,
    trust_repo=True,
).cuda()

with torch.inference_mode():
    features, _ = wavlm.extract_features(
        source=wav.cuda(),
        padding_mask=None,
        mask=False,
        ret_conv=False,
        output_layer=11,
        ret_layer_results=False,
    )  # [1, T, D]
    features = features.squeeze(0)  # [T, D]

    distances = torch.cdist(features, codebook, p=2)  # [T, K]
    units = torch.argmin(distances, dim=1)  # [T,]

print(units)

```

## Pretrained codebooks available

| Layer | k    | Bit rate (bps) |
| ----- | ---- | -------------- |
| 11    | 100  | 192            |
| 11    | 200  | 243            |
| 11    | 500  | 320            |
| 11    | 1000 | 386            |
| 11    | 2000 | 414            |

## More Information

The training script for the $K$-means codebooks can be found in `kmeans/train.py`.
More example scripts can be found in `example_scripts/`.
