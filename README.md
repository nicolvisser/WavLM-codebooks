# K-means Codebooks for WavLM

A quick way to access the pretrained [WavLM Large](https://github.com/microsoft/unilm/tree/master/wavlm) model and a few pretrained $K$-means codebooks on some of the layers.

## Requirements

See `pyproject.toml`.

## Usage

```python
import torch
import torchaudio

wav, sr = torchaudio.load(
    "/path/to/wav.wav"
)
assert sr == 16000

wavlm = torch.hub.load(
    "nicolvisser/wavlm-codebooks",
    "wavlm_large",
    progress=True,
    trust_repo=True,
).cuda()
wavlm.eval()

codebook = torch.hub.load(
    "nicolvisser/wavlm-codebooks",
    "codebook",
    layer=11,
    k=500,
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

    distances = torch.cdist(features.cuda(), codebook, p=2)  # [T, K]
    units = torch.argmin(distances, dim=1)  # [T,]

print(units)

```

## More Information

The training script for the $K$-means codebooks can be found in `kmeans/train.py`.
More example scripts can be found in `example_scripts/`.
