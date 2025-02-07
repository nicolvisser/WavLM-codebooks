import torch
import torchaudio

wav, sr = torchaudio.load(
    "/mnt/wsl/nvme/datasets/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
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
