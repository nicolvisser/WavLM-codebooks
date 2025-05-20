dependencies = ["torch", "torchaudio"]

from typing import Callable, Tuple

import torch
import torchaudio

from wavlm.WavLM import WavLM

release_url = "https://github.com/nicolvisser/WavLM-codebooks/releases/download/v0.1.0/"

codebook_urls = {
    (11, 100): release_url + "codebook-layer-11-k-100-8b2b254e.pt",
    (11, 200): release_url + "codebook-layer-11-k-200-55b06314.pt",
    (11, 500): release_url + "codebook-layer-11-k-500-2c2dee95.pt",
    (11, 1000): release_url + "codebook-layer-11-k-1000-db31d361.pt",
    (11, 2000): release_url + "codebook-layer-11-k-2000-af7a6260.pt",
}


def wavlm_large(map_location="cpu", progress=True) -> WavLM:
    return WavLM.from_pretrained_url(
        release_url + "wavlm-large-6fb4b3c3.pt",
        map_location=map_location,
        progress=progress,
    )


def wavlm_for_dpslm(map_location="cpu", progress=True) -> Tuple[WavLM, Callable]:
    model = WavLM.from_pretrained_url(
        release_url + "wavlm-large-6fb4b3c3.pt",
        map_location=map_location,
        progress=progress,
    )
    model.eval()

    @torch.inference_mode()
    def extract_features(
        model: torch.nn.Module, wav: torch.Tensor, sr: int
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        assert wav.ndim == 2, "wav must be a 2D tensor, with shape (1, T)"
        wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        wav = wav.to(device)
        features, _ = model.extract_features(wav, output_layer=11)
        features = features.squeeze(0)
        return features

    return model, extract_features


def codebook(layer: int, k: int, map_location="cpu", progress=True) -> torch.Tensor:
    if (layer, k) not in codebook_urls:
        raise ValueError(
            f"Pretrained codebook for layer {layer} and k {k} not found. Available codebooks: {codebook_urls.keys()}"
        )
    state_dict = torch.hub.load_state_dict_from_url(
        codebook_urls[(layer, k)],
        map_location=map_location,
        progress=progress,
        check_hash=True,
        weights_only=True,
    )
    codebook = state_dict["codebook"]
    print(f"WavLM codebook loaded with shape: {codebook.shape}")
    return codebook
