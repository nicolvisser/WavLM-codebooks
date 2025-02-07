from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class WaveformWithOutputPathDataset(Dataset):
    def __init__(
        self,
        input_dir: Path,
        input_pattern: str,
        output_dir: Path,
        output_extension: str,
    ):
        input_paths = list(input_dir.glob(input_pattern))
        output_paths = [
            (output_dir / ip.relative_to(input_dir)).with_suffix(output_extension)
            for ip in input_paths
        ]
        self.input_paths = input_paths
        self.output_paths = output_paths

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_path = self.input_paths[idx]
        output_path = self.output_paths[idx]
        wav, sr = torchaudio.load(input_path)
        assert sr == 16000
        wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        return wav, output_path


if __name__ == "__main__":

    wavlm = torch.hub.load(
        "nicolvisser/wavlm-codebooks",
        "wavlm_large",
        map_location="cuda",
        progress=True,
        trust_repo=True,
    ).cuda()
    wavlm.eval()

    codebook = torch.hub.load(
        "nicolvisser/wavlm-codebooks",
        "codebook",
        layer=11,
        k=500,
        map_location="cuda",
        progress=True,
        trust_repo=True,
    ).cuda()

    dataset = WaveformWithOutputPathDataset(
        input_dir=Path("/mnt/wsl/nvme/datasets/LibriSpeech/dev-clean"),
        input_pattern="**/*.flac",
        output_dir=Path("./output/kmeans_units"),
        output_extension=".npy",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],
        num_workers=16,
    )

    with torch.inference_mode():
        for wav, output_path in tqdm(dataloader):

            features, _ = wavlm.extract_features(
                source=wav.cuda(),
                padding_mask=None,
                mask=False,
                ret_conv=False,
                output_layer=11,
                ret_layer_results=False,
            )
            features = features.squeeze(0)

            # get cluster assignments
            distances = torch.cdist(features.cuda(), codebook, p=2)
            units = torch.argmin(distances, dim=1)

            units = units.cpu().numpy()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, units)
